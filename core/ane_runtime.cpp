// ane_runtime.cpp — ANE compile/load/eval wrapper + MIL kernel generation (pure C++)
// Uses _ANEInMemoryModel via private AppleNeuralEngine.framework through ObjC runtime C API
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <dlfcn.h>
#if __has_include(<IOSurface/IOSurface.h>)
#include <IOSurface/IOSurface.h>
#else
// iOS: IOSurface headers are split across multiple files
#include <IOSurface/IOSurfaceRef.h>
#include <IOSurface/IOSurfaceTypes.h>
#include <IOKit/IOReturn.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <ftw.h>
#include "ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

namespace ane_lm {

// ============ ObjC runtime helpers ============

static inline SEL    sel(const char* n) { return sel_registerName(n); }
static inline Class  cls(const char* n) { return (Class)objc_getClass(n); }

static id ns_str(const char* s) {
    return ((id(*)(Class,SEL,const char*))objc_msgSend)(cls("NSString"), sel("stringWithUTF8String:"), s);
}
static const char* to_cstr(id s) {
    if (!s) return "";
    return ((const char*(*)(id,SEL))objc_msgSend)(s, sel("UTF8String"));
}
static id ns_int(int v) {
    return ((id(*)(Class,SEL,int))objc_msgSend)(cls("NSNumber"), sel("numberWithInt:"), v);
}
static id ns_ulong(unsigned long v) {
    return ((id(*)(Class,SEL,unsigned long))objc_msgSend)(cls("NSNumber"), sel("numberWithUnsignedLong:"), v);
}
static id ns_data_nocopy(void* p, unsigned long len) {
    return ((id(*)(Class,SEL,void*,unsigned long,bool))objc_msgSend)(
        cls("NSData"), sel("dataWithBytesNoCopy:length:freeWhenDone:"), p, len, true);
}
static id ns_data(const void* p, unsigned long len) {
    return ((id(*)(Class,SEL,const void*,unsigned long))objc_msgSend)(
        cls("NSData"), sel("dataWithBytes:length:"), p, len);
}
static id ns_dict(id* keys, id* values, unsigned long count) {
    return ((id(*)(Class,SEL,id*,id*,unsigned long))objc_msgSend)(
        cls("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"), values, keys, count);
}
static id ns_empty_dict() {
    return ((id(*)(Class,SEL))objc_msgSend)(cls("NSDictionary"), sel("dictionary"));
}
static id ns_mutable_array(unsigned long cap) {
    return ((id(*)(Class,SEL,unsigned long))objc_msgSend)(
        cls("NSMutableArray"), sel("arrayWithCapacity:"), cap);
}
static void ns_array_add(id arr, id obj) {
    ((void(*)(id,SEL,id))objc_msgSend)(arr, sel("addObject:"), obj);
}
static id objc_retain_obj(id o) { return ((id(*)(id,SEL))objc_msgSend)(o, sel("retain")); }
static void objc_release_obj(id o) { if (o) ((void(*)(id,SEL))objc_msgSend)(o, sel("release")); }

// ============ C file helpers ============

static void mkdir_p(const std::string& path) {
    std::string tmp;
    for (size_t i = 0; i < path.size(); i++) {
        tmp += path[i];
        if (path[i] == '/' || i == path.size() - 1)
            mkdir(tmp.c_str(), 0755);
    }
}
static void write_file(const std::string& path, const void* data, size_t len) {
    FILE* f = fopen(path.c_str(), "wb");
    if (f) { fwrite(data, 1, len, f); fclose(f); }
}
static bool file_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}
static int nftw_rm_cb(const char* fpath, const struct stat*, int, struct FTW*) {
    return remove(fpath);
}
static void remove_dir(const std::string& path) {
    nftw(path.c_str(), nftw_rm_cb, 64, FTW_DEPTH | FTW_PHYS);
}

// ============ ANEKernel struct ============

struct ANEKernel {
    id model;           // _ANEInMemoryModel (retained)
    IOSurfaceRef* ioInputs;
    IOSurfaceRef* ioOutputs;
    id request;         // _ANERequest (retained)
    std::string tmpDir;
    int nInputs, nOutputs;
    size_t* inputBytes;
    size_t* outputBytes;
};

// ============ Global state ============

static Class g_ANEDesc = nullptr, g_ANEInMem = nullptr, g_ANEReq = nullptr, g_ANEIO = nullptr;
static bool g_ane_ok = false;
static int g_compile_count = 0;
static bool g_ane_persist_cache = true;
static int g_ane_cache_load_count = 0;

void ane_set_persist_cache(bool enabled) { g_ane_persist_cache = enabled; }
int ane_compile_count() { return g_compile_count; }
int ane_cache_loads() { return g_ane_cache_load_count; }

static std::string g_marker_root;
static std::once_flag g_marker_once;

static const std::string& ane_marker_root_dir() {
    std::call_once(g_marker_once, []() {
        const char* home = getenv("HOME");
        g_marker_root = std::string(home ? home : "/tmp") + "/Library/Caches/ane_lm/compiled_markers";
    });
    return g_marker_root;
}

static void ane_remove_compile_dir(const std::string& td, bool force_remove) {
    if (force_remove || !g_ane_persist_cache) remove_dir(td);
}

static std::once_flag g_ane_once;

void ane_init() {
    std::call_once(g_ane_once, []() {
        void* handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) {
            fprintf(stderr, "Warning: Failed to load AppleNeuralEngine.framework: %s\n", dlerror());
            return;
        }
        g_ANEDesc  = cls("_ANEInMemoryModelDescriptor");
        g_ANEInMem = cls("_ANEInMemoryModel");
        g_ANEReq   = cls("_ANERequest");
        g_ANEIO    = cls("_ANEIOSurfaceObject");
        if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
            fprintf(stderr, "Warning: ANE private classes not found\n");
            g_ANEDesc = g_ANEInMem = g_ANEReq = g_ANEIO = nullptr;
            return;
        }
        g_ane_ok = true;
    });
}

bool ane_available() { ane_init(); return g_ane_ok; }

// ============ IOSurface helpers ============

static IOSurfaceRef ane_create_surface(size_t bytes) {
    if (bytes == 0) bytes = 4;
    id keys[] = {
        (id)kIOSurfaceWidth, (id)kIOSurfaceHeight, (id)kIOSurfaceBytesPerElement,
        (id)kIOSurfaceBytesPerRow, (id)kIOSurfaceAllocSize, (id)kIOSurfacePixelFormat
    };
    id values[] = {
        ns_ulong(bytes), ns_int(1), ns_int(1),
        ns_ulong(bytes), ns_ulong(bytes), ns_int(0)
    };
    id dict = ns_dict(keys, values, 6);
    return IOSurfaceCreate((CFDictionaryRef)dict);
}

static bool ane_zero_surface(IOSurfaceRef surface) {
    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) {
        fprintf(stderr, "ANE: IOSurfaceLock failed while zeroing surface\n");
        return false;
    }
    memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface));
    IOSurfaceUnlock(surface, 0, NULL);
    return true;
}

// ============ Weight blob builder ============

static id build_weight_blob(const uint16_t* bf16_data, int num_elements) {
    size_t wsize = (size_t)num_elements * 2;
    size_t total = 64 + 64 + wsize;
    uint8_t* buf = (uint8_t*)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;

    uint16_t* fp16 = (uint16_t*)(buf + 128);
    bf16_to_f16_vec(fp16, bf16_data, num_elements);

    return ns_data_nocopy(buf, total);
}

static id ns_weight_entry(id blob) {
    id keys[]   = { ns_str("offset"), ns_str("data") };
    id values[] = { ns_int(0), blob };
    return ns_dict(keys, values, 2);
}

static id build_weight_dict_1(const uint16_t* bf16, int numel, const char* name) {
    id blob = build_weight_blob(bf16, numel);
    char kbuf[128]; snprintf(kbuf, sizeof(kbuf), "@model_path/weights/%s.bin", name);
    id k = ns_str(kbuf); id v = ns_weight_entry(blob);
    return ns_dict(&k, &v, 1);
}

static id build_weight_dict_2(
    const uint16_t* bf16_a, int numel_a, const char* name_a,
    const uint16_t* bf16_b, int numel_b, const char* name_b)
{
    id ba = build_weight_blob(bf16_a, numel_a);
    id bb = build_weight_blob(bf16_b, numel_b);
    char ka[128], kb[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    id keys[]   = { ns_str(ka), ns_str(kb) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb) };
    return ns_dict(keys, values, 2);
}

static id build_weight_dict_3(
    const uint16_t* bf16_a, int numel_a, const char* name_a,
    const uint16_t* bf16_b, int numel_b, const char* name_b,
    const uint16_t* bf16_c, int numel_c, const char* name_c)
{
    id ba = build_weight_blob(bf16_a, numel_a);
    id bb = build_weight_blob(bf16_b, numel_b);
    id bc = build_weight_blob(bf16_c, numel_c);
    char ka[128], kb[128], kc[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    snprintf(kc, sizeof(kc), "@model_path/weights/%s.bin", name_c);
    id keys[]   = { ns_str(ka), ns_str(kb), ns_str(kc) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb), ns_weight_entry(bc) };
    return ns_dict(keys, values, 3);
}

// ============ MIL program generation ============

#define MIL_HEADER \
    "program(1.0)\n" \
    "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n" \
    "{\n"

#define SP ANE_SPATIAL

static id mil_gen_matmul(int out_dim, int in_dim) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = tensor<string, []>(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = W, x = x)"
        "[name = tensor<string, []>(\"cv\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP,
        out_dim, in_dim, out_dim, in_dim,
        out_dim, SP);
    return ns_data(buf, n);
}

static id mil_gen_fused_2(int a_out, int b_out, int in_dim) {
    int total_out = a_out + b_out;
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wa = const()[name = tensor<string, []>(\"Wa\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wa.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wb = const()[name = tensor<string, []>(\"Wb\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wb.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> ya = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wa, x = x)"
        "[name = tensor<string, []>(\"ca\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> yb = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wb, x = x)"
        "[name = tensor<string, []>(\"cb\")];\n"
        "        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n"
        "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = concat(values = (ya, yb), axis = ax, "
        "interleave = ci)[name = tensor<string, []>(\"cc\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP,
        a_out, in_dim, a_out, in_dim,
        b_out, in_dim, b_out, in_dim,
        a_out, SP,
        b_out, SP,
        total_out, SP);
    return ns_data(buf, n);
}

static id mil_gen_fused_3(int a_out, int b_out, int c_out, int in_dim) {
    int total_out = a_out + b_out + c_out;
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wa = const()[name = tensor<string, []>(\"Wa\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wa.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wb = const()[name = tensor<string, []>(\"Wb\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wb.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wc = const()[name = tensor<string, []>(\"Wc\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wc.bin\"), "
        "offset = tensor<uint64, []>(64)))];\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> ya = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wa, x = x)"
        "[name = tensor<string, []>(\"ca\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> yb = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wb, x = x)"
        "[name = tensor<string, []>(\"cb\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> yc = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wc, x = x)"
        "[name = tensor<string, []>(\"cc\")];\n"
        "        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n"
        "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = concat(values = (ya, yb, yc), axis = ax, "
        "interleave = ci)[name = tensor<string, []>(\"ct\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP,
        a_out, in_dim, a_out, in_dim,
        b_out, in_dim, b_out, in_dim,
        c_out, in_dim, c_out, in_dim,
        a_out, SP,
        b_out, SP,
        c_out, SP,
        total_out, SP);
    return ns_data(buf, n);
}

static id mil_gen_fused_ffn(int dim, int inter_ch) {
    char buf[8192];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wg = const()[name = tensor<string, []>(\"Wg\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wg.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wu = const()[name = tensor<string, []>(\"Wu\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wu.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = tensor<string, []>(\"Wd\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/wd.bin\"), offset = tensor<uint64, []>(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wg, x = x)[name = tensor<string, []>(\"cg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wu, x = x)[name = tensor<string, []>(\"cu\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr, pad = pd, "
        "pad_type = pt, strides = st, weight = Wd, x = fused)[name = tensor<string, []>(\"cd\")];\n"
        "    } -> (out);\n"
        "}\n",
        dim, SP,
        inter_ch, dim, inter_ch, dim,
        inter_ch, dim, inter_ch, dim,
        dim, inter_ch, dim, inter_ch,
        inter_ch, SP, inter_ch, SP,
        inter_ch, SP, inter_ch, SP, inter_ch, SP,
        dim, SP);
    return ns_data(buf, n);
}

// ============ Core compile/eval/free ============

static ANEKernel* ane_compile_raw(id milText, id wdict,
                                   int nInputs, size_t* inputSizes,
                                   int nOutputs, size_t* outputSizes) {
    if (!ane_available()) return nullptr;

    void* local_pool = objc_autoreleasePoolPush();

    // Create descriptor
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, sel("modelWithMILText:weights:optionsPlist:"),
        milText, wdict ? wdict : ns_empty_dict(), (id)nullptr);
    if (!desc) {
        fprintf(stderr, "ANE: modelWithMILText failed\n");
        objc_autoreleasePoolPop(local_pool);
        return nullptr;
    }

    // Create in-memory model
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, sel("inMemoryModelWithDescriptor:"), desc);
    if (!mdl) {
        fprintf(stderr, "ANE: inMemoryModelWithDescriptor returned nil\n");
        objc_autoreleasePoolPop(local_pool);
        return nullptr;
    }

    // Get hex identifier for cache key
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, sel("hexStringIdentifier"));
    std::string modelId;
    if (hx) {
        bool isStr = ((bool(*)(id,SEL,Class))objc_msgSend)(hx, sel("isKindOfClass:"), cls("NSString"));
        if (isStr) {
            unsigned long len = ((unsigned long(*)(id,SEL))objc_msgSend)(hx, sel("length"));
            if (len > 0) modelId = to_cstr(hx);
        }
    }
    if (modelId.empty()) {
        id uuid = ((id(*)(Class,SEL))objc_msgSend)(cls("NSUUID"), sel("UUID"));
        id uuidStr = ((id(*)(id,SEL))objc_msgSend)(uuid, sel("UUIDString"));
        modelId = to_cstr(uuidStr);
    }

    const char* tmpenv = getenv("TMPDIR");
    std::string td = std::string(tmpenv ? tmpenv : "/tmp") + "/" + modelId;
    const std::string& markerRoot = ane_marker_root_dir();
    mkdir_p(markerRoot);
    std::string compiledMarker = markerRoot + "/" + modelId + ".ok";
    mkdir_p(td + "/weights");

    // Write MIL text to file
    const void* milBytes = ((const void*(*)(id,SEL))objc_msgSend)(milText, sel("bytes"));
    unsigned long milLen = ((unsigned long(*)(id,SEL))objc_msgSend)(milText, sel("length"));
    write_file(td + "/model.mil", milBytes, milLen);

    // Write weight files
    if (wdict) {
        id allKeys = ((id(*)(id,SEL))objc_msgSend)(wdict, sel("allKeys"));
        unsigned long keyCount = ((unsigned long(*)(id,SEL))objc_msgSend)(allKeys, sel("count"));
        for (unsigned long i = 0; i < keyCount; i++) {
            id key = ((id(*)(id,SEL,unsigned long))objc_msgSend)(allKeys, sel("objectAtIndex:"), i);
            std::string keyStr = to_cstr(key);
            std::string relPath = keyStr;
            size_t pos = relPath.find("@model_path/");
            if (pos != std::string::npos) relPath.erase(pos, strlen("@model_path/"));
            std::string fullPath = td + "/" + relPath;

            id entry = ((id(*)(id,SEL,id))objc_msgSend)(wdict, sel("objectForKey:"), key);
            id data = ((id(*)(id,SEL,id))objc_msgSend)(entry, sel("objectForKey:"), ns_str("data"));
            const void* dataBytes = ((const void*(*)(id,SEL))objc_msgSend)(data, sel("bytes"));
            unsigned long dataLen = ((unsigned long(*)(id,SEL))objc_msgSend)(data, sel("length"));
            write_file(fullPath, dataBytes, dataLen);
        }
    }

    // Cache check / compile / load
    id e = nullptr;
    bool loaded_from_cache = false;
    if (g_ane_persist_cache && file_exists(compiledMarker)) {
        e = nullptr;
        bool ok = ((bool(*)(id,SEL,unsigned int,id,id*))objc_msgSend)(
            mdl, sel("loadWithQoS:options:error:"), 21, ns_empty_dict(), &e);
        if (ok) {
            loaded_from_cache = true;
            g_ane_cache_load_count++;
        } else {
            remove(compiledMarker.c_str());
            e = nullptr;
        }
    }

    if (!loaded_from_cache) {
        e = nullptr;
        if (!((bool(*)(id,SEL,unsigned int,id,id*))objc_msgSend)(
                mdl, sel("compileWithQoS:options:error:"), 21, ns_empty_dict(), &e)) {
            fprintf(stderr, "ANE compile failed: %s\n",
                e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
            remove(compiledMarker.c_str());
            ane_remove_compile_dir(td, true);
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
        e = nullptr;
        if (!((bool(*)(id,SEL,unsigned int,id,id*))objc_msgSend)(
                mdl, sel("loadWithQoS:options:error:"), 21, ns_empty_dict(), &e)) {
            fprintf(stderr, "ANE load failed: %s\n",
                e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
            remove(compiledMarker.c_str());
            ane_remove_compile_dir(td, true);
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
        g_compile_count++;
        if (g_ane_persist_cache) {
            write_file(compiledMarker, "ok", 2);
        } else {
            remove(compiledMarker.c_str());
        }
    }

    // Create kernel struct
    ANEKernel* k = new ANEKernel();
    k->model = objc_retain_obj(mdl);
    k->tmpDir = td;
    k->nInputs = nInputs;
    k->nOutputs = nOutputs;
    k->inputBytes = (size_t*)malloc(nInputs * sizeof(size_t));
    k->outputBytes = (size_t*)malloc(nOutputs * sizeof(size_t));
    memcpy(k->inputBytes, inputSizes, nInputs * sizeof(size_t));
    memcpy(k->outputBytes, outputSizes, nOutputs * sizeof(size_t));

    // Create IOSurfaces
    k->ioInputs = (IOSurfaceRef*)malloc(nInputs * sizeof(IOSurfaceRef));
    k->ioOutputs = (IOSurfaceRef*)malloc(nOutputs * sizeof(IOSurfaceRef));
    for (int i = 0; i < nInputs; i++) {
        k->ioInputs[i] = ane_create_surface(inputSizes[i]);
        if (!k->ioInputs[i] || !ane_zero_surface(k->ioInputs[i])) {
            fprintf(stderr, "ANE: failed to init input IOSurface %d\n", i);
            delete k;
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
    }
    for (int i = 0; i < nOutputs; i++) {
        k->ioOutputs[i] = ane_create_surface(outputSizes[i]);
        if (!k->ioOutputs[i] || !ane_zero_surface(k->ioOutputs[i])) {
            fprintf(stderr, "ANE: failed to init output IOSurface %d\n", i);
            delete k;
            objc_autoreleasePoolPop(local_pool);
            return nullptr;
        }
    }

    // Create ANE request
    id wIns = ns_mutable_array(nInputs);
    id iIdx = ns_mutable_array(nInputs);
    for (int i = 0; i < nInputs; i++) {
        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, sel("objectWithIOSurface:"), k->ioInputs[i]);
        ns_array_add(wIns, ioObj);
        ns_array_add(iIdx, ns_int(i));
    }
    id wOuts = ns_mutable_array(nOutputs);
    id oIdx = ns_mutable_array(nOutputs);
    for (int i = 0; i < nOutputs; i++) {
        id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, sel("objectWithIOSurface:"), k->ioOutputs[i]);
        ns_array_add(wOuts, ioObj);
        ns_array_add(oIdx, ns_int(i));
    }
    k->request = objc_retain_obj(
        ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq, sel("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            wIns, iIdx, wOuts, oIdx, (id)nullptr, (id)nullptr, ns_int(0)));

    objc_autoreleasePoolPop(local_pool);
    return k;
}

static bool ane_eval_raw(ANEKernel* k) {
    id e = nullptr;
    bool ok = ((bool(*)(id,SEL,unsigned int,id,id,id*))objc_msgSend)(
        k->model, sel("evaluateWithQoS:options:request:error:"),
        21, ns_empty_dict(), k->request, &e);
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n",
            e ? to_cstr(((id(*)(id,SEL))objc_msgSend)(e, sel("description"))) : "unknown");
    }
    return ok;
}

// ============ Public API implementations ============

#if defined(__aarch64__) || defined(__arm64__)
typedef __fp16 ane_fp16_t;
#define ANE_USE_NATIVE_FP16 1
#else
#define ANE_USE_NATIVE_FP16 0
#endif

bool ane_matvec(ANEKernel* k, float* output, const float* input, int in_dim, int out_dim) {
    IOSurfaceRef in_surface = k->ioInputs[0];
    if (IOSurfaceLock(in_surface, 0, NULL) != kIOReturnSuccess) {
        fprintf(stderr, "ANE: IOSurfaceLock(input) failed\n");
        return false;
    }
    uint16_t* in_base = (uint16_t*)IOSurfaceGetBaseAddress(in_surface);
#if ANE_USE_NATIVE_FP16
    ane_fp16_t* in_base_h = (ane_fp16_t*)in_base;
#endif
#pragma clang loop vectorize(enable)
    for (int c = 0, idx = 0; c < in_dim; c++, idx += ANE_SPATIAL) {
#if ANE_USE_NATIVE_FP16
        in_base_h[idx] = (ane_fp16_t)input[c];
#else
        in_base[idx] = f32_to_f16(input[c]);
#endif
    }
    IOSurfaceUnlock(in_surface, 0, NULL);

    if (!ane_eval_raw(k)) return false;

    IOSurfaceRef out_surface = k->ioOutputs[0];
    if (IOSurfaceLock(out_surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        fprintf(stderr, "ANE: IOSurfaceLock(output) failed\n");
        return false;
    }
    const uint16_t* out_base = (const uint16_t*)IOSurfaceGetBaseAddress(out_surface);
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* out_base_h = (const ane_fp16_t*)out_base;
#endif
#pragma clang loop vectorize(enable)
    for (int c = 0, idx = 0; c < out_dim; c++, idx += ANE_SPATIAL) {
#if ANE_USE_NATIVE_FP16
        output[c] = (float)out_base_h[idx];
#else
        output[c] = f16_to_f32(out_base[idx]);
#endif
    }
    IOSurfaceUnlock(out_surface, kIOSurfaceLockReadOnly, NULL);

    return true;
}

bool ane_matvec_batch(ANEKernel* k, float* output, const float* input,
                      int in_dim, int out_dim, int batch_size) {
    if (batch_size <= 0 || batch_size > ANE_SPATIAL) return false;
    if (batch_size == 1) return ane_matvec(k, output, input, in_dim, out_dim);

    // Write input: pack batch_size vectors into spatial positions 0..B-1
    IOSurfaceRef in_surface = k->ioInputs[0];
    if (IOSurfaceLock(in_surface, 0, NULL) != kIOReturnSuccess) {
        fprintf(stderr, "ANE: IOSurfaceLock(input) failed in batch\n");
        return false;
    }
    void* in_addr = IOSurfaceGetBaseAddress(in_surface);
    memset(in_addr, 0, IOSurfaceGetAllocSize(in_surface));
#if ANE_USE_NATIVE_FP16
    ane_fp16_t* in_base_h = (ane_fp16_t*)in_addr;
    for (int b = 0; b < batch_size; b++) {
        const float* inp = input + (size_t)b * in_dim;
        for (int c = 0; c < in_dim; c++) {
            in_base_h[c * ANE_SPATIAL + b] = (ane_fp16_t)inp[c];
        }
    }
#else
    uint16_t* in_base = (uint16_t*)in_addr;
    for (int b = 0; b < batch_size; b++) {
        const float* inp = input + (size_t)b * in_dim;
        for (int c = 0; c < in_dim; c++) {
            in_base[c * ANE_SPATIAL + b] = f32_to_f16(inp[c]);
        }
    }
#endif
    IOSurfaceUnlock(in_surface, 0, NULL);

    if (!ane_eval_raw(k)) return false;

    // Read output: unpack batch_size vectors from spatial positions
    IOSurfaceRef out_surface = k->ioOutputs[0];
    if (IOSurfaceLock(out_surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        fprintf(stderr, "ANE: IOSurfaceLock(output) failed in batch\n");
        return false;
    }
#if ANE_USE_NATIVE_FP16
    const ane_fp16_t* out_base_h = (const ane_fp16_t*)IOSurfaceGetBaseAddress(out_surface);
    for (int b = 0; b < batch_size; b++) {
        float* outp = output + (size_t)b * out_dim;
        for (int c = 0; c < out_dim; c++) {
            outp[c] = (float)out_base_h[c * ANE_SPATIAL + b];
        }
    }
#else
    const uint16_t* out_base = (const uint16_t*)IOSurfaceGetBaseAddress(out_surface);
    for (int b = 0; b < batch_size; b++) {
        float* outp = output + (size_t)b * out_dim;
        for (int c = 0; c < out_dim; c++) {
            outp[c] = (float)f16_to_f32(out_base[c * ANE_SPATIAL + b]);
        }
    }
#endif
    IOSurfaceUnlock(out_surface, kIOSurfaceLockReadOnly, NULL);

    return true;
}

void ane_free(ANEKernel* k) {
    if (!k) return;
    id e = nullptr;
    ((bool(*)(id,SEL,unsigned int,id*))objc_msgSend)(
        k->model, sel("unloadWithQoS:error:"), 21, &e);
    for (int i = 0; i < k->nInputs; i++) CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) CFRelease(k->ioOutputs[i]);
    if (!g_ane_persist_cache) {
        remove_dir(k->tmpDir);
    }
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    objc_release_obj(k->request);
    objc_release_obj(k->model);
    delete k;
}

void ane_free_layer(LayerANEKernels* lk) {
    ane_free(lk->first_proj);
    ane_free(lk->o_proj);
    ane_free(lk->fused_ffn);
    lk->first_proj = lk->o_proj = lk->fused_ffn = nullptr;
}

// ============ High-level compile functions ============

ANEKernel* ane_compile_matmul(const uint16_t* bf16_weights, int out_dim, int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = build_weight_dict_1(bf16_weights, out_dim * in_dim, "weight");
    id mil = mil_gen_matmul(out_dim, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_2(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = build_weight_dict_2(bf16_a, a_out * in_dim, "wa",
                                    bf16_b, b_out * in_dim, "wb");
    id mil = mil_gen_fused_2(a_out, b_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_3(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                const uint16_t* bf16_c, int c_out,
                                int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = build_weight_dict_3(bf16_a, a_out * in_dim, "wa",
                                    bf16_b, b_out * in_dim, "wb",
                                    bf16_c, c_out * in_dim, "wc");
    id mil = mil_gen_fused_3(a_out, b_out, c_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out + c_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_ffn(const uint16_t* gate_bf16, const uint16_t* up_bf16,
                                  const uint16_t* down_bf16, int dim, int inter_ch) {
    void* pool = objc_autoreleasePoolPush();
    id wg = build_weight_blob(gate_bf16, inter_ch * dim);
    id wu = build_weight_blob(up_bf16, inter_ch * dim);
    id wd = build_weight_blob(down_bf16, dim * inter_ch);

    id keys[]   = { ns_str("@model_path/weights/wg.bin"),
                    ns_str("@model_path/weights/wu.bin"),
                    ns_str("@model_path/weights/wd.bin") };
    id values[] = { ns_weight_entry(wg), ns_weight_entry(wu), ns_weight_entry(wd) };
    id wdict = ns_dict(keys, values, 3);

    id mil = mil_gen_fused_ffn(dim, inter_ch);
    size_t in_size = (size_t)dim * SP * sizeof(uint16_t);
    size_t out_size = (size_t)dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_size, 1, &out_size);
    objc_autoreleasePoolPop(pool);
    return r;
}

// ============ Blob file loading ============

static id load_blob_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ANE: cannot open blob %s\n", path.c_str());
        return (id)nullptr;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc(len);
    fread(buf, 1, len, f);
    fclose(f);
    return ns_data_nocopy(buf, len);
}

static id blob_weight_dict_1(const std::string& path, const char* name) {
    id blob = load_blob_file(path);
    if (!blob) return (id)nullptr;
    char kbuf[128]; snprintf(kbuf, sizeof(kbuf), "@model_path/weights/%s.bin", name);
    id k = ns_str(kbuf); id v = ns_weight_entry(blob);
    return ns_dict(&k, &v, 1);
}

static id blob_weight_dict_2(const std::string& a_path, const char* name_a,
                              const std::string& b_path, const char* name_b) {
    id ba = load_blob_file(a_path);
    id bb = load_blob_file(b_path);
    if (!ba || !bb) return (id)nullptr;
    char ka[128], kb[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    id keys[]   = { ns_str(ka), ns_str(kb) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb) };
    return ns_dict(keys, values, 2);
}

static id blob_weight_dict_3(const std::string& a_path, const char* name_a,
                              const std::string& b_path, const char* name_b,
                              const std::string& c_path, const char* name_c) {
    id ba = load_blob_file(a_path);
    id bb = load_blob_file(b_path);
    id bc = load_blob_file(c_path);
    if (!ba || !bb || !bc) return (id)nullptr;
    char ka[128], kb[128], kc[128];
    snprintf(ka, sizeof(ka), "@model_path/weights/%s.bin", name_a);
    snprintf(kb, sizeof(kb), "@model_path/weights/%s.bin", name_b);
    snprintf(kc, sizeof(kc), "@model_path/weights/%s.bin", name_c);
    id keys[]   = { ns_str(ka), ns_str(kb), ns_str(kc) };
    id values[] = { ns_weight_entry(ba), ns_weight_entry(bb), ns_weight_entry(bc) };
    return ns_dict(keys, values, 3);
}

// ============ High-level compile from blob files ============

ANEKernel* ane_compile_matmul_blob(const std::string& blob_path, int out_dim, int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = blob_weight_dict_1(blob_path, "weight");
    if (!wdict) { objc_autoreleasePoolPop(pool); return nullptr; }
    id mil = mil_gen_matmul(out_dim, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_2_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = blob_weight_dict_2(a_path, "wa", b_path, "wb");
    if (!wdict) { objc_autoreleasePoolPop(pool); return nullptr; }
    id mil = mil_gen_fused_2(a_out, b_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_3_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     const std::string& c_path, int c_out,
                                     int in_dim) {
    void* pool = objc_autoreleasePoolPush();
    id wdict = blob_weight_dict_3(a_path, "wa", b_path, "wb", c_path, "wc");
    if (!wdict) { objc_autoreleasePoolPop(pool); return nullptr; }
    id mil = mil_gen_fused_3(a_out, b_out, c_out, in_dim);
    size_t in_bytes = (size_t)in_dim * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)(a_out + b_out + c_out) * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_bytes, 1, &out_bytes);
    objc_autoreleasePoolPop(pool);
    return r;
}

ANEKernel* ane_compile_fused_ffn_blob(const std::string& gate_path, const std::string& up_path,
                                       const std::string& down_path, int dim, int inter_ch) {
    void* pool = objc_autoreleasePoolPush();
    id wg = load_blob_file(gate_path);
    id wu = load_blob_file(up_path);
    id wd = load_blob_file(down_path);
    if (!wg || !wu || !wd) { objc_autoreleasePoolPop(pool); return nullptr; }

    id keys[]   = { ns_str("@model_path/weights/wg.bin"),
                    ns_str("@model_path/weights/wu.bin"),
                    ns_str("@model_path/weights/wd.bin") };
    id values[] = { ns_weight_entry(wg), ns_weight_entry(wu), ns_weight_entry(wd) };
    id wdict = ns_dict(keys, values, 3);

    id mil = mil_gen_fused_ffn(dim, inter_ch);
    size_t in_size = (size_t)dim * SP * sizeof(uint16_t);
    size_t out_size = (size_t)dim * SP * sizeof(uint16_t);
    ANEKernel* r = ane_compile_raw(mil, wdict, 1, &in_size, 1, &out_size);
    objc_autoreleasePoolPop(pool);
    return r;
}

} // namespace ane_lm
