// Static member initialization for Timer
#include <ane_lm/common.h>

namespace ane_lm {

bool g_verbose = false;
mach_timebase_info_data_t Timer::tb = {0, 0};
bool Timer::tb_init = false;

} // namespace ane_lm
