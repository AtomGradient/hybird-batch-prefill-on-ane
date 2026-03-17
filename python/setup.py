"""Setup script for ane_lm Python bindings.

Build and install:
    pip install ./python/

Or develop mode:
    pip install -e ./python/
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.fspath(Path(self.get_ext_fullpath(ext.name)).parent.resolve())

        # Get pybind11 cmake dir
        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_temp = os.fspath(Path(self.build_temp) / ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)],
            cwd=build_temp,
            check=True,
        )


setup(
    name="ane-lm",
    version="0.1.0",
    description="ANE-LM: Neural Engine batch prefill inference for Qwen3.5/Qwen3",
    author="AtomGradient",
    ext_modules=[CMakeExtension("ane_lm", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.11",
)
