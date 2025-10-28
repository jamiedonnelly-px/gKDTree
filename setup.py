import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import setuptools
from setuptools.command.build_ext import build_ext

LOGGER = logging.getLogger(__name__)

# constants to use
cwd = Path(__file__).parent.absolute()
CUDA_DEFAULT_PATH = "/usr/local/cuda"
ALLOWED_ARCH = ["x86_64"]
API_NAME="gKDTree"

def _find_cuda() -> str:
    """Attempts to find a NVCC binary to compile the source files with."""
    # Check NVCC_PATH environment variable first
    if nvcc_path := os.environ.get("NVCC_PATH"):
        if os.path.exists(nvcc_path):
            return nvcc_path
        else:
            LOGGER.warning(f"NVCC_PATH points to non-existent file: {nvcc_path}")
    
    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME', CUDA_DEFAULT_PATH)
    nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
    
    if os.path.exists(nvcc_path):
        return nvcc_path
    
    raise RuntimeError(f"NVCC not found at {nvcc_path}. "
                      "Please install CUDA toolkit or set NVCC_PATH environment variable.")

def _py_ver() -> str:
    """ Find current python version in form {major}.{minor}"""
    ver_info = sys.version_info
    return f"{ver_info.major}_{ver_info.minor}"

def _cuda_ver() -> str:
    """ Find current python version in form {major}.{minor}"""
    try:
        nvidia_smi_output = subprocess.run("nvidia-smi --version", shell=True, check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError(f"Unable to find cuda on the current system: {e}")
    try:
        cuda_ver_string = nvidia_smi_output.stdout.split("CUDA Version")[-1]
        cuda_ver = re.search("[+-]?([0-9]*[.])?[0-9]+", cuda_ver_string).group()
    except Exception:
        raise RuntimeError(f"Was unable to find a version format from output: {nvidia_smi_output}.")

    return ".".join(cuda_ver.split(".")[:2]).replace(".", "_")

def _find_precompiled() -> str:
    """ Attempts to find precompiled binary for current system. """
    # Check architecture
    arch = platform.machine()
    if arch not in ALLOWED_ARCH:
        raise RuntimeError(f"Architecture {arch} unsupported. \
                           Only {ALLOWED_ARCH} currently supported.")
    # Find versions
    py_ver, cuda_ver = _py_ver(), _cuda_ver()
    binary_name = f"_internal_cuda{cuda_ver}-py{py_ver}_{arch}.so"
    precompiled_dir = cwd / API_NAME / "precompiled_binaries"
    if (binary := precompiled_dir / binary_name).is_file():
        return binary
    raise RuntimeError(f"Was unable to find supported binary: {binary_name}.")

class CMakeExtension(setuptools.Extension):
    def __init__(self, name, sourcedir="", cmake_args=(), exclude_arch=False):
        setuptools.Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.cmake_args = cmake_args
        self.exclude_arch = exclude_arch

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        # check for CPU-only build
        if int(os.environ.get("FORCE_CPU_ONLY", 0)):
            LOGGER.info(f"Using CPU-only version of the package.")
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Create the build directory
        os.makedirs(extdir, exist_ok=True)

        # Attempt to find precompiled binary
        try:
            precompiled_binary = _find_precompiled()
        except RuntimeError:
            precompiled_binary = None
        except Exception as e:
            raise Exception(f"Build failed with exception {e}.")

        # If found - copy binary and return 
        if precompiled_binary and not int(os.environ.get("FORCE_BUILD_FROM_SOURCE", 0)):
            target_path = self.get_ext_fullpath(ext.name)
            LOGGER.info(f"Found matching precompiled binary. Copying {precompiled_binary} to {target_path}.")
            shutil.copy2(precompiled_binary, extdir)
            return 

        # Find CUDA
        try:
            nvcc_path = _find_cuda()
        except RuntimeError as e:
            LOGGER.warning(f"Encountered error {e}. Building {__name__} in CPU-only mode.")
            return
        cmake_args = [
            f"-DCMAKE_CUDA_COMPILER={nvcc_path}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        cmake_args.extend(ext.cmake_args)

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake"] + cmake_args + [ext.sourcedir], cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", f"-j{max(os.cpu_count() // 2, 1)}"],
            cwd=build_temp,
        )


def clone_submodule():
    """Clones the git submodules found .gitmodules in project directory."""
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd
    )


def main():
    clone_submodule()
    setuptools.setup(
        ext_modules=[CMakeExtension("_internal")],
        ext_package="gKDTree",
        cmdclass=dict(build_ext=CMakeBuild),
    )


if __name__ == "__main__":
    main()