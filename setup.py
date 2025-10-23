import os
import subprocess
import sys

import setuptools
from setuptools.command.build_ext import build_ext

# constants to use
cwd = os.path.dirname(os.path.abspath(__file__))
CUDA_DEFAULT_PATH = "/usr/local/cuda"

def _find_cuda() -> str:
    """ Attempts to find a NVCC binary to compile the source files with. """
    # Attempt to find environment variable set
    if not (nvcc_path := os.environ.get("NVCC_PATH", False)):
        cuda_home = cuda_home = os.environ.get('CUDA_HOME', CUDA_DEFAULT_PATH)
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
    if not os.path.exists(nvcc_path):
        raise RuntimeError(f"NVCC not found at {nvcc_path}. Please try assign the environment variable `NVCC_PATH` to a valid binary.")
    return nvcc_path

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
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(f"Extension dir: {extdir}", file=sys.stderr)

        # Create the build directory
        os.makedirs(extdir, exist_ok=True)

        # Find CUDA
        nvcc_path = _find_cuda()
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