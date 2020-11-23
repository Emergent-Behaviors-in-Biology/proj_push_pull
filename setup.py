import sys, os

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(DIR, "extern", "pybind11"))

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
del sys.path[-1]

from setuptools import setup
import pathlib

# touch .cpp file to force compile
pathlib.Path('cpp_src/python_bindings.cpp').touch()

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

include_dirs=["eigen-3.3.8"]
# include_dirs = []

ext_modules = [
    Pybind11Extension("push_pull_amp",
        ["cpp_src/python_bindings.cpp"],
        include_dirs=include_dirs,
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="push_pull_amp",
    version=__version__,
    author="Jason W. Rocks and Pankaj Mehta",
    author_email="jrocks@bu.edu",
    url="https://github.com/Emergent-Behaviors-in-Biology/proj_push_pull.git",
    description="Push-pull amplifier fitting functions.",
    long_description="Program for efficient computation of loss functions for fitting push-pull amplifier models to experimental data.",
    ext_modules=ext_modules,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)







