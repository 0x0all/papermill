from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name="gbm",

    ext_modules=[
        Extension(
            "gbm",
            sources=["py_wrap.pyx",],
            extra_compile_args=["-O3", "-msse2", "-fopenmp", "-std=c++11"],
            extra_link_args=["-lgomp"],
            language="c++",
            )
        ],
    cmdclass = {"build_ext": build_ext},
    )

