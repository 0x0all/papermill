from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name="papermill",
    version="0.0.1",
    description="xgboost-style gradient boosting machine",
    url="https://github.com/khyh/papermill",
    author="khyh",
    author_email="khyh@outlook.com",

    # main
    py_modules=[ "papermill" ],

    # ext
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

