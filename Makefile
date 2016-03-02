.PHONY: clean all

all: py_wrap.pyx wrap.hpp matrix.hpp gbm.hpp
	python3 setup.py build_ext -i

clean:
	rm -f  py_wrap.cpp
	rm -f  *.so
	rm -rf build
	rm -rf __pycache__
