from distutils.core import setup, Extension

setup(name="layeriterator", version="1.0", ext_modules=[Extension("layeriterator", ["layer_iterator.cpp"])])
