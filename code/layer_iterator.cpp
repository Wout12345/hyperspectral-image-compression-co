#include <Python.h>
#include <iostream>
#include "numpy/arrayobject.h"

// Based on:
// https://www.tutorialspoint.com/python/python_further_extensions.htm
// https://stackoverflow.com/questions/43621948/c-python-module-import-error-undefined-symbol-py-initmodule3-py-initmodu
// https://docs.scipy.org/doc/numpy-1.15.0/user/c-info.how-to-extend.html

// Functions

static PyObject *extract_layers(PyObject *self, PyObject *args) {
	
	// Extracts layers in range [start:end] from multi-dimensional array in and stores them in the flat array out in a consistent order
	// Order doesn't necessarily give the best memory access pattern
	// arguments: in (array), start (int), end (int), out (array)
	
	int start, end;
	PyArrayObject *in, *out;
	
	if PyArg_ParseTuple(args, "O!iiO!", &PyArray_Type, &in, &start, &end, &PyArray_Type, &out) return NULL;
	
	// Test
	std::cout << in << std::endl << std::flush;
	std::cout << start << std::endl << std::flush;
	std::cout << end << std::endl << std::flush;
	std::cout << out << std::endl << std::flush;
	
	// Iterate over values
	int index = 0;
	for(int dim = 0; dim < PyArray_NDIM(in); dim++) {
		// Iterate over all indices with:
		// - index_i with i < dim: index_i < start
		// - index_i with i == dim: start <= index_i < end
		// - index_i with i > dim: index_i < end
	}
	
	// Return None
	return PyArray_Return(out);

}

static PyObject *insert_layers(PyObject *self, PyObject *args) {
	
	// Insert layers into range [start:end] in multi-dimensional array out from the flat array in in a consistent order
	// Order doesn't necessarily give the best memory access pattern
	// arguments: in (array), start (int), end (int), out (array)
	
	// TODO
	
	// Return None for now
	Py_RETURN_NONE;
	
}

// Test function

static PyObject *test(PyObject *self, PyObject *args) {
	// Test function
	std::cout << "Hello from C++" << std::endl << std::flush;
	return Py_BuildValue("s", "Another string!");
}

// Main structure for module

static PyMethodDef module_methods[] = {
	{"test", (PyCFunction) test, METH_NOARGS, NULL},
	{"extract_layers", (PyCFunction) extract_layers, METH_VARARGS, NULL},
	{"insert_layers", (PyCFunction) insert_layers, METH_VARARGS, NULL},
	{NULL} // Sentinel
};

static struct PyModuleDef layeriterator = {
	PyModuleDef_HEAD_INIT,
	"layeriterator", /* name of module */
	"Sorry, I didn't write documentation for this module.\n", /* module documentation, may be NULL */
	-1, // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
	module_methods
};

PyMODINIT_FUNC PyInit_layeriterator(void) {
	import_array();
	return PyModule_Create(&layeriterator);
}
