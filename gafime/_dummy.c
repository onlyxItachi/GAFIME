#include <Python.h>
static struct PyModuleDef dummy_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Dummy extension block",
    -1,
    NULL, NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&dummy_module);
}
