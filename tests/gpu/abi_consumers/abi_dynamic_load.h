#ifndef GAFIME_TEST_ABI_DYNAMIC_LOAD_H
#define GAFIME_TEST_ABI_DYNAMIC_LOAD_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE GafimeTestLibrary;
typedef FARPROC GafimeTestSymbol;

static GafimeTestLibrary gafime_test_library_open(const char* path) {
    return LoadLibraryA(path);
}

static GafimeTestSymbol gafime_test_library_symbol(GafimeTestLibrary library, const char* name) {
    return GetProcAddress(library, name);
}

static void gafime_test_library_close(GafimeTestLibrary library) {
    if (library != NULL) {
        FreeLibrary(library);
    }
}

static const char* gafime_test_library_error(void) {
    return "LoadLibrary/GetProcAddress failed";
}
#else
#include <dlfcn.h>
typedef void* GafimeTestLibrary;
typedef void* GafimeTestSymbol;

static GafimeTestLibrary gafime_test_library_open(const char* path) {
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

static GafimeTestSymbol gafime_test_library_symbol(GafimeTestLibrary library, const char* name) {
    dlerror();
    return dlsym(library, name);
}

static void gafime_test_library_close(GafimeTestLibrary library) {
    if (library != NULL) {
        dlclose(library);
    }
}

static const char* gafime_test_library_error(void) {
    const char* error = dlerror();
    return error == NULL ? "dynamic loader failed without an error string" : error;
}
#endif

/* ISO C does not define a direct data-pointer-to-function-pointer cast. */
#define GAFIME_TEST_LOAD_FUNCTION(library, destination, symbol_name)                    \
    do {                                                                                \
        GafimeTestSymbol gafime_test_symbol_value =                                     \
            gafime_test_library_symbol((library), (symbol_name));                       \
        if (gafime_test_symbol_value == NULL) {                                         \
            fprintf(stderr, "%s: %s\n", (symbol_name), gafime_test_library_error());   \
            gafime_test_library_close((library));                                       \
            return 1;                                                                   \
        }                                                                               \
        _Static_assert(sizeof(destination) == sizeof(gafime_test_symbol_value),         \
                       "function and dynamic-symbol pointers must have equal size");   \
        memcpy(&(destination), &gafime_test_symbol_value, sizeof(destination));         \
    } while (0)

#endif
