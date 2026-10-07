// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && cgo

package pithjpeg

/*
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int32_t (*pith_decode_fn)(const uint8_t *, size_t, uint32_t, uint64_t *);
typedef int32_t (*pith_channels_fn)(const uint8_t *, size_t, uint32_t *);
typedef int32_t (*pith_kernel_fn)(const uint8_t *, size_t, uint8_t *, size_t);

static int32_t pith_call_decode(void *fn, const uint8_t *data, size_t len,
                                uint32_t layout, uint64_t *digest) {
    return ((pith_decode_fn)fn)(data, len, layout, digest);
}

static int32_t pith_call_channels(void *fn, const uint8_t *data, size_t len,
                                  uint32_t *channels) {
    return ((pith_channels_fn)fn)(data, len, channels);
}

static int32_t pith_call_kernel(void *fn, const uint8_t *in, size_t in_len,
                                uint8_t *out, size_t out_len) {
    return ((pith_kernel_fn)fn)(in, in_len, out, out_len);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ffiSymbol resolves one exported symbol of one open cdylib handle.
func ffiSymbol(handle unsafe.Pointer, libPath, name string) (unsafe.Pointer, error) {
	cName := C.CString(name)
	sym := C.dlsym(handle, cName)
	C.free(unsafe.Pointer(cName))
	if sym == nil {
		return nil, fmt.Errorf("pithjpeg: symbol %s missing from %s", name, libPath)
	}
	return sym, nil
}

// openCdylib dlopens libPath with error text surfaced verbatim.
func openCdylib(libPath string) (unsafe.Pointer, error) {
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_LOCAL)
	if handle == nil {
		msg := "unknown dlopen failure"
		if e := C.dlerror(); e != nil {
			msg = C.GoString(e)
		}
		return nil, fmt.Errorf("pithjpeg: dlopen(%s): %s", libPath, msg)
	}
	return handle, nil
}

// ffiDecode opens the cdylib, resolves pith_jpeg_decode and calls it.
func ffiDecode(libPath string, data *byte, n int, layout uint32, digest *uint64) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	sym, err := ffiSymbol(handle, libPath, "pith_jpeg_decode")
	if err != nil {
		return 0, err
	}
	rc := C.pith_call_decode(sym, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), C.uint32_t(layout), (*C.uint64_t)(unsafe.Pointer(digest)))
	return int32(rc), nil
}

// ffiChannels opens the cdylib, resolves pith_jpeg_channels and calls
// it.
func ffiChannels(libPath string, data *byte, n int, channels *uint32) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	sym, err := ffiSymbol(handle, libPath, "pith_jpeg_channels")
	if err != nil {
		return 0, err
	}
	rc := C.pith_call_channels(sym, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), (*C.uint32_t)(unsafe.Pointer(channels)))
	return int32(rc), nil
}

// ffiKernel opens the cdylib, resolves one of the fixed-buffer kernel
// symbols and calls it.
func ffiKernel(libPath, symbol string, in *byte, inLen int, out *byte, outLen int) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)
	sym, err := ffiSymbol(handle, libPath, symbol)
	if err != nil {
		return 0, err
	}
	rc := C.pith_call_kernel(sym, (*C.uint8_t)(unsafe.Pointer(in)), C.size_t(inLen), (*C.uint8_t)(unsafe.Pointer(out)), C.size_t(outLen))
	return int32(rc), nil
}
