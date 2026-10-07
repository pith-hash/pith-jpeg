// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build windows

package pithjpeg

import (
	"fmt"
	"syscall"
	"unsafe"
)

// openProc loads libPath and resolves name. The library is released
// before returning: on Windows FreeLibrary unmaps the cdylib, so the
// proc must be used (and its results copied out) inside the caller.
func openProc(libPath, name string) (proc uintptr, release func(), err error) {
	lib, err := syscall.LoadLibrary(libPath)
	if err != nil {
		return 0, nil, fmt.Errorf("pithjpeg: LoadLibrary(%s): %w", libPath, err)
	}
	release = func() { syscall.FreeLibrary(lib) }
	proc, err = syscall.GetProcAddress(lib, name)
	if err != nil {
		release()
		return 0, nil, fmt.Errorf("pithjpeg: symbol %s missing from %s: %w", name, libPath, err)
	}
	return proc, release, nil
}

// ffiDecode opens the cdylib, resolves pith_jpeg_decode and calls it.
func ffiDecode(libPath string, data *byte, n int, layout uint32, digest *uint64) (int32, error) {
	proc, release, err := openProc(libPath, "pith_jpeg_decode")
	if err != nil {
		return 0, err
	}
	defer release()
	r, _, _ := syscall.SyscallN(
		proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(layout),
		uintptr(unsafe.Pointer(digest)),
	)
	return int32(r), nil
}

// ffiChannels opens the cdylib, resolves pith_jpeg_channels and calls
// it.
func ffiChannels(libPath string, data *byte, n int, channels *uint32) (int32, error) {
	proc, release, err := openProc(libPath, "pith_jpeg_channels")
	if err != nil {
		return 0, err
	}
	defer release()
	r, _, _ := syscall.SyscallN(
		proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(unsafe.Pointer(channels)),
	)
	return int32(r), nil
}

// ffiKernel opens the cdylib, resolves one of the fixed-buffer kernel
// symbols and calls it.
func ffiKernel(libPath, symbol string, in *byte, inLen int, out *byte, outLen int) (int32, error) {
	proc, release, err := openProc(libPath, symbol)
	if err != nil {
		return 0, err
	}
	defer release()
	r, _, _ := syscall.SyscallN(
		proc,
		uintptr(unsafe.Pointer(in)),
		uintptr(inLen),
		uintptr(unsafe.Pointer(out)),
		uintptr(outLen),
	)
	return int32(r), nil
}
