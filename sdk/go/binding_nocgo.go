// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && !cgo

package pithjpeg

import "fmt"

// The FFI entry points are unavailable without cgo on unix: there is
// no pure-Go dlopen in the standard library. Build with CGO_ENABLED=1
// (the CD pipeline always does).
func ffiDecode(string, *byte, int, uint32, *uint64) (int32, error) {
	return 0, fmt.Errorf("pithjpeg: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

func ffiChannels(string, *byte, int, *uint32) (int32, error) {
	return 0, fmt.Errorf("pithjpeg: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

func ffiKernel(string, string, *byte, int, *byte, int) (int32, error) {
	return 0, fmt.Errorf("pithjpeg: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}
