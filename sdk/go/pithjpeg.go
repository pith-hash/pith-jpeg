// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

// Package pithjpeg provides Go bindings for the pith-jpeg Rust
// cdylib: JPEG decoding and kernel replay.
//
// The single Rust core (built by `cargo build --release`) is loaded at
// runtime; the package carries zero module dependencies. On unix the
// cdylib is opened with dlopen through cgo, on Windows with
// LoadLibrary through the standard syscall package — both resolve the
// library through the same discovery chain, so `go build ./... &&
// go test ./...` works unchanged on every OS the CD matrix builds.
//
// Discovery order (the suite's cdylib convention):
//
//  1. PITH_CDYLIB — an explicit cdylib file path;
//  2. PITH_CDYLIB_DIR — a directory scanned for the cdylib names (the
//     CD pipeline points this at target/release);
//  3. <repo root>/target/release — the repository working-tree layout,
//     anchored at this package's source directory, so a source
//     checkout runs against a local cargo build unconfigured.
//
// The FFI surface is allocation-free: pith_jpeg_decode digests the
// decoded sample bytes inside the cdylib and returns the fnv1a64 the
// reference.json digests section pins; pith_jpeg_channels reports the
// native channel count; the four kernel entry points replay the
// numeric vectors over fixed-size little-endian f64 bit-pattern
// buffers (8 bytes per lane; reference.json renders each lane as
// big-endian hex digits).
package pithjpeg

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

// Status codes returned by the cdylib's C ABI.
const (
	// StatusOK: success.
	StatusOK int32 = 0
	// StatusInvalid: a caller argument is invalid — a null pointer
	// with a non-zero length, an unknown layout code, a wrong buffer
	// length.
	StatusInvalid int32 = -1
	// StatusRejected: the decoder refused the input (malformed JPEG),
	// or the requested conversion is not offered (RGB→gray). An empty
	// stream is refused as malformed whatever the pointer.
	StatusRejected int32 = -2
)

// Output layout codes — the pith-image identities.
const (
	// LayoutGray8: 8-bit grayscale.
	LayoutGray8 uint32 = 0
	// LayoutRgb8: 8-bit RGB, channel-interleaved.
	LayoutRgb8 uint32 = 2
	// LayoutRgba8: 8-bit RGBA, channel-interleaved.
	LayoutRgba8 uint32 = 4
)

// Block is one 8×8 kernel block: 64 f64 lanes.
const Block = 64

// cdylibNames are the file names cargo may drop into the build
// directory, per platform (windows / linux / macOS).
var cdylibNames = []string{"pith_jpeg.dll", "libpith_jpeg.so", "libpith_jpeg.dylib"}

// FfiError reports a non-zero status code from the cdylib.
type FfiError struct {
	// Op is the FFI operation name.
	Op string
	// Status is the raw status code the FFI returned.
	Status int32
}

func (e *FfiError) Error() string {
	kind := "unknown failure"
	switch e.Status {
	case StatusInvalid:
		kind = "invalid argument"
	case StatusRejected:
		kind = "input rejected"
	}
	return fmt.Sprintf("%s failed: %s (status %d)", e.Op, kind, e.Status)
}

// FindCdylib locates the cdylib through the suite's discovery chain.
func FindCdylib() (string, error) {
	if p := os.Getenv("PITH_CDYLIB"); p != "" {
		if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
			return filepath.Abs(p)
		}
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("pithjpeg: cannot locate the package source directory")
	}
	pkgDir := filepath.Dir(thisFile)
	repoRoot := filepath.Dir(filepath.Dir(pkgDir)) // sdk/go -> sdk -> repo root

	var dirs []string
	if env := os.Getenv("PITH_CDYLIB_DIR"); env != "" {
		dirs = append(dirs, env)
		if !filepath.IsAbs(env) {
			dirs = append(dirs, filepath.Join(repoRoot, env))
		}
	}
	dirs = append(dirs, filepath.Join(repoRoot, "target", "release"))
	for _, dir := range dirs {
		for _, name := range cdylibNames {
			p := filepath.Join(dir, name)
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf(
		"pithjpeg: no cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR and <repo>/target/release); run `cargo build --release` first",
	)
}

// locate resolves the cdylib path once per process.
var locate = sync.OnceValues(FindCdylib)

// DecodeDigest decodes a JPEG and returns the fnv1a64 digest of the
// sample bytes in layout encoding: row-major, channel-interleaved —
// the exact value reference.json pins for the native layout.
func DecodeDigest(data []byte, layout uint32) (uint64, error) {
	libPath, err := locate()
	if err != nil {
		return 0, err
	}
	var digest uint64
	var dataPtr *byte
	if len(data) > 0 {
		dataPtr = &data[0]
	}
	status, err := ffiDecode(libPath, dataPtr, len(data), layout, &digest)
	if err != nil {
		return 0, err
	}
	if status != StatusOK {
		return 0, &FfiError{Op: "pith_jpeg_decode", Status: status}
	}
	return digest, nil
}

// Channels returns the frame's native channel count (1 or 3) — the
// digest-exact layout is LayoutGray8 for 1 and LayoutRgb8 for 3.
func Channels(data []byte) (uint32, error) {
	libPath, err := locate()
	if err != nil {
		return 0, err
	}
	var channels uint32
	var dataPtr *byte
	if len(data) > 0 {
		dataPtr = &data[0]
	}
	status, err := ffiChannels(libPath, dataPtr, len(data), &channels)
	if err != nil {
		return 0, err
	}
	if status != StatusOK {
		return 0, &FfiError{Op: "pith_jpeg_channels", Status: status}
	}
	return channels, nil
}

// lanesToBytes packs f64 lanes as little-endian IEEE-754 bit patterns.
func lanesToBytes(values []float64) []byte {
	buf := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return buf
}

// bytesToLanes unpacks little-endian f64 bit patterns into lanes.
func bytesToLanes(buf []byte) []float64 {
	out := make([]float64, len(buf)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[i*8:]))
	}
	return out
}

// laneValue decodes one big-endian hex lane of reference.json.
func laneValue(hex string) float64 {
	var bits uint64
	for i := range 16 {
		bits = bits<<4 | uint64(hexDigit(hex[i]))
	}
	return math.Float64frombits(bits)
}

func hexDigit(c byte) byte {
	switch {
	case c >= '0' && c <= '9':
		return c - '0'
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10
	default:
		panic(fmt.Sprintf("pithjpeg: bad hex digit %q", c))
	}
}

// replayKernel feeds one kernel op over little-endian f64 bit-pattern
// buffers and returns the decoded output lanes.
func replayKernel(op, symbol string, in []float64, outLanes int) ([]float64, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	inBuf := lanesToBytes(in)
	out := make([]byte, outLanes*8)
	var inPtr, outPtr *byte
	if len(inBuf) > 0 {
		inPtr = &inBuf[0]
	}
	if len(out) > 0 {
		outPtr = &out[0]
	}
	status, err := ffiKernel(libPath, symbol, inPtr, len(inBuf), outPtr, len(out))
	if err != nil {
		return nil, err
	}
	if status != StatusOK {
		return nil, &FfiError{Op: op, Status: status}
	}
	return bytesToLanes(out), nil
}

// DequantZigzag replays the "dequant.zigzag.8x8" kernel: 64 DQT
// payload values in scan order in, the same values in natural
// (row-major) order out (natural[ZIGZAG[k]] = payload[k]).
func DequantZigzag(payload []float64) ([]float64, error) {
	if len(payload) != Block {
		return nil, fmt.Errorf("pithjpeg: dequant zigzag needs exactly %d lanes, got %d", Block, len(payload))
	}
	return replayKernel("pith_jpeg_dequant_zigzag", "pith_jpeg_dequant_zigzag", payload, Block)
}

// Dequant replays the "dequant.8x8" kernel: 64 natural-order
// coefficients and 64 quantizer values in, the 64 element-wise
// products out.
func Dequant(coefs, qt []float64) ([]float64, error) {
	if len(coefs) != Block || len(qt) != Block {
		return nil, fmt.Errorf("pithjpeg: dequant needs exactly %d+%d lanes", Block, Block)
	}
	return replayKernel("pith_jpeg_dequant", "pith_jpeg_dequant", append(append([]float64{}, coefs...), qt...), Block)
}

// IdctIslow replays the "idct.islow.8x8" kernel: the same input layout
// as Dequant, the 64 u8 samples of the fixed-point islow transcription
// out as f64 lanes.
func IdctIslow(coefs, qt []float64) ([]float64, error) {
	if len(coefs) != Block || len(qt) != Block {
		return nil, fmt.Errorf("pithjpeg: idct islow needs exactly %d+%d lanes", Block, Block)
	}
	return replayKernel("pith_jpeg_idct_islow", "pith_jpeg_idct_islow", append(append([]float64{}, coefs...), qt...), Block)
}

// IdctOracle replays the "idct.oracle.8x8" kernel: 64 dequantized
// coefficients in, the raw orthonormal f64 DCT-III oracle result out.
// Platform-libm-shaped: compare within the recorded tolerance, never
// bit-for-bit.
func IdctOracle(block []float64) ([]float64, error) {
	if len(block) != Block {
		return nil, fmt.Errorf("pithjpeg: idct oracle needs exactly %d lanes, got %d", Block, len(block))
	}
	return replayKernel("pith_jpeg_idct_oracle", "pith_jpeg_idct_oracle", block, Block)
}
