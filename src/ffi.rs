//! The C ABI surface of `pith-jpeg`: the entry points the Python
//! (ctypes), Node (koffi) and Go (cgo) SDKs bind through.
//!
//! The suite's FFI convention, defined by the pilot cdylibs and
//! mirrored by every `pith-*` cdylib:
//!
//! * flat `#[unsafe(no_mangle)] pub unsafe extern "C"` functions — raw
//!   pointers plus lengths, no structs across the boundary;
//! * every function returns a status code (see the constants below),
//!   never a `Result`, never a panic: a `panic = "abort"` cdylib must
//!   not be reachable from a foreign caller;
//! * nothing here allocates, so — unlike the decode-style cdylibs —
//!   this crate ships no `_free`: every output is a caller-provided
//!   fixed-size slot (the `pith-image` pilot set that precedent with
//!   `pith_image_phash`);
//! * the `unsafe` allowance is confined to this module; every core
//!   module stays unsafe-free behind the crate-root `#![deny]`.
//!
//! # Operations
//!
//! * [`pith_jpeg_decode`] — decode a whole JPEG and write **one
//!   `fnv1a64` digest** of the resulting sample bytes: row-major,
//!   channel-interleaved, exactly the buffer the committed
//!   `reference.json` `digests` section pins (and the `.raw`
//!   companions hold). `layout` selects the output encoding via the
//!   `pith-image` layout codes ([`PITH_LAYOUT_GRAY8`] = 0,
//!   [`PITH_LAYOUT_RGB8`] = 2, [`PITH_LAYOUT_RGBA8`] = 4); a grayscale
//!   image widens by channel replication, an RGB image to RGBA by
//!   opaque alpha, and RGB→gray is refused rather than invented.
//! * [`pith_jpeg_channels`] — the frame's native channel count (1 or
//!   3), so an SDK can pick the digest-exact layout without
//!   hard-coding fixture tables.
//! * [`pith_jpeg_dequant_zigzag`], [`pith_jpeg_dequant`],
//!   [`pith_jpeg_idct_islow`], [`pith_jpeg_idct_oracle`] — the four
//!   kernels the `reference.json` `vectors` section pins, so an SDK
//!   can replay every numeric vector. Buffers carry raw `f64` IEEE-754
//!   bit patterns as **little-endian u64** (one 8-byte lane per
//!   value); lengths are explicit and enforced.
//!
//! Status codes follow the suite convention: [`PITH_OK`],
//! [`PITH_E_INVALID`] (caller bug: null pointer, unknown layout,
//! wrong buffer length), [`PITH_E_REJECTED`] (the decoder refused the
//! input or the conversion is not offered).

#![allow(unsafe_code)]

use crate::huffman::ZIGZAG;
use crate::idct::dequant_idct_into;
use crate::{Jpeg, decode};
use pith_digest::fnv1a64;
use pith_math::idct2_2d;

/// Status: success.
pub const PITH_OK: i32 = 0;
/// Status: a caller argument is invalid — null pointer, unknown
/// layout, or a buffer whose length does not match the operation.
pub const PITH_E_INVALID: i32 = -1;
/// Status: the core decoder refused the input (malformed JPEG), or
/// the requested conversion is not offered (RGB→gray).
pub const PITH_E_REJECTED: i32 = -2;

/// Layout code: 8-bit grayscale (the `pith-image` code).
pub const PITH_LAYOUT_GRAY8: u32 = 0;
/// Layout code: 8-bit RGB, channel-interleaved (the `pith-image` code).
pub const PITH_LAYOUT_RGB8: u32 = 2;
/// Layout code: 8-bit RGBA, channel-interleaved (the `pith-image` code).
pub const PITH_LAYOUT_RGBA8: u32 = 4;

/// One `f64` lane in the kernel wire format, in bytes.
const LANE: usize = 8;
/// Every kernel op moves one 8×8 block: 64 values.
const BLOCK: usize = 64;
/// The two-input kernels carry 64 coefficients + 64 quantizer values.
const TWO_BLOCK: usize = 128;

/// Reads `n` little-endian `f64` bit patterns from `buf`.
///
/// # Safety
///
/// `buf` must point to `n * 8` readable bytes.
unsafe fn read_lanes(buf: *const u8, n: usize) -> Vec<f64> {
    let raw = unsafe { core::slice::from_raw_parts(buf, n * LANE) };
    raw.chunks_exact(LANE)
        .map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().expect("8 bytes"))))
        .collect()
}

/// Writes `values` as little-endian `f64` bit patterns into `buf`.
///
/// # Safety
///
/// `buf` must point to `values.len() * 8` writable bytes.
unsafe fn write_lanes(buf: *mut u8, values: &[f64]) {
    let raw = unsafe { core::slice::from_raw_parts_mut(buf, values.len() * LANE) };
    for (dst, v) in raw.chunks_exact_mut(LANE).zip(values) {
        dst.copy_from_slice(&v.to_bits().to_le_bytes());
    }
}

/// Length-checked lane read for the shell bodies: maps a wrong buffer
/// length to [`PITH_E_INVALID`].
fn lanes_in(buf: *const u8, len: usize, want: usize) -> Result<Vec<f64>, i32> {
    if buf.is_null() || len != want * LANE {
        return Err(PITH_E_INVALID);
    }
    Ok(unsafe { read_lanes(buf, want) })
}

/// Length-checked lane write for the shell bodies.
///
/// # Safety
///
/// `out` must be valid for `out_len` bytes when non-null.
unsafe fn lanes_out(out: *mut u8, out_len: usize, want: usize, values: &[f64]) -> Result<(), i32> {
    if out.is_null() || out_len != want * LANE {
        return Err(PITH_E_INVALID);
    }
    unsafe { write_lanes(out, values) };
    Ok(())
}

// ------------------------------------------------------------ decode side

/// The digest and channel count of one decoded frame under `layout`,
/// the safe core behind [`pith_jpeg_decode`] and
/// [`pith_jpeg_channels`]-style callers.
fn digest_for(bytes: &[u8], layout: u32) -> Result<(u64, u32), i32> {
    match layout {
        PITH_LAYOUT_GRAY8 | PITH_LAYOUT_RGB8 | PITH_LAYOUT_RGBA8 => {}
        _ => return Err(PITH_E_INVALID),
    }
    let jpeg = decode(bytes).map_err(|_| PITH_E_REJECTED)?;
    let native_channels: u32 = match &jpeg {
        Jpeg::Gray(_) => 1,
        Jpeg::Rgb(_) => 3,
    };
    let bytes: alloc::vec::Vec<u8> = match (&jpeg, layout) {
        (Jpeg::Gray(img), PITH_LAYOUT_GRAY8) => img.as_slice().to_vec(),
        (Jpeg::Rgb(img), PITH_LAYOUT_RGB8) => img.as_slice().to_vec(),
        (Jpeg::Gray(img), PITH_LAYOUT_RGB8) => {
            img.as_slice().iter().flat_map(|&g| [g, g, g]).collect()
        }
        (Jpeg::Gray(img), PITH_LAYOUT_RGBA8) => img
            .as_slice()
            .iter()
            .flat_map(|&g| [g, g, g, u8::MAX])
            .collect(),
        (Jpeg::Rgb(img), PITH_LAYOUT_RGBA8) => img
            .as_slice()
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], u8::MAX])
            .collect(),
        // RGB→gray would invent a luma weighting the suite never
        // pinned; refuse instead of guessing.
        (Jpeg::Rgb(_), PITH_LAYOUT_GRAY8) => return Err(PITH_E_REJECTED),
        _ => unreachable!("layout membership checked above"),
    };
    Ok((fnv1a64(&bytes), native_channels))
}

/// Decodes a JPEG and writes one `fnv1a64` digest of the resulting
/// sample bytes through `digest`: row-major, channel-interleaved, in
/// the `layout` encoding — exactly the buffer the committed
/// `reference.json` `digests` pin for the native layout.
///
/// # Safety
///
/// `data` must point to `len` readable bytes and `digest` to one
/// writable `u64`; both stay valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_jpeg_decode(
    data: *const u8,
    len: usize,
    layout: u32,
    digest: *mut u64,
) -> i32 {
    if data.is_null() && len != 0 {
        return PITH_E_INVALID;
    }
    if digest.is_null() {
        return PITH_E_INVALID;
    }
    if data.is_null() {
        // Null-ified zero-length bindings (Node's `Buffer.alloc(0)`) or
        // an explicit empty hand-off: the empty stream is a refusal,
        // never a slice.
        return PITH_E_REJECTED;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match digest_for(bytes, layout) {
        Ok((value, _)) => {
            unsafe { *digest = value };
            PITH_OK
        }
        Err(status) => status,
    }
}

/// Writes the frame's native channel count (1 or 3) through `out`
/// without converting anything: the digest-exact layout for a fixture
/// is [`PITH_LAYOUT_GRAY8`] when this reports 1 and
/// [`PITH_LAYOUT_RGB8`] when it reports 3.
///
/// # Safety
///
/// `data` must point to `len` readable bytes and `out` to one
/// writable `u32`; both stay valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_jpeg_channels(data: *const u8, len: usize, out: *mut u32) -> i32 {
    if data.is_null() && len != 0 {
        return PITH_E_INVALID;
    }
    if out.is_null() {
        return PITH_E_INVALID;
    }
    if data.is_null() {
        // Same refusal rule as `pith_jpeg_decode`.
        return PITH_E_REJECTED;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match decode(bytes) {
        Ok(jpeg) => {
            unsafe {
                *out = match jpeg {
                    Jpeg::Gray(_) => 1,
                    Jpeg::Rgb(_) => 3,
                };
            }
            PITH_OK
        }
        Err(_) => PITH_E_REJECTED,
    }
}

// ------------------------------------------------------------ kernel side

/// Replays the `dequant.zigzag.8x8` kernel: 64 DQT-payload values in
/// scan order go in, the same values in natural (row-major) order come
/// out (`out[ZIGZAG[k]] = in[k]`).
///
/// # Safety
///
/// `input` must point to `input_len` readable bytes and `output` to
/// `output_len` writable bytes; both stay valid for the duration of
/// the call. Buffers carry little-endian `f64` bit patterns; both
/// lengths must be exactly 512.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_jpeg_dequant_zigzag(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    let payload = match lanes_in(input, input_len, BLOCK) {
        Ok(v) => v,
        Err(status) => return status,
    };
    let mut natural = alloc::vec![0.0; BLOCK];
    for (k, &val) in payload.iter().enumerate() {
        natural[ZIGZAG[k] as usize] = val;
    }
    match unsafe { lanes_out(output, output_len, BLOCK, &natural) } {
        Ok(()) => PITH_OK,
        Err(status) => status,
    }
}

/// Replays the `dequant.8x8` kernel: 64 natural-order coefficients
/// followed by 64 natural-order quantizer values go in, the 64
/// element-wise products come out.
///
/// # Safety
///
/// `input` must point to `input_len` readable bytes and `output` to
/// `output_len` writable bytes; both stay valid for the duration of
/// the call. Buffers carry little-endian `f64` bit patterns; the
/// input length must be exactly 1024 and the output length exactly
/// 512.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_jpeg_dequant(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    let lanes = match lanes_in(input, input_len, TWO_BLOCK) {
        Ok(v) => v,
        Err(status) => return status,
    };
    let products: Vec<f64> = lanes[..BLOCK]
        .iter()
        .zip(&lanes[BLOCK..])
        .map(|(&c, &q)| c * q)
        .collect();
    match unsafe { lanes_out(output, output_len, BLOCK, &products) } {
        Ok(()) => PITH_OK,
        Err(status) => status,
    }
}

/// Replays the `idct.islow.8x8` kernel: the same 128-value input as
/// [`pith_jpeg_dequant`], the 64 `u8` samples of the fixed-point islow
/// transcription (dequantize + two-pass IDCT + level shift + clamp)
/// come out as `f64` lanes.
///
/// # Safety
///
/// `input` must point to `input_len` readable bytes and `output` to
/// `output_len` writable bytes; both stay valid for the duration of
/// the call. Buffers carry little-endian `f64` bit patterns; the
/// input length must be exactly 1024 and the output length exactly
/// 512.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_jpeg_idct_islow(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    let lanes = match lanes_in(input, input_len, TWO_BLOCK) {
        Ok(v) => v,
        Err(status) => return status,
    };
    let coefs: Vec<i32> = lanes[..BLOCK].iter().map(|&c| c as i32).collect();
    let qt: Vec<u16> = lanes[BLOCK..].iter().map(|&q| q as u16).collect();
    let mut plane = [0u8; BLOCK];
    dequant_idct_into(&coefs, &qt, &mut plane, 8, 0, 0);
    let samples: Vec<f64> = plane.iter().map(|&s| f64::from(s)).collect();
    match unsafe { lanes_out(output, output_len, BLOCK, &samples) } {
        Ok(()) => PITH_OK,
        Err(status) => status,
    }
}

/// Replays the `idct.oracle.8x8` kernel: 64 dequantized coefficients
/// go in, the raw orthonormal `f64` DCT-III oracle result comes out
/// (no level shift, no clamp). The oracle is platform-`libm`-shaped:
/// compare within the recorded tolerance, never bit-for-bit.
///
/// # Safety
///
/// `input` must point to `input_len` readable bytes and `output` to
/// `output_len` writable bytes; both stay valid for the duration of
/// the call. Buffers carry little-endian `f64` bit patterns; both
/// lengths must be exactly 512.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pith_jpeg_idct_oracle(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    let mut block = match lanes_in(input, input_len, BLOCK) {
        Ok(v) => v,
        Err(status) => return status,
    };
    idct2_2d(&mut block, 8, 8);
    match unsafe { lanes_out(output, output_len, BLOCK, &block) } {
        Ok(()) => PITH_OK,
        Err(status) => status,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK, PITH_E_INVALID, PITH_E_REJECTED, PITH_LAYOUT_GRAY8, PITH_LAYOUT_RGB8,
        PITH_LAYOUT_RGBA8, PITH_OK, digest_for, lanes_in, lanes_out, pith_jpeg_channels,
        pith_jpeg_decode, pith_jpeg_dequant, pith_jpeg_dequant_zigzag, pith_jpeg_idct_islow,
        pith_jpeg_idct_oracle,
    };
    use crate::reference;

    fn lanes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect()
    }

    fn unlanes(buf: &[u8]) -> Vec<f64> {
        buf.chunks_exact(8)
            .map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().expect("8 bytes"))))
            .collect()
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture")
    }

    /// Every committed fixture digests to its recorded value through
    /// the FFI core, in its native layout, and `channels` reports the
    /// matching native count.
    #[test]
    fn ffi_digests_reproduce_every_recorded_value() {
        for (name, want) in reference::digest_values() {
            let file = reference::digests()
                .iter()
                .find(|d| d.name == name)
                .expect("recorded digest")
                .file;
            let bytes = fixture(file);
            let layout = if name.contains("gray") {
                PITH_LAYOUT_GRAY8
            } else {
                PITH_LAYOUT_RGB8
            };
            let (digest, _) = digest_for(&bytes, layout).expect("decode");
            assert_eq!(digest, want, "{name}");
        }
    }

    /// The raw FFI entry point: OK + digest on a fixture, refusals on
    /// garbage, empty input, unknown layout and the refused RGB→gray
    /// conversion; null pointers are [`PITH_E_INVALID`].
    #[test]
    fn ffi_decode_statuses() {
        let bytes = fixture("base_gray.jpg");
        let mut digest: u64 = 0;
        let status = unsafe {
            pith_jpeg_decode(bytes.as_ptr(), bytes.len(), PITH_LAYOUT_GRAY8, &mut digest)
        };
        assert_eq!(status, PITH_OK);
        let want = reference::digest_values()
            .into_iter()
            .find(|(n, _)| *n == "base_gray")
            .expect("recorded")
            .1;
        assert_eq!(digest, want);

        // Pinned drift guard: the Rust-side literal the three SDK
        // harnesses pin too.
        assert_eq!(want, 0xe0ce_77f0_0e2c_b066);

        let mut channels: u32 = 0;
        let status = unsafe { pith_jpeg_channels(bytes.as_ptr(), bytes.len(), &mut channels) };
        assert_eq!(status, PITH_OK);
        assert_eq!(channels, 1);

        let garbage = b"not a jpeg at all";
        let status = unsafe {
            pith_jpeg_decode(
                garbage.as_ptr(),
                garbage.len(),
                PITH_LAYOUT_GRAY8,
                &mut digest,
            )
        };
        assert_eq!(status, PITH_E_REJECTED);

        let empty = [];
        let status = unsafe { pith_jpeg_decode(empty.as_ptr(), 0, PITH_LAYOUT_GRAY8, &mut digest) };
        assert_eq!(status, PITH_E_REJECTED);

        let status = unsafe { pith_jpeg_decode(bytes.as_ptr(), bytes.len(), 9, &mut digest) };
        assert_eq!(status, PITH_E_INVALID);

        let color = fixture("base_444.jpg");
        let status = unsafe {
            pith_jpeg_decode(color.as_ptr(), color.len(), PITH_LAYOUT_GRAY8, &mut digest)
        };
        assert_eq!(status, PITH_E_REJECTED);

        let status =
            unsafe { pith_jpeg_decode(core::ptr::null(), 0, PITH_LAYOUT_GRAY8, &mut digest) };
        assert_eq!(status, PITH_E_REJECTED);
        let status = unsafe {
            pith_jpeg_decode(
                bytes.as_ptr(),
                bytes.len(),
                PITH_LAYOUT_GRAY8,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(status, PITH_E_INVALID);
    }

    /// Layout conversions: gray widens by replication (digests move),
    /// RGB widens to RGBA by opaque alpha.
    #[test]
    fn ffi_layout_conversions() {
        let gray = fixture("base_gray.jpg");
        let mut native: u64 = 0;
        let mut widened: u64 = 0;
        unsafe {
            assert_eq!(
                pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_GRAY8, &mut native),
                PITH_OK
            );
            assert_eq!(
                pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_RGB8, &mut widened),
                PITH_OK
            );
            assert_ne!(native, widened);
            assert_eq!(
                pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_RGBA8, &mut widened),
                PITH_OK
            );
        }

        let color = fixture("base_444.jpg");
        let mut rgba: u64 = 0;
        unsafe {
            assert_eq!(
                pith_jpeg_decode(color.as_ptr(), color.len(), PITH_LAYOUT_RGBA8, &mut rgba),
                PITH_OK
            );
        }
        assert_ne!(rgba, 0);
    }

    /// The four kernel entry points reproduce the recorded outputs
    /// (bit-exact on the integer pipeline, in-tolerance on the
    /// oracle); wrong lengths and nulls are [`PITH_E_INVALID`].
    #[test]
    fn ffi_kernels_reproduce_the_recorded_vectors() {
        for vector in reference::vectors() {
            let input_lanes = lanes(&vector.input);
            let mut output = vec![0u8; vector.output.len() * 8];
            let status = unsafe {
                match vector.name {
                    "dequant.zigzag.8x8" => pith_jpeg_dequant_zigzag(
                        input_lanes.as_ptr(),
                        input_lanes.len(),
                        output.as_mut_ptr(),
                        output.len(),
                    ),
                    "dequant.8x8" => pith_jpeg_dequant(
                        input_lanes.as_ptr(),
                        input_lanes.len(),
                        output.as_mut_ptr(),
                        output.len(),
                    ),
                    "idct.islow.8x8" => pith_jpeg_idct_islow(
                        input_lanes.as_ptr(),
                        input_lanes.len(),
                        output.as_mut_ptr(),
                        output.len(),
                    ),
                    "idct.oracle.8x8" => pith_jpeg_idct_oracle(
                        input_lanes.as_ptr(),
                        input_lanes.len(),
                        output.as_mut_ptr(),
                        output.len(),
                    ),
                    other => panic!("unmapped vector {other}"),
                }
            };
            assert_eq!(status, PITH_OK, "{}", vector.name);
            let got = unlanes(&output);
            if vector.exact {
                assert_eq!(got, vector.output, "{}", vector.name);
            } else {
                for (i, (&g, &w)) in got.iter().zip(&vector.output).enumerate() {
                    let budget = vector.tol_abs.max(vector.tol_rel * w.abs());
                    assert!(
                        (g - w).abs() <= budget,
                        "{} lane {i}: {g} vs {w}",
                        vector.name
                    );
                }
            }
        }
    }

    /// Kernel refusals: wrong lengths, null pointers.
    #[test]
    fn ffi_kernel_refusals() {
        let ok_in = vec![0u8; BLOCK * 8];
        let mut ok_out = vec![0u8; BLOCK * 8];
        let short_in = vec![0u8; (BLOCK - 1) * 8];
        let status = unsafe {
            pith_jpeg_dequant_zigzag(
                short_in.as_ptr(),
                short_in.len(),
                ok_out.as_mut_ptr(),
                ok_out.len(),
            )
        };
        assert_eq!(status, PITH_E_INVALID);
        let status = unsafe {
            pith_jpeg_dequant_zigzag(
                ok_in.as_ptr(),
                ok_in.len(),
                ok_out.as_mut_ptr(),
                ok_out.len() - 8,
            )
        };
        assert_eq!(status, PITH_E_INVALID);
        let status = unsafe {
            pith_jpeg_dequant_zigzag(
                core::ptr::null(),
                ok_in.len(),
                ok_out.as_mut_ptr(),
                ok_out.len(),
            )
        };
        assert_eq!(status, PITH_E_INVALID);
        let status = unsafe {
            pith_jpeg_dequant(
                ok_in.as_ptr(),
                ok_in.len(),
                ok_out.as_mut_ptr(),
                ok_out.len(),
            )
        };
        assert_eq!(status, PITH_E_INVALID);
        let two = vec![0u8; 128 * 8];
        let status = unsafe {
            pith_jpeg_idct_islow(two.as_ptr(), two.len(), ok_out.as_mut_ptr(), ok_out.len())
        };
        assert_eq!(status, PITH_OK);
        let status = unsafe {
            pith_jpeg_idct_oracle(
                ok_in.as_ptr(),
                ok_in.len(),
                ok_out.as_mut_ptr(),
                ok_out.len(),
            )
        };
        assert_eq!(status, PITH_OK);
    }

    /// The lane helpers enforce exact lengths and round-trip bits.
    #[test]
    fn lane_helpers_roundtrip() {
        assert_eq!(
            lanes_in(core::ptr::null(), 0, BLOCK).unwrap_err(),
            PITH_E_INVALID
        );
        let values = vec![-0.5f64, 1.5, f64::MAX, -1.0e-300];
        let bytes = lanes(&values);
        let mut out = vec![0u8; bytes.len()];
        unsafe { lanes_out(out.as_mut_ptr(), out.len(), values.len(), &values).expect("write") };
        assert_eq!(unlanes(&out), values);
        unsafe {
            assert_eq!(
                lanes_out(out.as_mut_ptr(), out.len() - 1, values.len(), &values).unwrap_err(),
                PITH_E_INVALID
            );
        }
    }
}
