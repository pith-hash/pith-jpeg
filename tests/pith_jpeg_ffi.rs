//! FFI-contract conformance driven through the cdylib's public C ABI
//! surface, in an integration target of its own.
//!
//! This exists for a coverage-mechanics reason as much as a testing
//! one: with `crate-type = ["cdylib", "rlib"]`, `cargo llvm-cov
//! --workspace --all-targets` links the library into every test
//! binary, and the merged per-function record of `src/ffi.rs` ends up
//! owned by whichever target runs last. Driving every export from this
//! target (alphabetically the last integration suite) keeps that
//! merged record honest — the same fix wave A shipped for the other
//! ported crates.

#![allow(unsafe_code)]

use pith_jpeg::ffi::{
    PITH_E_INVALID, PITH_E_REJECTED, PITH_LAYOUT_GRAY8, PITH_LAYOUT_RGB8, PITH_LAYOUT_RGBA8,
    PITH_OK, pith_jpeg_channels, pith_jpeg_decode, pith_jpeg_dequant, pith_jpeg_dequant_zigzag,
    pith_jpeg_idct_islow, pith_jpeg_idct_oracle,
};
use pith_jpeg::reference;

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

#[test]
fn every_committed_digest_is_reproduced_through_the_ffi() {
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
        let mut digest: u64 = 0;
        let status = unsafe { pith_jpeg_decode(bytes.as_ptr(), bytes.len(), layout, &mut digest) };
        assert_eq!(status, PITH_OK, "{name}");
        assert_eq!(digest, want, "{name}");

        // The native channel count matches the layout the digest pins.
        let mut channels: u32 = 0;
        let status = unsafe { pith_jpeg_channels(bytes.as_ptr(), bytes.len(), &mut channels) };
        assert_eq!(status, PITH_OK, "{name}");
        let native_layout = if channels == 1 {
            PITH_LAYOUT_GRAY8
        } else {
            PITH_LAYOUT_RGB8
        };
        assert_eq!(native_layout, layout, "{name}");
    }
}

#[test]
fn decode_statuses_and_pinned_literal() {
    let gray = fixture("base_gray.jpg");
    let mut digest: u64 = 0;
    let status =
        unsafe { pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_GRAY8, &mut digest) };
    assert_eq!(status, PITH_OK);
    // Pinned drift guard: the literal the three SDK harnesses pin too
    // (reference.json digests["base_gray"]).
    assert_eq!(digest, 0xe0ce_77f0_0e2c_b066);

    // Unknown layout is a caller bug; garbage and empty input are
    // refusals; RGB→gray is refused rather than invented; nulls are
    // caller bugs.
    let color = fixture("base_444.jpg");
    let cases = [
        (gray.as_ptr(), gray.len(), 9u32, PITH_E_INVALID),
        (
            b"not a jpeg".as_ptr(),
            10,
            PITH_LAYOUT_GRAY8,
            PITH_E_REJECTED,
        ),
        (
            color.as_ptr(),
            color.len(),
            PITH_LAYOUT_GRAY8,
            PITH_E_REJECTED,
        ),
        (core::ptr::null(), 0, PITH_LAYOUT_GRAY8, PITH_E_REJECTED),
    ];
    for (ptr, len, layout, want) in cases {
        let status = unsafe { pith_jpeg_decode(ptr, len, layout, &mut digest) };
        assert_eq!(status, want);
    }
    let status = unsafe {
        pith_jpeg_decode(
            gray.as_ptr(),
            gray.len(),
            PITH_LAYOUT_GRAY8,
            core::ptr::null_mut(),
        )
    };
    assert_eq!(status, PITH_E_INVALID);

    let empty: [u8; 0] = [];
    let status = unsafe { pith_jpeg_decode(empty.as_ptr(), 0, PITH_LAYOUT_GRAY8, &mut digest) };
    assert_eq!(status, PITH_E_REJECTED);

    let mut channels: u32 = 0;
    let status = unsafe { pith_jpeg_channels(b"x".as_ptr(), 1, &mut channels) };
    assert_eq!(status, PITH_E_REJECTED);
    let status = unsafe { pith_jpeg_channels(gray.as_ptr(), gray.len(), core::ptr::null_mut()) };
    assert_eq!(status, PITH_E_INVALID);
}

#[test]
fn layout_conversions_widen_without_touching_natives() {
    let gray = fixture("base_gray.jpg");
    let mut native: u64 = 0;
    let mut rgb: u64 = 0;
    let mut rgba: u64 = 0;
    unsafe {
        assert_eq!(
            pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_GRAY8, &mut native),
            PITH_OK
        );
        assert_eq!(
            pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_RGB8, &mut rgb),
            PITH_OK
        );
        assert_eq!(
            pith_jpeg_decode(gray.as_ptr(), gray.len(), PITH_LAYOUT_RGBA8, &mut rgba),
            PITH_OK
        );
    }
    assert_ne!(native, rgb);
    assert_ne!(native, rgba);
    assert_ne!(rgb, rgba);

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

#[test]
fn every_kernel_vector_is_replayed_through_the_ffi() {
    for vector in reference::vectors() {
        let input = lanes(&vector.input);
        let mut output = vec![0u8; vector.output.len() * 8];
        let status = unsafe {
            match vector.name {
                "dequant.zigzag.8x8" => pith_jpeg_dequant_zigzag(
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                ),
                "dequant.8x8" => pith_jpeg_dequant(
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                ),
                "idct.islow.8x8" => pith_jpeg_idct_islow(
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    output.len(),
                ),
                "idct.oracle.8x8" => pith_jpeg_idct_oracle(
                    input.as_ptr(),
                    input.len(),
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

#[test]
fn kernel_refusals_cover_null_and_wrong_lengths() {
    let block = vec![0u8; 64 * 8];
    let two = vec![0u8; 128 * 8];
    let mut out = vec![0u8; 64 * 8];

    let status =
        unsafe { pith_jpeg_dequant_zigzag(two.as_ptr(), two.len(), out.as_mut_ptr(), out.len()) };
    assert_eq!(status, PITH_E_INVALID);
    let status = unsafe {
        pith_jpeg_dequant_zigzag(block.as_ptr(), block.len(), out.as_mut_ptr(), out.len() - 8)
    };
    assert_eq!(status, PITH_E_INVALID);
    let status = unsafe {
        pith_jpeg_dequant_zigzag(core::ptr::null(), block.len(), out.as_mut_ptr(), out.len())
    };
    assert_eq!(status, PITH_E_INVALID);
    let status =
        unsafe { pith_jpeg_dequant(block.as_ptr(), block.len(), out.as_mut_ptr(), out.len()) };
    assert_eq!(status, PITH_E_INVALID);
    let status =
        unsafe { pith_jpeg_idct_islow(two.as_ptr(), two.len(), out.as_mut_ptr(), out.len()) };
    assert_eq!(status, PITH_OK);
    let status =
        unsafe { pith_jpeg_idct_oracle(block.as_ptr(), block.len(), out.as_mut_ptr(), out.len()) };
    assert_eq!(status, PITH_OK);
}
