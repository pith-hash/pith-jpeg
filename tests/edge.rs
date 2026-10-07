//! Edge-coverage streams: hand-built minimal JPEGs for decoder paths
//! the Pillow conformance corpus cannot reach — unusual sampling
//! factors (h1v2 fancy, box fallbacks, 4:1 replication, fractional
//! ratios), Adobe APP14 variants, and marker corner cases (DNL, DAC,
//! short DRI, second SOS, restarts in single-component scans).
//!
//! Every stream shares one minimal table set: a single 8-bit DQT
//! (all 16s), a DC Huffman table and an AC Huffman table where each
//! needed symbol costs one bit, so one all-zero block costs exactly
//! two `0` bits. Expected outputs are structural (dimensions, Ok/Err,
//! error kind) — the byte-exact contract stays with the committed
//! conformance corpus.

use pith_jpeg::{Jpeg, decode};

/// Segment builder: marker byte + big-endian length (incl. itself) +
/// body.
fn segment(marker: u8, body: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, marker];
    v.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    v.extend_from_slice(body);
    v
}

fn dqt(tq: u8, val: u8) -> Vec<u8> {
    let mut body = vec![tq];
    body.extend(std::iter::repeat_n(val, 64));
    segment(0xDB, &body)
}

/// DHT with one-bit codes: `symbols` get codes 0..n at length 1.
fn dht(tc: u8, th: u8, symbols: &[u8]) -> Vec<u8> {
    let mut counts = [0u8; 16];
    counts[0] = symbols.len() as u8;
    let mut body = vec![(tc << 4) | th];
    body.extend_from_slice(&counts);
    body.extend_from_slice(symbols);
    segment(0xC4, &body)
}

/// SOF0: precision 8, `w x h`, components as `(id, hv, tq)`.
fn sof0(w: u16, h: u16, comps: &[(u8, u8, u8)]) -> Vec<u8> {
    let mut body = vec![8u8];
    body.extend_from_slice(&h.to_be_bytes());
    body.extend_from_slice(&w.to_be_bytes());
    body.push(comps.len() as u8);
    for &(id, hv, tq) in comps {
        body.extend_from_slice(&[id, hv, tq]);
    }
    segment(0xC0, &body)
}

/// SOS: `comps` as `(id, td_ta)`, then Ss=0, Se=63, Ah/Al=0.
fn sos(comps: &[(u8, u8)]) -> Vec<u8> {
    let mut body = vec![comps.len() as u8];
    for &(id, tdta) in comps {
        body.extend_from_slice(&[id, tdta]);
    }
    body.extend_from_slice(&[0x00, 0x3F, 0x00]);
    segment(0xDA, &body)
}

fn app14(transform: u8) -> Vec<u8> {
    let mut body = b"Adobe".to_vec();
    body.extend_from_slice(&[0, 100, 0, 0, 0, 0, transform]);
    segment(0xEE, &body)
}

fn dri(n: u16) -> Vec<u8> {
    segment(0xDD, &n.to_be_bytes())
}

fn dnl() -> Vec<u8> {
    segment(0xDC, &[0x00, 0x00])
}

fn dac() -> Vec<u8> {
    segment(0xCC, &[0x00, 0x00])
}

/// Raw marker pair (e.g. RST0-7, SOI, EOI).
fn marker(code: u8) -> Vec<u8> {
    vec![0xFF, code]
}

/// Entropy-bit packer: MSB-first, `1`-padding to byte boundaries.
struct Bits {
    byte: u8,
    n: u32,
    out: Vec<u8>,
}

impl Bits {
    fn new() -> Self {
        Bits {
            byte: 0,
            n: 0,
            out: Vec::new(),
        }
    }

    fn put(&mut self, bit: bool) {
        self.byte = (self.byte << 1) | u8::from(bit);
        self.n += 1;
        if self.n == 8 {
            self.out.push(self.byte);
            if self.byte == 0xFF {
                // Entropy byte stuffing, as every encoder must.
                self.out.push(0x00);
            }
            self.byte = 0;
            self.n = 0;
        }
    }

    /// `count` zero bits (DC category 0 and AC EOB codes).
    fn zeros(&mut self, count: u32) {
        for _ in 0..count {
            self.put(false);
        }
    }

    /// MSB-first raw bits of `value` (DC category payload).
    fn value(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.put((value >> i) & 1 == 1);
        }
    }

    fn pad(&mut self) {
        while self.n != 0 {
            self.put(true);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.pad();
        self.out
    }
}

/// Prefix shared by all streams: SOI, tables, optional extras.
fn tables(dc_symbols: &[u8]) -> Vec<u8> {
    let mut v = marker(0xD8);
    v.extend(dqt(0x00, 16));
    v.extend(dht(0, 0, dc_symbols));
    v.extend(dht(1, 0, &[0x00]));
    v
}

/// One decoded all-zero block costs two `0` bits (DC cat-0 + AC EOB).
const BLOCK_BITS: u32 = 2;

/// Entropy bytes for `blocks` all-zero blocks of a scan with no
/// restart markers.
fn plain_entropy(blocks: usize) -> Vec<u8> {
    let mut b = Bits::new();
    b.zeros(BLOCK_BITS * blocks as u32);
    b.finish()
}

#[test]
fn gray_single_component_scan_honors_restart_interval() {
    // 16x8 gray, DRI=1: the non-interleaved walk restarts between the
    // two MCUs — consume_rst must find the literal RST0 bytes.
    let mut stream = tables(&[0x00]);
    stream.extend(dri(1));
    stream.extend(sof0(16, 8, &[(1, 0x11, 0)]));
    stream.extend(sos(&[(1, 0x00)]));
    stream.extend(Bits::new_with_zeros(BLOCK_BITS).finish());
    stream.extend(marker(0xD0)); // RST0
    stream.extend(Bits::new_with_zeros(BLOCK_BITS).finish());
    stream.extend(marker(0xD9));
    let img = decode(&stream).expect("restart-bearing gray stream decodes");
    assert_eq!((img.width(), img.height()), (16, 8));
}

impl Bits {
    fn new_with_zeros(count: u32) -> Self {
        let mut b = Bits::new();
        b.zeros(count);
        b
    }
}

#[test]
fn empty_sos_segment_is_a_truncation() {
    let mut stream = tables(&[0x00]);
    stream.extend(sof0(8, 8, &[(1, 0x11, 0)]));
    stream.extend(&[0xFF, 0xDA, 0x00, 0x02]); // SOS, no body at all
    let err = decode(&stream).expect_err("empty SOS must fail");
    assert!(err.to_string().contains("SOS header"), "{err}");
}

#[test]
fn sos_component_count_zero_is_rejected() {
    let mut stream = tables(&[0x00]);
    stream.extend(sof0(8, 8, &[(1, 0x11, 0)]));
    stream.extend(sos(&[])); // ns = 0 encoded by the builder
    let err = decode(&stream).expect_err("ns=0 must fail");
    assert!(err.to_string().contains("scan component count"), "{err}");
}

#[test]
fn interleaved_scan_in_single_component_frame_is_rejected() {
    // ns=2 but both scan entries point at the same (only) component,
    // so the per-component lookups succeed and the final interleaving
    // guard fires.
    let mut stream = tables(&[0x00]);
    stream.extend(sof0(8, 8, &[(1, 0x11, 0)]));
    stream.extend(sos(&[(1, 0x00), (1, 0x00)]));
    let err = decode(&stream).expect_err("ns=2 over a 1-component frame");
    assert!(err.to_string().contains("interleaved scan"), "{err}");
}

/// 3-component stream with per-component sampling `samps`, frame
/// `w x h`, and enough entropy for the resulting block count.
fn ycbcr_stream(w: u16, h: u16, y_hv: u8, c_hv: u8, extra: &[Vec<u8>]) -> Vec<u8> {
    let hmax = (y_hv >> 4).max(c_hv >> 4) as usize;
    let vmax = (y_hv & 15).max(c_hv & 15) as usize;
    let mcus_x = w.div_ceil(8 * hmax as u16) as usize;
    let mcus_y = h.div_ceil(8 * vmax as u16) as usize;
    let blocks = mcus_x * mcus_y * ((y_hv >> 4) as usize) * ((y_hv & 15) as usize)
        + 2 * mcus_x * mcus_y * ((c_hv >> 4) as usize) * ((c_hv & 15) as usize);
    let mut stream = tables(&[0x00]);
    for seg in extra {
        stream.extend(seg);
    }
    stream.extend(sof0(w, h, &[(1, y_hv, 0), (2, c_hv, 0), (3, c_hv, 0)]));
    stream.extend(sos(&[(1, 0x00), (2, 0x00), (3, 0x00)]));
    stream.extend(plain_entropy(blocks));
    stream.extend(marker(0xD9));
    stream
}

#[test]
fn h1v2_vertical_fancy_upsample_decodes() {
    // 4:4:0-shaped: luma h1v2, chroma h1v1 — the h1v2 triangle filter.
    let stream = ycbcr_stream(16, 16, 0x12, 0x11, &[]);
    let img = decode(&stream).expect("h1v2 stream decodes");
    assert_eq!((img.width(), img.height()), (16, 16));
    assert!(matches!(img, Jpeg::Rgb(_)));
}

#[test]
fn narrow_h2v1_uses_box_fallback() {
    // 4x8 4:2:2: chroma down_w = 2 ≤ 2 — the plain box filter.
    let stream = ycbcr_stream(4, 8, 0x21, 0x11, &[]);
    let img = decode(&stream).expect("narrow h2v1 stream decodes");
    assert_eq!((img.width(), img.height()), (4, 8));
}

#[test]
fn narrow_h2v2_uses_box_fallback() {
    // 4x4 4:2:0: chroma down grid 2x2 — the 2D box filter.
    let stream = ycbcr_stream(4, 4, 0x22, 0x11, &[]);
    let img = decode(&stream).expect("narrow h2v2 stream decodes");
    assert_eq!((img.width(), img.height()), (4, 4));
}

#[test]
fn four_to_one_replication_decodes() {
    // 32x8 with luma h4v1: chroma upsamples by integral 4x1 replicate.
    let stream = ycbcr_stream(32, 8, 0x41, 0x11, &[]);
    let img = decode(&stream).expect("4:1 replication stream decodes");
    assert_eq!((img.width(), img.height()), (32, 8));
}

#[test]
fn fractional_sampling_ratio_is_unsupported() {
    // Luma h4, chroma h3: 4 % 3 != 0 — no integral upsample exists.
    let stream = ycbcr_stream(32, 8, 0x41, 0x31, &[]);
    let err = decode(&stream).expect_err("fractional ratio must fail");
    assert!(err.to_string().contains("fractional sampling"), "{err}");
}

#[test]
fn rgb_component_ids_passthrough_without_app14() {
    // Component ids literally "R","G","B" and no Adobe marker: RGB
    // passthrough by id sniffing.
    let mut stream = tables(&[0x00]);
    stream.extend(sof0(
        8,
        8,
        &[(b'R', 0x11, 0), (b'G', 0x11, 0), (b'B', 0x11, 0)],
    ));
    stream.extend(sos(&[(b'R', 0x00), (b'G', 0x00), (b'B', 0x00)]));
    stream.extend(plain_entropy(3));
    stream.extend(marker(0xD9));
    let img = decode(&stream).expect("RGB-id stream decodes");
    assert!(matches!(img, Jpeg::Rgb(_)));
}

#[test]
fn app14_transform_zero_is_rgb_passthrough() {
    let stream = ycbcr_stream(8, 8, 0x11, 0x11, &[app14(0)]);
    let img = decode(&stream).expect("transform=0 stream decodes");
    assert!(matches!(img, Jpeg::Rgb(_)));
}

#[test]
fn app14_transform_two_names_ycck() {
    let stream = ycbcr_stream(8, 8, 0x11, 0x11, &[app14(2)]);
    let err = decode(&stream).expect_err("transform=2 must fail");
    assert!(err.to_string().contains("YCCK"), "{err}");
}

#[test]
fn dnl_marker_is_skipped() {
    let stream = ycbcr_stream(8, 8, 0x11, 0x11, &[dnl()]);
    let img = decode(&stream).expect("DNL-bearing stream decodes");
    assert_eq!((img.width(), img.height()), (8, 8));
}

#[test]
fn dac_marker_names_arithmetic_coding() {
    let stream = ycbcr_stream(8, 8, 0x11, 0x11, &[dac()]);
    let err = decode(&stream).expect_err("DAC must fail");
    assert!(err.to_string().contains("arithmetic"), "{err}");
}

#[test]
fn short_dri_segment_is_a_truncation() {
    let mut stream = tables(&[0x00]);
    stream.extend(&[0xFF, 0xDD, 0x00, 0x03, 0x00]); // DRI body of 1 byte
    stream.extend(sof0(8, 8, &[(1, 0x11, 0)]));
    stream.extend(sos(&[(1, 0x00)]));
    stream.extend(plain_entropy(1));
    stream.extend(marker(0xD9));
    let err = decode(&stream).expect_err("1-byte DRI must fail");
    assert!(err.to_string().contains("DRI segment"), "{err}");
}

#[test]
fn second_sos_in_sequential_stream_is_rejected() {
    let mut stream = tables(&[0x00]);
    stream.extend(sof0(8, 8, &[(1, 0x11, 0)]));
    stream.extend(sos(&[(1, 0x00)]));
    stream.extend(plain_entropy(1));
    stream.extend(sos(&[(1, 0x00)]));
    stream.extend(plain_entropy(1));
    stream.extend(marker(0xD9));
    let err = decode(&stream).expect_err("second SOS must fail");
    assert!(err.to_string().contains("second SOS"), "{err}");
}

/// DC category-11 stream: every block carries the same huge DC
/// coefficient, driving the color converter into its clamp arms.
fn dc11_stream(payload: u32) -> Vec<u8> {
    let mut stream = tables(&[0x00, 0x0B]); // DC codes: 0 -> "0", 11 -> "1"
    stream.extend(sof0(8, 8, &[(1, 0x11, 0), (2, 0x11, 0), (3, 0x11, 0)]));
    stream.extend(sos(&[(1, 0x00), (2, 0x00), (3, 0x00)]));
    let mut b = Bits::new();
    for _ in 0..3 {
        b.put(true); // DC symbol 11 (1-bit code "1")
        b.value(payload, 11); // category-11 payload
        b.zeros(1); // AC EOB
    }
    stream.extend(b.finish());
    stream.extend(marker(0xD9));
    stream
}

#[test]
fn huge_positive_dc_decodes_without_wrap_or_panic() {
    // +2047 x qt 16 overflows the nominal range: the decoder must land
    // on its documented range-limit behavior, never panic and never
    // emit out-of-range samples (the u8 planes make that structural).
    let stream = dc11_stream(0x7FF);
    let img = decode(&stream).expect("DC +2047 stream decodes");
    assert!(matches!(img, Jpeg::Rgb(_)));
}

#[test]
fn huge_negative_dc_never_panics() {
    // Payload 0x400 = -1024: exercises the negative-extend path and
    // the range-limit wrap regime; only panic-freedom is asserted.
    let stream = dc11_stream(0x400);
    let _ = decode(&stream).expect("DC -1024 stream decodes");
}
