//! The committed reference contract of this crate, re-expressed over
//! the crate's own pipeline.
//!
//! [`tools/gen-reference`](../../tools/gen-reference) recomputes
//! everything in this module and either writes `reference.json` (repo
//! root) or verifies the committed copy against the recomputation;
//! CI runs the verify mode on every commit, and CD ships the file
//! with every SDK artifact. Python, Node and Go SDKs test against the
//! same bytes.
//!
//! # File schema (version 1)
//!
//! Two sections, both sorted by name, two-space indentation, LF line
//! endings, one trailing newline — byte-identical across
//! regenerations:
//!
//! * `digests` — one entry per committed `tests/fixtures/*.jpg`
//!   conformance fixture: [`pith_digest::fnv1a64`] over the decoded
//!   sample bytes in row-major, channel-interleaved order (the crate's
//!   own output layout, exactly what the `.raw` companions hold).
//!   Digests must match bit-for-bit.
//! * `vectors` — the DCT/dequant numerics in the suite's numeric
//!   schema (the `pith-math` convention): every `f64` is its raw
//!   IEEE-754 bit pattern as 16-digit lowercase hex, so consumers
//!   never see a decimal round-trip. Each vector records `exact`
//!   plus the two tolerance budgets `tol_abs`/`tol_rel` (hex `f64`
//!   like everything else). Verification policy:
//!   * `exact: true` — the pipeline is pure integer arithmetic
//!     (de-zigzag, dequantize, the islow IDCT transcription), so any
//!     correct implementation must reproduce the recorded bits.
//!   * `exact: false` — the `f64` trig kernel
//!     ([`pith_math::idct2_2d`], the orthonormal oracle documented in
//!     [`crate`]) depends on the platform `libm`'s `sin`/`cos`, which
//!     may differ in the last ulp across platforms; verification
//!     bounds the per-value error by
//!     `max(tol_abs, tol_rel · |expected|)`. The absolute floor
//!     (1e-13) covers the cancellation noise of near-zero bins, where
//!     platform differences move the value by a few ulps of the
//!     *summands*, not of the tiny result.
//!
//! Vector input layouts (documented here because the file carries no
//! per-field prose): all inputs are built from integer arithmetic
//! alone, so input generation is itself platform-independent.
//!
//! * `dequant.zigzag.8x8` — input: 64 DQT-payload values in JPEG scan
//!   order; output: the same values in natural (row-major) order,
//!   i.e. `out[ZIGZAG[k]] = in[k]`.
//! * `dequant.8x8` — input: 64 natural-order coefficients followed by
//!   64 natural-order quantizer values (one flat 128-value buffer);
//!   output: the 64 `i64` products, exactly representable in `f64`.
//! * `idct.islow.8x8` — same input layout as `dequant.8x8`; output:
//!   the 64 `u8` samples of the [`crate::idct`] `islow` transcription
//!   (dequantize + two-pass fixed-point IDCT + level shift + clamp),
//!   written into a stride-8 block.
//! * `idct.oracle.8x8` — input: the 64 *dequantized* coefficients
//!   (the `dequant.8x8` output, so SDK tests chain the two); output:
//!   the raw [`pith_math::idct2_2d`] result, no level shift, no clamp.

use crate::decode;
use crate::huffman::ZIGZAG;
use crate::idct::dequant_idct_into;
use pith_digest::fnv1a64;
use pith_math::idct2_2d;

/// Absolute tolerance floor for the `f64` oracle vector: covers the
/// cancellation noise of near-zero trig bins (see the module docs).
pub const TOL_ABS: f64 = 1e-13;

/// Relative tolerance for the `f64` oracle vector, applied to the
/// expected magnitude.
pub const TOL_REL: f64 = 1e-12;

/// One recorded numeric vector: the named kernel, its input, and the
/// output the committed file must carry.
#[derive(Clone, Debug)]
pub struct Vector {
    /// Vector name as it appears in `reference.json`.
    pub name: &'static str,
    /// Whether the output must match bit-for-bit (integer pipeline) or
    /// within the recorded tolerance (`f64` trig kernels).
    pub exact: bool,
    /// `[w, h]` of the recorded matrix, when the vector is a 2D block.
    pub shape: Option<[usize; 2]>,
    /// Tolerance budgets (both recorded per vector even when
    /// [`Vector::exact`] is true).
    pub tol_abs: f64,
    /// See [`Vector::tol_abs`].
    pub tol_rel: f64,
    /// Kernel input, hex-recorded in the file.
    pub input: Vec<f64>,
    /// Kernel output the committed file must carry.
    pub output: Vec<f64>,
}

/// One committed conformance fixture and its expected decode digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest {
    /// Digest name as it appears in `reference.json`.
    pub name: &'static str,
    /// Fixture file under `tests/fixtures/`.
    pub file: &'static str,
}

/// The eleven committed fixtures, in fixture-table order (the
/// conformance suite's own order); [`reference_json`] sorts by name
/// when it serializes.
#[must_use]
pub fn digests() -> &'static [Digest] {
    DIGESTS
}

const DIGESTS: &[Digest] = &[
    Digest {
        name: "base_420_odd",
        file: "base_420_odd.jpg",
    },
    Digest {
        name: "base_420_rst",
        file: "base_420_rst.jpg",
    },
    Digest {
        name: "base_422",
        file: "base_422.jpg",
    },
    Digest {
        name: "base_422_odd",
        file: "base_422_odd.jpg",
    },
    Digest {
        name: "base_440",
        file: "base_440.jpg",
    },
    Digest {
        name: "base_444",
        file: "base_444.jpg",
    },
    Digest {
        name: "base_gray",
        file: "base_gray.jpg",
    },
    Digest {
        name: "prog_420",
        file: "prog_420.jpg",
    },
    Digest {
        name: "prog_444",
        file: "prog_444.jpg",
    },
    Digest {
        name: "prog_444_rst",
        file: "prog_444_rst.jpg",
    },
    Digest {
        name: "prog_gray",
        file: "prog_gray.jpg",
    },
];

/// Deterministic coefficient block, natural order: `i32` values in
/// `[-512, 511]` from an integer LCG — the signed-11-bit coefficient
/// range the entropy decoder can emit.
fn coef_block() -> Vec<f64> {
    (0..64)
        .map(|i: i32| (((i * 37 + 11) % 1024) - 512) as f64)
        .collect()
}

/// Deterministic quantizer block, natural order: `u16` values in
/// `[1, 32]` — a typical mid-quality table's magnitude range.
fn quant_block() -> Vec<f64> {
    (0..64)
        .map(|i: u16| ((i * 17 + 3) % 32 + 1) as f64)
        .collect()
}

/// Deterministic DQT-payload block, JPEG scan order: `u8` values.
fn scan_payload() -> Vec<f64> {
    (0..64)
        .map(|k: u16| f64::from(((k * 91 + 13) % 256) as u8))
        .collect()
}

/// The dequantized coefficient block the IDCT vectors share.
fn dequantized() -> Vec<f64> {
    let coefs = coef_block();
    let qt = quant_block();
    coefs.iter().zip(qt.iter()).map(|(&c, &q)| c * q).collect()
}

/// Every numeric vector the suite shares, recomputed on each
/// invocation.
#[must_use]
pub fn vectors() -> Vec<Vector> {
    let mut v: Vec<Vector> = Vec::new();

    // -- De-zigzag (the DQT payload walk) --------------------------------
    let payload = scan_payload();
    let mut natural = vec![0.0; 64];
    for (k, &val) in payload.iter().enumerate() {
        natural[ZIGZAG[k] as usize] = val;
    }
    v.push(Vector {
        name: "dequant.zigzag.8x8",
        exact: true,
        shape: Some([8, 8]),
        tol_abs: TOL_ABS,
        tol_rel: TOL_REL,
        input: payload,
        output: natural,
    });

    // -- Dequantize (element-wise product, exact in f64) -----------------
    let coefs = coef_block();
    let qt = quant_block();
    let products = dequantized();
    let mut input = coefs;
    input.extend_from_slice(&qt);
    v.push(Vector {
        name: "dequant.8x8",
        exact: true,
        shape: None,
        tol_abs: TOL_ABS,
        tol_rel: TOL_REL,
        input,
        output: products.clone(),
    });

    // -- islow IDCT (fixed-point transcription, byte-exact) --------------
    let coefs = coef_block();
    let qt = quant_block();
    let coefs: Vec<i32> = coefs.iter().map(|&c| c as i32).collect();
    let qt: Vec<u16> = qt.iter().map(|&q| q as u16).collect();
    let mut plane = [0u8; 64];
    dequant_idct_into(&coefs, &qt, &mut plane, 8, 0, 0);
    v.push(Vector {
        name: "idct.islow.8x8",
        exact: true,
        shape: None,
        tol_abs: TOL_ABS,
        tol_rel: TOL_REL,
        input: {
            let mut input = coefs.iter().map(|&c| f64::from(c)).collect::<Vec<f64>>();
            input.extend(qt.iter().map(|&q| f64::from(q)));
            input
        },
        output: plane.iter().map(|&s| f64::from(s)).collect(),
    });

    // -- Orthonormal f64 oracle (platform-libm-shaped) -------------------
    let mut f = dequantized();
    idct2_2d(&mut f, 8, 8);
    v.push(Vector {
        name: "idct.oracle.8x8",
        exact: false,
        shape: Some([8, 8]),
        tol_abs: TOL_ABS,
        tol_rel: TOL_REL,
        input: dequantized(),
        output: f,
    });

    v
}

/// Decodes a committed fixture and digests the pixel output.
///
/// The file is read relative to the crate manifest directory (the same
/// convention the conformance tests use), so both `cargo test` and
/// `cargo run --bin gen-reference` resolve it from a repository
/// checkout.
///
/// # Panics
///
/// Panics if the fixture is unreadable or fails to decode: the
/// committed corpus must always decode, and a generator that cannot
/// reproduce a digest must be loud, not silent.
#[must_use]
pub fn digest_of(d: &Digest) -> u64 {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(d.file);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    match decode(&bytes) {
        Ok(crate::Jpeg::Gray(img)) => fnv1a64(img.as_slice()),
        Ok(crate::Jpeg::Rgb(img)) => fnv1a64(img.as_slice()),
        Err(why) => panic!("fixture {} no longer decodes: {why}", d.file),
    }
}

/// Every fixture digest, recomputed.
#[must_use]
pub fn digest_values() -> Vec<(&'static str, u64)> {
    DIGESTS.iter().map(|d| (d.name, digest_of(d))).collect()
}

/// `f64` → canonical 16-digit lowercase hex of the raw bit pattern.
#[must_use]
pub fn hx(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn push_f64s(out: &mut String, vs: &[f64], indent: &str) {
    out.push_str("[\n");
    for v in vs {
        out.push_str(indent);
        out.push('"');
        out.push_str(&hx(*v));
        out.push_str("\",\n");
    }
    // Trim the trailing comma of the last element for strict JSON.
    if !vs.is_empty() {
        let l = out.len() - 2;
        out.truncate(l);
        out.push('\n');
    }
    out.push_str(indent.trim_end());
    out.push(']');
}

/// Serializes the canonical `reference.json` bytes.
///
/// Deterministic: sections sorted by name, two-space indentation, a
/// single trailing newline — byte-identical across regenerations on
/// the same platform. The `f64` oracle vector's bits are
/// platform-libm-shaped by design; cross-platform verification is
/// semantic ([`verify_str`]), never a byte compare of the whole file.
///
/// # Panics
///
/// Panics if a committed fixture is unreadable or fails to decode
/// (see [`digest_of`]).
#[must_use]
pub fn reference_json() -> String {
    let mut digests: Vec<(&str, u64)> = digest_values();
    digests.sort_unstable_by_key(|&(name, _)| name);
    let mut vectors = vectors();
    vectors.sort_unstable_by(|a, b| a.name.cmp(b.name));

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"schema\": 1,\n");
    out.push_str("  \"suite\": \"pith-jpeg\",\n");
    out.push_str(
        "  \"note\": \"every f64 is its raw IEEE-754 bit pattern as 16-digit \
lowercase hex; exact vectors must match bit-for-bit, the orthonormal-oracle \
vector within max(tol_abs, tol_rel * |expected|); digests are fnv1a64 over \
the decoded sample bytes (row-major, channel-interleaved); dequant.8x8 and \
idct.islow.8x8 inputs are 64 coefficients followed by 64 quant values, \
natural order\",\n",
    );
    out.push_str("  \"digests\": {\n");
    for (i, (name, digest)) in digests.iter().enumerate() {
        out.push_str(&format!("    \"{name}\": \"{digest:016x}\""));
        out.push_str(if i + 1 == digests.len() { "\n" } else { ",\n" });
    }
    out.push_str("  },\n");
    out.push_str("  \"vectors\": {\n");
    for (i, v) in vectors.iter().enumerate() {
        out.push_str("    \"");
        out.push_str(v.name);
        out.push_str("\": {\n");
        out.push_str("      \"exact\": ");
        out.push_str(if v.exact { "true" } else { "false" });
        out.push_str(",\n");
        if let Some([w, h]) = v.shape {
            out.push_str(&format!("      \"shape\": [{w}, {h}],\n"));
        }
        out.push_str("      \"tol_abs\": \"");
        out.push_str(&hx(v.tol_abs));
        out.push_str("\",\n");
        out.push_str("      \"tol_rel\": \"");
        out.push_str(&hx(v.tol_rel));
        out.push_str("\",\n");
        out.push_str("      \"input\": ");
        push_f64s(&mut out, &v.input, "        ");
        out.push_str(",\n");
        out.push_str("      \"output\": ");
        push_f64s(&mut out, &v.output, "        ");
        out.push('\n');
        out.push_str("    }");
        out.push_str(if i + 1 == vectors.len() { "\n" } else { ",\n" });
    }
    out.push_str("  }\n}\n");
    out
}

/// Compares the committed `reference.json` against a fresh
/// recomputation, per-value and per-policy (never a whole-file byte
/// compare: the oracle vector's bits are platform-libm-shaped).
///
/// # Errors
///
/// A description naming the first divergence per offending vector, or
/// the read failure.
///
/// # Panics
///
/// Panics if a committed fixture is unreadable or fails to decode
/// (see [`digest_of`]).
pub fn verify() -> Result<(), String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reference.json");
    let committed = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    verify_str(&committed)
}

/// One vector as parsed back out of the committed file.
struct Committed {
    exact: bool,
    tol_abs: f64,
    tol_rel: f64,
    input: Vec<f64>,
    output: Vec<f64>,
}

/// The comparison underlying [`verify`]: a committed render against a
/// fresh recomputation.
///
/// # Errors
///
/// A description naming every divergence found.
///
/// # Panics
///
/// Panics if a committed fixture is unreadable or fails to decode
/// (see [`digest_of`]).
pub fn verify_str(committed: &str) -> Result<(), String> {
    let mut p = Json::new(committed);
    p.expect(b'{')?;
    let mut committed_digests: Vec<(String, u64)> = Vec::new();
    let mut committed_vectors: Vec<(String, Committed)> = Vec::new();
    loop {
        match p.peek() {
            Some(b'}') => break,
            Some(b'"') => {
                let key = p.string()?;
                p.expect(b':')?;
                match key.as_str() {
                    "digests" => committed_digests = p.entries(Json::hex_u64)?,
                    "vectors" => committed_vectors = p.entries(Json::vector)?,
                    _ => p.skip_value()?,
                }
                if p.peek() == Some(b',') {
                    p.i += 1;
                }
            }
            got => {
                return Err(format!(
                    "expected key or }}, found {:?} at byte {}",
                    got.map(char::from),
                    p.i
                ));
            }
        }
    }

    let mut fails: Vec<String> = Vec::new();

    // -- digests: bit-exact ----------------------------------------------
    let fresh = digest_values();
    for (name, want) in &fresh {
        match committed_digests.iter().find(|(n, _)| n == name) {
            None => fails.push(format!("[digest {name}] missing from committed file")),
            Some((_, got)) => {
                if *got != *want {
                    fails.push(format!(
                        "[digest {name}] drifted: committed {got:016x}, recomputed {want:016x}"
                    ));
                }
            }
        }
    }
    for (name, _) in &committed_digests {
        if !fresh.iter().any(|&(n, _)| n == name.as_str()) {
            fails.push(format!("[digest {name}] not recomputable: unknown fixture"));
        }
    }

    // -- vectors ----------------------------------------------------------
    let fresh_vectors = vectors();
    for want in &fresh_vectors {
        let pos = committed_vectors
            .iter()
            .position(|(name, _)| name == want.name);
        let Some(idx) = pos else {
            fails.push(format!("[{}] missing from committed file", want.name));
            continue;
        };
        let (_, got) = committed_vectors.swap_remove(idx);
        if got.exact != want.exact {
            fails.push(format!("[{}] exact flag drifted", want.name));
        }
        if got.tol_abs.to_bits() != want.tol_abs.to_bits()
            || got.tol_rel.to_bits() != want.tol_rel.to_bits()
        {
            fails.push(format!("[{}] tolerance budgets drifted", want.name));
        }
        if got.input.len() != want.input.len() || got.output.len() != want.output.len() {
            fails.push(format!("[{}] length drifted", want.name));
            continue;
        }
        let input_drift = got
            .input
            .iter()
            .zip(&want.input)
            .any(|(g, w)| g.to_bits() != w.to_bits());
        if input_drift {
            fails.push(format!("[{}] input drifted", want.name));
        }
        for (i, (g, w)) in got.output.iter().zip(&want.output).enumerate() {
            if want.exact {
                if g.to_bits() != w.to_bits() {
                    fails.push(format!(
                        "[{}] output[{i}] drifted: committed {}, recomputed {}",
                        want.name,
                        hx(*g),
                        hx(*w)
                    ));
                    break;
                }
            } else {
                let budget = want.tol_abs.max(want.tol_rel * w.abs());
                let err = (g - w).abs();
                if err > budget {
                    fails.push(format!(
                        "[{}] output[{i}] drifted beyond tolerance: committed {}, recomputed {}, err {err:e}, budget {budget:e}",
                        want.name,
                        hx(*g),
                        hx(*w)
                    ));
                    break;
                }
            }
        }
    }
    for (name, _) in &committed_vectors {
        if !fresh_vectors.iter().any(|v| v.name == name.as_str()) {
            fails.push(format!("[{name}] not recomputable: unknown vector"));
        }
    }

    if fails.is_empty() {
        Ok(())
    } else {
        Err(fails.join("\n"))
    }
}

/// Tiny JSON walker for exactly the schema [`reference_json`] emits:
/// objects, arrays, strings, booleans — every numeric payload is a hex
/// string, so no float parsing ever happens.
struct Json<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Json<'a> {
    fn new(s: &'a str) -> Self {
        Json {
            b: s.as_bytes(),
            i: 0,
        }
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        match self.peek() {
            Some(got) if got == c => {
                self.i += 1;
                Ok(())
            }
            got => Err(format!(
                "expected {:?}, found {:?} at byte {}",
                c as char,
                got.map(char::from),
                self.i
            )),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'"' {
            self.i += 1;
        }
        if self.i >= self.b.len() {
            return Err("unterminated string".into());
        }
        let s = core::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        self.i += 1;
        Ok(s.to_owned())
    }

    fn hex_u64(&mut self) -> Result<u64, String> {
        let s = self.string()?;
        u64::from_str_radix(&s, 16).map_err(|e| format!("bad hex u64 {s:?}: {e}"))
    }

    fn hex_f64(&mut self) -> Result<f64, String> {
        let s = self.string()?;
        let bits = u64::from_str_radix(&s, 16).map_err(|e| format!("bad hex f64 {s:?}: {e}"))?;
        Ok(f64::from_bits(bits))
    }

    fn boolean(&mut self) -> Result<bool, String> {
        self.ws();
        if self.b[self.i..].starts_with(b"true") {
            self.i += 4;
            Ok(true)
        } else if self.b[self.i..].starts_with(b"false") {
            self.i += 5;
            Ok(false)
        } else {
            Err("expected boolean".into())
        }
    }

    fn number(&mut self) -> Result<i64, String> {
        self.ws();
        let start = self.i;
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_digit() {
            self.i += 1;
        }
        core::str::from_utf8(&self.b[start..self.i])
            .map_err(|e| e.to_string())?
            .parse::<i64>()
            .map_err(|e| format!("bad integer: {e}"))
    }

    fn f64_array(&mut self) -> Result<Vec<f64>, String> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(out);
        }
        loop {
            out.push(self.hex_f64()?);
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(out);
                }
                got => {
                    return Err(format!(
                        "expected , or ] in array, found {:?}",
                        got.map(char::from)
                    ));
                }
            }
        }
    }

    fn vector(&mut self) -> Result<Committed, String> {
        self.expect(b'{')?;
        let mut c = Committed {
            exact: false,
            tol_abs: 0.0,
            tol_rel: 0.0,
            input: Vec::new(),
            output: Vec::new(),
        };
        loop {
            match self.peek() {
                Some(b'}') => {
                    self.i += 1;
                    return Ok(c);
                }
                Some(b'"') => {
                    let key = self.string()?;
                    self.expect(b':')?;
                    match key.as_str() {
                        "exact" => c.exact = self.boolean()?,
                        "tol_abs" => c.tol_abs = self.hex_f64()?,
                        "tol_rel" => c.tol_rel = self.hex_f64()?,
                        "input" => c.input = self.f64_array()?,
                        "output" => c.output = self.f64_array()?,
                        "shape" => self.skip_value()?,
                        other => return Err(format!("unexpected vector key {other:?}")),
                    }
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    }
                }
                got => {
                    return Err(format!(
                        "expected key or }}, found {:?}",
                        got.map(char::from)
                    ));
                }
            }
        }
    }

    fn skip_value(&mut self) -> Result<(), String> {
        match self.peek() {
            Some(b'"') => {
                self.string()?;
            }
            // Arrays in this schema are hex-string arrays (input/output)
            // or bare-integer arrays (shape); both are skipped by
            // bracket counting — the schema's strings never contain
            // brackets.
            Some(b'[') => {
                self.i += 1;
                let mut depth = 1;
                while depth > 0 {
                    match self.peek() {
                        Some(b'[') => {
                            depth += 1;
                            self.i += 1;
                        }
                        Some(b']') => {
                            depth -= 1;
                            self.i += 1;
                        }
                        Some(_) => self.i += 1,
                        None => return Err("unterminated array".into()),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut depth = 1;
                while depth > 0 {
                    match self.peek() {
                        Some(b'{') => {
                            depth += 1;
                            self.i += 1;
                        }
                        Some(b'}') => {
                            depth -= 1;
                            self.i += 1;
                        }
                        Some(_) => self.i += 1,
                        None => return Err("unterminated object".into()),
                    }
                }
            }
            Some(b't' | b'f') => {
                self.boolean()?;
            }
            Some(c) if c.is_ascii_digit() || c == b'-' => {
                self.number()?;
            }
            got => return Err(format!("unexpected value start {:?}", got.map(char::from))),
        }
        Ok(())
    }

    /// Parses the `{ name: value, ... }` body of a section whose key
    /// and colon are already consumed, applying `item` to every entry.
    fn entries<T>(
        &mut self,
        item: fn(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<(String, T)>, String> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(b'}') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'"') => {
                    let name = self.string()?;
                    self.expect(b':')?;
                    let value = item(self)?;
                    out.push((name, value));
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    }
                }
                got => {
                    return Err(format!(
                        "expected key or }}, found {:?}",
                        got.map(char::from)
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Committed, Digest, Json, TOL_ABS, TOL_REL, digest_of, digest_values, hx, reference_json,
        vectors, verify_str,
    };

    /// The canonical render is deterministic: two runs agree.
    #[test]
    fn render_is_deterministic() {
        assert_eq!(reference_json(), reference_json());
    }

    /// The committed file verifies against a fresh recomputation.
    #[test]
    fn verify_accepts_current() {
        verify_str(&reference_json()).expect("current render verifies");
    }

    /// Replaces `from` with `to`, asserting `from` occurs exactly once
    /// so the perturbation lands where the test aims.
    fn replace_unique(committed: &str, from: &str, to: &str) -> String {
        assert_eq!(committed.matches(from).count(), 1, "{from} is not unique");
        committed.replacen(from, to, 1)
    }

    /// The first digest whose hex occurs exactly once in the render.
    fn unique_digest_hex(committed: &str) -> (&'static str, String) {
        digest_values()
            .iter()
            .map(|&(name, digest)| (name, format!("{digest:016x}")))
            .find(|(_, hex)| committed.matches(hex.as_str()).count() == 1)
            .expect("a digest unique in the render")
    }

    /// The first oracle output whose hex occurs exactly once in the
    /// render.
    fn unique_oracle_hex(committed: &str) -> (f64, String) {
        vectors()
            .into_iter()
            .find(|v| v.name == "idct.oracle.8x8")
            .expect("oracle vector")
            .output
            .into_iter()
            .map(|v| (v, hx(v)))
            .find(|(_, hex)| committed.matches(hex.as_str()).count() == 1)
            .expect("an oracle output unique in the render")
    }

    /// A stale digest fails verification by name.
    #[test]
    fn verify_rejects_stale_digest() {
        let committed = reference_json();
        let (name, hex) = unique_digest_hex(&committed);
        let stale = replace_unique(&committed, &hex, "0000000000000000");
        let err = verify_str(&stale).expect_err("stale digest must fail");
        assert!(err.contains(&format!("[digest {name}]")), "{err}");
    }

    /// A drifted exact vector fails bit-for-bit.
    #[test]
    fn verify_rejects_drifted_exact_output() {
        let exact: Vec<_> = vectors().into_iter().filter(|v| v.exact).collect();
        let committed = reference_json();
        // Target an output sample whose hex occurs exactly once in the
        // render, so the drift lands in an exact vector's output, not
        // in some other array.
        let (name, committed_hex, drifted_hex) = exact
            .iter()
            .find_map(|v| {
                v.output.iter().find_map(|&s| {
                    let hex = hx(s);
                    let drifted = hx(f64::from_bits(s.to_bits() ^ 1));
                    (committed.matches(hex.as_str()).count() == 1).then_some((v.name, hex, drifted))
                })
            })
            .expect("an exact output sample unique in the render");
        let stale = committed.replacen(&committed_hex, &drifted_hex, 1);
        let err = verify_str(&stale).expect_err("drifted exact output must fail");
        assert!(err.contains(name), "{err}");
    }

    /// An in-budget perturbation of the oracle output still verifies:
    /// the tolerance path is exercised, not just the bit path.
    #[test]
    fn verify_accepts_in_budget_oracle_drift() {
        let committed = reference_json();
        let (value, hex) = unique_oracle_hex(&committed);
        // One ulp: far inside max(tol_abs, tol_rel * |expected|) for
        // every magnitude in this block.
        let nudged = f64::from_bits(value.to_bits() + 1);
        let stale = replace_unique(&committed, &hex, &hx(nudged));
        verify_str(&stale).expect("one-ulp drift is inside budget");
    }

    /// An out-of-budget perturbation of the oracle output fails.
    #[test]
    fn verify_rejects_out_of_budget_oracle_drift() {
        let committed = reference_json();
        let (value, hex) = unique_oracle_hex(&committed);
        let blown = value + 1.0;
        let stale = replace_unique(&committed, &hex, &hx(blown));
        let err = verify_str(&stale).expect_err("unit drift must fail");
        assert!(err.contains("idct.oracle.8x8"), "{err}");
    }

    /// An unknown extra vector in the committed file is rejected.
    #[test]
    fn verify_rejects_unknown_vector() {
        let extra = format!(
            "    \"bogus.vector\": {{\n      \"exact\": true,\n      \"tol_abs\": \"{}\",\n      \"tol_rel\": \"{}\",\n      \"input\": [\"0000000000000000\"],\n      \"output\": [\"0000000000000000\"]\n    }},\n",
            hx(TOL_ABS),
            hx(TOL_REL)
        );
        let stale = reference_json().replacen(
            "  \"vectors\": {\n",
            &format!("  \"vectors\": {{\n{extra}"),
            1,
        );
        let err = verify_str(&stale).expect_err("unknown vector must fail");
        assert!(err.contains("bogus.vector"), "{err}");
    }

    /// Every digest recomputes to a stable, non-zero value.
    #[test]
    fn digests_are_stable_and_nonzero() {
        for d in digest_values() {
            assert_ne!(d.1, 0, "{} digests to zero", d.0);
        }
        let d = Digest {
            name: "x",
            file: "base_444.jpg",
        };
        assert_eq!(
            digest_of(&d),
            digest_values()
                .iter()
                .find(|(n, _)| *n == "base_444")
                .unwrap()
                .1
        );
    }

    /// The tolerance budgets match the suite convention.
    #[test]
    fn tolerance_budgets_are_the_suite_floors() {
        assert_eq!(hx(TOL_ABS), "3d3c25c268497682");
        assert_eq!(hx(TOL_REL), "3d719799812dea11");
    }

    /// The JSON walker reports malformed input instead of panicking.
    #[test]
    fn json_walker_rejects_garbage() {
        let mut p = Json::new("{");
        assert!(p.entries(Json::hex_u64).is_err());
        let mut p = Json::new("{\"a\": \"zz\"}");
        assert!(p.entries(Json::hex_u64).is_err());
        let mut p = Json::new("\"zz\" ");
        assert!(p.hex_u64().is_err());
        let mut p = Json::new("[1, 2]");
        assert!(p.f64_array().is_err());
        let mut p = Json::new("\"true\"");
        assert!(p.boolean().is_err());
        let mut p = Json::new("{} ");
        assert!(p.skip_value().is_ok());
        let mut p = Json::new("]");
        assert!(p.skip_value().is_err());
    }

    /// verify_str refuses structurally broken committed files.
    #[test]
    fn verify_rejects_broken_structure() {
        assert!(verify_str("").is_err()); // no object at all
        assert!(verify_str("[1]").is_err()); // not an object
        assert!(verify_str("{\"digests\": 5}").is_err()); // section not an object
        // An unknown top-level section carrying an object or an array
        // is skipped, then the missing-digest failure surfaces.
        assert!(verify_str("{\"bogus\": {\"a\": [1]}, \"tail\": [1, 2]}").is_err());
        // A vector entry with an unexpected key.
        assert!(verify_str("{\"vectors\": {\"v\": {\"bogus\": 1}}}").is_err());
        // A vector entry whose exact flag is not a boolean.
        assert!(verify_str("{\"vectors\": {\"v\": {\"exact\": 1}}}").is_err());
        // A digest entry that is not hex.
        assert!(verify_str("{\"digests\": {\"a\": \"zz\"}}").is_err());
        // A committed file with neither section fails on the digests.
        assert!(verify_str("{}").is_err());
    }

    /// digest_of must be loud about unreadable or undecodable
    /// fixtures: a generator that cannot reproduce a digest is a bug.
    #[test]
    #[should_panic(expected = "cannot read fixture")]
    fn missing_fixture_panics() {
        let _ = digest_of(&Digest {
            name: "absent",
            file: "does_not_exist.jpg",
        });
    }

    /// A fixture that exists but is not a JPEG panics with a decode
    /// diagnosis.
    #[test]
    #[should_panic(expected = "no longer decodes")]
    fn undecodable_fixture_panics() {
        let _ = digest_of(&Digest {
            name: "junk",
            file: "PROVENANCE.md",
        });
    }

    /// Vector defaults parse back from a minimal object.
    #[test]
    fn committed_defaults() {
        let mut p = Json::new("{}");
        let c: Committed = p.vector().expect("empty object");
        assert!(!c.exact);
        assert!(c.input.is_empty() && c.output.is_empty());
    }
}
