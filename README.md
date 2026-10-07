<p align="center">
  <img src="https://pith-jpeg.n24q02m.com/logo.svg" alt="pith-jpeg" width="120">
</p>

<h1 align="center">pith-jpeg</h1>

<p align="center">
  <strong>pith jpeg lane: baseline and progressive JPEG decoding, byte-exact with libjpeg islow (zero-dep Rust)</strong>
</p>

<p align="center">
  <a href="https://github.com/pith-hash/pith-jpeg/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/pith-hash/pith-jpeg/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/pith-hash/pith-jpeg/actions/workflows/cd.yml"><img alt="CD" src="https://github.com/pith-hash/pith-jpeg/actions/workflows/cd.yml/badge.svg"></a>
  <a href="https://github.com/pith-hash/pith-jpeg/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/pith-hash/pith-jpeg?display_name=tag&sort=semver"></a>
  <a href="https://github.com/n24q02m/better-semantic-release"><img alt="semantic-release" src="https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/pith-hash/pith-jpeg"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-pith-suite-contract">Suite contract</a>
</p>

<!-- BEGIN: AUTO-GENERATED-CROSS-PROMO -->
<!-- END: AUTO-GENERATED-CROSS-PROMO -->

## The pith suite contract

pith-jpeg is part of the **pith** suite (pith-hash). Every suite repository
follows the same rules; CI enforces them mechanically:

- **Naming**: a library is always `pith-<domain>` (`pith-image`, `pith-audio`,
  `pith-zip`, ...). The curator/repository of repositories is the bare
  `pith-hash`. Never invent a second naming scheme inside the suite.
- **Version pinning**: cross-library dependencies pin `~0.1` (e.g.
  `pith-image = { version = "~0.1", path = "../pith-image" }`). The whole suite
  moves together inside 0.1.x; breaking changes require a suite-wide version
  bump, never a silent minor drift.
- **Zero third-party dependencies**: every crate depends only on other
  `pith-*` crates plus `std`. `scripts/check-zero-deps.py` (run in CI) fails
  the build on any other crate, for normal, build and dev dependencies alike.
- **No unsafe**: every crate root carries `#![forbid(unsafe_code)]`.
- **Hex-exact vectors**: `reference.json` at the repo root is the
  cross-language source of truth. The `gen-reference` binary regenerates it;
  CI verifies the committed copy is current (`gen-reference verify`), and CD
  ships the regenerated file with every SDK artifact. Python, Node and Go SDKs
  MUST test against the same bytes.

## Repository layout

```
crates/            one published crate per suite lib (pith-<domain>)
tools/gen-reference  the vector generator binary (bin name: gen-reference)
sdk/python         ctypes wheel; build backend reads PITH_CDYLIB_DIR
sdk/node           koffi-based package; prebuilds/<os-arch>/ carry the cdylib
sdk/go             cgo binding; go.mod carries the module's cgo flags
fuzz/corpus        fuzz inputs, replayed by tests/fuzz_corpus.rs (parser crates)
reference.json     hex-exact cross-SDK test vectors
```

## Overview

JPEG decoding per ITU T.81: baseline (SOF0), extended sequential (SOF1)
and progressive (SOF2) Huffman-coded frames — spectral selection,
successive approximation and EOBRUN included — with restart markers,
4:4:4/4:2:2/4:4:0/4:2:0 and other integral sampling factors, and 1- or
3-component output as `pith_image::raster::Image` (`Gray`/`Rgb`, 8-bit).

The pixel pipeline is a byte-exact transcription of libjpeg-turbo's
`islow` integer IDCT, `fancy` triangle chroma upsampling and
fixed-point YCbCr→RGB, validated against Pillow 12.3.0 fixtures byte
for byte (see `tests/fixtures/PROVENANCE.md`). Arithmetic-coded,
lossless, differential, hierarchical, 12-bit and CMYK/YCCK streams are
refused with `Error::Unsupported` naming the coding process.

## Install

Rust (the core library):

```bash
cargo add pith-jpeg
```

Python / Node / Go SDKs are published from the same cdylib on every release;
see the release assets or the package registries for the matching version.

## Quick start

Rust (the core library):

```rust
let img = pith_jpeg::decode(&bytes)?;

match img {
    pith_jpeg::Jpeg::Gray(g) => println!("{}x{}", g.width(), g.height()),
    pith_jpeg::Jpeg::Rgb(rgb) => println!("{}x{}", rgb.width(), rgb.height()),
}
```

`decode` never panics on untrusted input: malformed structure surfaces
as `Error::Truncated` / `Error::BadValue`, refusals as
`Error::Unsupported`. Hex-exact cross-SDK vectors — decode digests for
every committed fixture plus the DCT/dequant numerics — live in
`reference.json` (regenerate with `cargo run --bin gen-reference`,
verify with `cargo run --bin gen-reference -- verify`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © pith-hash
