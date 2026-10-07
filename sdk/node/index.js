'use strict';
/**
 * pith-jpeg SDK: JPEG decoding and kernel replay through koffi.
 *
 * Decodes a native JPEG (baseline SOF0/SOF1 and progressive SOF2,
 * byte-exact with libjpeg `-dct int -nosmooth`) through the Rust
 * cdylib. The cdylib computes the fnv1a64 digest of the decoded sample
 * bytes internally, so this module stays zero-dependency: one call per
 * fixture returns the exact digest the `reference.json` `digests`
 * section pins. The four numeric kernels the `vectors` section pins
 * replay through the same boundary over little-endian f64 bit-pattern
 * buffers (8 bytes per lane; the reference file renders each lane as
 * big-endian hex digits, parsed with `BigInt('0x' + hex)`).
 *
 * The cdylib is located through the suite's discovery chain:
 * 1. `PITH_CDYLIB` — explicit cdylib file path;
 * 2. `PITH_CDYLIB_DIR` — directory scanned for the cdylib names (the
 *    CD pipeline points this at `target/release`);
 * 3. the module directory itself (a packaged npm module ships the
 *    cdylib as package data);
 * 4. `<repo root>/target/release` — the working-tree layout.
 */

/** Status: success. */
const STATUS_OK = 0;
/** Status: a caller argument is invalid — null pointer, unknown
 * layout, wrong buffer length. */
const STATUS_INVALID = -1;
/** Status: the decoder refused the input (malformed JPEG), or the
 * requested conversion is not offered (RGB→gray). */
const STATUS_REJECTED = -2;

/** Output layout: 8-bit grayscale (the pith-image code). */
const LAYOUT_GRAY8 = 0;
/** Output layout: 8-bit RGB, channel-interleaved (the pith-image
 * code). */
const LAYOUT_RGB8 = 2;
/** Output layout: 8-bit RGBA, channel-interleaved (the pith-image
 * code). */
const LAYOUT_RGBA8 = 4;

/** One 8×8 kernel block: 64 f64 lanes. */
const BLOCK = 64;

/** Every cdylib file name cargo may drop into the build directory,
 * per platform (windows / linux / macOS). */
const CDYLIB_NAMES = ['pith_jpeg.dll', 'libpith_jpeg.so', 'libpith_jpeg.dylib'];

const path = require('node:path');

/** An error class thrown for every non-zero status code. */
class FfiError extends Error {
  /**
   * @param {string} op the FFI operation name
   * @param {number} status the raw status code
   */
  constructor(op, status) {
    const detail =
      status === STATUS_INVALID
        ? 'invalid argument'
        : status === STATUS_REJECTED
          ? 'input rejected'
          : 'unknown failure';
    super(`${op} failed: ${detail} (status ${status})`);
    /** @type {string} */
    this.name = 'FfiError';
    /** The raw status code the FFI returned. @type {number} */
    this.status = status;
  }
}

/** An error class thrown when no cdylib is found. */
class LibraryNotFoundError extends Error {
  constructor() {
    super(
      'no pith-jpeg cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, ' +
        'the module directory and <repo>/target/release); ' +
        'run `cargo build --release` first',
    );
    /** @type {string} */
    this.name = 'LibraryNotFoundError';
  }
}

/**
 * Locates the cdylib through the suite's discovery chain.
 *
 * @returns {string} the absolute path to the cdylib file
 * @throws {LibraryNotFoundError} when nothing is found
 */
function findCdylib() {
  const fs = require('node:fs');
  const explicit = process.env.PITH_CDYLIB;
  if (explicit && fs.existsSync(explicit) && fs.statSync(explicit).isFile()) {
    return path.resolve(explicit);
  }
  /** @type {string[]} */
  const repoRoot = path.resolve(__dirname, '..', '..');
  const dirs = [];
  const envDir = process.env.PITH_CDYLIB_DIR;
  if (envDir) {
    dirs.push(envDir);
    if (!path.isAbsolute(envDir)) {
      // CD and local runs invoke tools from the repository root or
      // from sdk/<lang>; resolve the env value against both.
      dirs.push(path.resolve(envDir));
      dirs.push(path.join(repoRoot, envDir));
    }
  }
  dirs.push(__dirname); // packaged npm module
  dirs.push(path.join(repoRoot, 'target', 'release'));
  for (const dir of dirs) {
    for (const name of CDYLIB_NAMES) {
      const p = path.join(dir, name);
      if (fs.existsSync(p) && fs.statSync(p).isFile()) return p;
    }
  }
  throw new LibraryNotFoundError();
}

/** Lazily loaded, symbol-bound cdylib. @type {import('koffi').KoffiLibrary | null} */
let cached = undefined;

/**
 * Loads the cdylib and binds the exported symbols lazily, once.
 *
 * @returns {import('koffi').KoffiLibrary} the loaded library
 * @throws {LibraryNotFoundError} when nothing is found
 */
function loadLibrary() {
  if (cached !== undefined) return cached;
  const koffi = require('koffi');
  const lib = koffi.load(findCdylib());
  // Out-params are declared through koffi.out(koffi.pointer(...)) and
  // addressed through one-element array holders (read result[0]); the
  // kernel output buffers are caller-owned byte buffers written in
  // place.
  cached = {
    _decode: lib.func('pith_jpeg_decode', 'int32_t', [
      'uint8_t *',
      'uint64',
      'uint32',
      koffi.out(koffi.pointer('uint64')),
    ]),
    _channels: lib.func('pith_jpeg_channels', 'int32_t', [
      'uint8_t *',
      'uint64',
      koffi.out(koffi.pointer('uint32')),
    ]),
    _dequant_zigzag: lib.func('pith_jpeg_dequant_zigzag', 'int32_t', [
      'uint8_t *',
      'uint64',
      koffi.out(koffi.pointer('uint8_t')),
      'uint64',
    ]),
    _dequant: lib.func('pith_jpeg_dequant', 'int32_t', [
      'uint8_t *',
      'uint64',
      koffi.out(koffi.pointer('uint8_t')),
      'uint64',
    ]),
    _idct_islow: lib.func('pith_jpeg_idct_islow', 'int32_t', [
      'uint8_t *',
      'uint64',
      koffi.out(koffi.pointer('uint8_t')),
      'uint64',
    ]),
    _idct_oracle: lib.func('pith_jpeg_idct_oracle', 'int32_t', [
      'uint8_t *',
      'uint64',
      koffi.out(koffi.pointer('uint8_t')),
      'uint64',
    ]),
  };
  return cached;
}

/**
 * Packs f64 lanes into a little-endian bit-pattern buffer.
 *
 * @param {number[]} values one f64 per lane
 * @returns {Buffer} 8 bytes per lane, little-endian IEEE-754 bits
 */
function lanesToBuffer(values) {
  const buf = Buffer.alloc(values.length * 8);
  for (let i = 0; i < values.length; i++) {
    buf.writeDoubleLE(values[i], i * 8);
  }
  return buf;
}

/**
 * Unpacks f64 lanes from a little-endian bit-pattern buffer.
 *
 * @param {Buffer} buf 8 bytes per lane
 * @returns {number[]} one f64 per lane
 */
function bufferToLanes(buf) {
  /** @type {number[]} */
  const out = [];
  for (let i = 0; i + 8 <= buf.length; i += 8) {
    out.push(buf.readDoubleLE(i));
  }
  return out;
}

/**
 * Parses a reference.json f64 lane: big-endian hex digits of the u64
 * bit pattern.
 *
 * @param {string} hex 16 hex digits
 * @returns {{bits: bigint, value: number}} bits and the f64 value
 */
function parseLane(hex) {
  const bits = BigInt('0x' + hex);
  const be = Buffer.from(hex, 'hex');
  return { bits, value: be.readDoubleBE(0) };
}

/**
 * Decodes a JPEG and returns the fnv1a64 digest of the sample bytes in
 * `layout` encoding: row-major, channel-interleaved — the exact value
 * `reference.json` pins for the native layout.
 *
 * @param {Buffer} data the JPEG byte stream
 * @param {number} layout LAYOUT_GRAY8 / LAYOUT_RGB8 / LAYOUT_RGBA8
 * @returns {bigint} the 64-bit digest
 * @throws {FfiError} STATUS_REJECTED for malformed input and the
 *   refused RGB→gray conversion; STATUS_INVALID for an unknown layout
 */
function decodeDigest(data, layout) {
  const lib = loadLibrary();
  const out = [0n];
  const status = lib._decode(data, data.length, layout, out);
  if (status !== STATUS_OK) throw new FfiError('pith_jpeg_decode', status);
  return out[0];
}

/**
 * Returns the frame's native channel count (1 or 3) — the digest-exact
 * layout is LAYOUT_GRAY8 for 1 and LAYOUT_RGB8 for 3.
 *
 * @param {Buffer} data the JPEG byte stream
 * @returns {number} 1 or 3
 * @throws {FfiError} on refusal
 */
function channels(data) {
  const lib = loadLibrary();
  const out = [0];
  const status = lib._channels(data, data.length, out);
  if (status !== STATUS_OK) throw new FfiError('pith_jpeg_channels', status);
  return out[0];
}

/**
 * Feeds lanes to one kernel op and returns the decoded output lanes.
 *
 * @param {Function} op the bound kernel symbol
 * @param {number[]} values input lanes
 * @param {number} outLanes output lane count (64)
 * @returns {number[]} output lanes
 */
function replay(op, values, outLanes) {
  const lib = loadLibrary();
  const input = lanesToBuffer(values);
  const out = Buffer.alloc(outLanes * 8);
  const status = op(input, input.length, out, out.length);
  if (status !== STATUS_OK) throw new FfiError(op.name || 'kernel', status);
  return bufferToLanes(out);
}

/**
 * Replays the `dequant.zigzag.8x8` kernel: 64 DQT-payload values in
 * scan order in, the same values in natural (row-major) order out.
 *
 * @param {number[]} payload 64 f64 lanes
 * @returns {number[]} 64 f64 lanes
 */
function dequantZigzag(payload) {
  return replay(loadLibrary()._dequant_zigzag, payload, BLOCK);
}

/**
 * Replays the `dequant.8x8` kernel: 64 natural-order coefficients and
 * 64 quantizer values in, the 64 element-wise products out.
 *
 * @param {number[]} coefs 64 f64 lanes
 * @param {number[]} qt 64 f64 lanes
 * @returns {number[]} 64 f64 lanes
 */
function dequant(coefs, qt) {
  return replay(loadLibrary()._dequant, coefs.concat(qt), BLOCK);
}

/**
 * Replays the `idct.islow.8x8` kernel: the same input layout as
 * {@link dequant}, the 64 u8 samples of the fixed-point islow
 * transcription out as f64 lanes.
 *
 * @param {number[]} coefs 64 f64 lanes
 * @param {number[]} qt 64 f64 lanes
 * @returns {number[]} 64 f64 lanes
 */
function idctIslow(coefs, qt) {
  return replay(loadLibrary()._idct_islow, coefs.concat(qt), BLOCK);
}

/**
 * Replays the `idct.oracle.8x8` kernel: 64 dequantized coefficients
 * in, the raw orthonormal f64 DCT-III oracle result out.
 * Platform-libm-shaped: compare within the recorded tolerance, never
 * bit-for-bit.
 *
 * @param {number[]} block 64 f64 lanes
 * @returns {number[]} 64 f64 lanes
 */
function idctOracle(block) {
  return replay(loadLibrary()._idct_oracle, block, BLOCK);
}

module.exports = {
  STATUS_OK,
  STATUS_INVALID,
  STATUS_REJECTED,
  LAYOUT_GRAY8,
  LAYOUT_RGB8,
  LAYOUT_RGBA8,
  BLOCK,
  FfiError,
  LibraryNotFoundError,
  findCdylib,
  parseLane,
  decodeDigest,
  channels,
  dequantZigzag,
  dequant,
  idctIslow,
  idctOracle,
};
