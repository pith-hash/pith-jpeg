'use strict';
/** Hex-exact conformance: the committed reference vectors through koffi.
 *
 * Every digest in the repository-root `reference.json` is replayed
 * against its committed fixture file and compared bit-for-bit — the
 * cdylib computes the fnv1a64 over the decoded sample bytes, so the
 * comparison is the raw u64. The four numeric `vectors` are replayed
 * through the kernel entry points: bit-exact on the integer pipeline,
 * within the recorded tolerance on the platform-shaped f64 oracle.
 * The same contract the Rust `gen-reference verify` gate and the
 * Python/Go SDKs check.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const jpeg = require('../index.js');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'reference.json'), 'utf8'));
const DIGESTS = REFERENCE.digests;
const VECTORS = REFERENCE.vectors;

function fixtureBytes(file) {
  return fs.readFileSync(path.join(REPO_ROOT, 'tests', 'fixtures', file));
}

function lanesFromHex(inputHex) {
  return inputHex.map((h) => Buffer.from(h, 'hex').readDoubleBE(0));
}

test('cdylib is discoverable', () => {
  assert.match(jpeg.findCdylib(), /pith_jpeg\.(dll|so|dylib)$/);
});

for (const name of Object.keys(DIGESTS).sort()) {
  test(`digest ${name} is reproduced bit-exact`, () => {
    const data = fixtureBytes(`${name}.jpg`);
    const native = jpeg.channels(data);
    const layout = native === 1 ? jpeg.LAYOUT_GRAY8 : jpeg.LAYOUT_RGB8;
    const digest = jpeg.decodeDigest(data, layout);
    assert.equal(digest, BigInt('0x' + DIGESTS[name]), name);
  });
}

test('pinned literal base_gray', () => {
  const data = fixtureBytes('base_gray.jpg');
  assert.equal(jpeg.decodeDigest(data, jpeg.LAYOUT_GRAY8), 0xe0ce77f00e2cb066n);
});

test('native channels report', () => {
  assert.equal(jpeg.channels(fixtureBytes('base_gray.jpg')), 1);
  assert.equal(jpeg.channels(fixtureBytes('base_444.jpg')), 3);
});

test('layout conversions widen', () => {
  const data = fixtureBytes('base_gray.jpg');
  const digests = new Set([
    jpeg.decodeDigest(data, jpeg.LAYOUT_GRAY8),
    jpeg.decodeDigest(data, jpeg.LAYOUT_RGB8),
    jpeg.decodeDigest(data, jpeg.LAYOUT_RGBA8),
  ]);
  assert.equal(digests.size, 3);
});

test('rgb to gray is refused', () => {
  const data = fixtureBytes('base_444.jpg');
  assert.throws(() => jpeg.decodeDigest(data, jpeg.LAYOUT_GRAY8), (err) => {
    assert.equal(err.status, jpeg.STATUS_REJECTED);
    return true;
  });
});

test('malformed input is refused', () => {
  assert.throws(() => jpeg.decodeDigest(Buffer.from('not a jpeg at all'), jpeg.LAYOUT_GRAY8), {
    status: jpeg.STATUS_REJECTED,
  });
  assert.throws(() => jpeg.decodeDigest(Buffer.alloc(0), jpeg.LAYOUT_GRAY8), {
    status: jpeg.STATUS_REJECTED,
  });
});

test('unknown layout is invalid', () => {
  const data = fixtureBytes('base_gray.jpg');
  assert.throws(() => jpeg.decodeDigest(data, 9), { status: jpeg.STATUS_INVALID });
});

for (const name of Object.keys(VECTORS).sort()) {
  test(`kernel vector ${name} is reproduced`, () => {
    const vector = VECTORS[name];
    const input = lanesFromHex(vector.input);
    const tolAbs = Buffer.from(vector.tol_abs, 'hex').readDoubleBE(0);
    const tolRel = Buffer.from(vector.tol_rel, 'hex').readDoubleBE(0);

    let got;
    if (name === 'dequant.zigzag.8x8') got = jpeg.dequantZigzag(input);
    else if (name === 'dequant.8x8') got = jpeg.dequant(input.slice(0, 64), input.slice(64));
    else if (name === 'idct.islow.8x8') got = jpeg.idctIslow(input.slice(0, 64), input.slice(64));
    else got = jpeg.idctOracle(input);

    assert.equal(got.length, vector.output.length, name);
    got.forEach((g, i) => {
      const want = Buffer.from(vector.output[i], 'hex');
      if (vector.exact) {
        const gotBuf = Buffer.alloc(8);
        gotBuf.writeDoubleLE(g, 0);
        assert.equal(gotBuf.readBigUint64LE(0), want.readBigUint64BE(0), `${name} lane ${i}`);
      } else {
        const w = want.readDoubleBE(0);
        const budget = Math.max(tolAbs, tolRel * Math.abs(w));
        assert.ok(Math.abs(g - w) <= budget, `${name} lane ${i}: ${g} vs ${w}`);
      }
    });
  });
}
