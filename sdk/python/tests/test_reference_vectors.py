# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""Hex-exact conformance: the committed reference vectors through ctypes.

Every digest in the repository-root ``reference.json`` is replayed
against its committed fixture file and compared bit-for-bit — the
cdylib computes the ``fnv1a64`` over the decoded sample bytes, so the
comparison is the raw u64. The four numeric ``vectors`` are replayed
through the kernel entry points: bit-exact on the integer pipeline,
within the recorded tolerance on the platform-shaped ``f64`` oracle.
The same contract the Rust ``gen-reference verify`` gate and the
Node/Go SDKs check.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from pith_jpeg import (
    LAYOUT_GRAY8,
    LAYOUT_RGB8,
    STATUS_INVALID,
    STATUS_REJECTED,
    FfiError,
    channels,
    decode_digest,
    dequant,
    dequant_zigzag,
    find_cdylib,
    idct_islow,
    idct_oracle,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = json.loads((REPO_ROOT / "reference.json").read_text(encoding="utf-8"))
DIGESTS = REFERENCE["digests"]
VECTORS = REFERENCE["vectors"]


def fixture_bytes(file: str) -> bytes:
    return (REPO_ROOT / "tests" / "fixtures" / file).read_bytes()


def lanes_to_values(input_hex: list[str]) -> list[float]:
    """Decodes the reference file's hex f64 bit patterns to floats.

    The hex strings are the ``format!("{:016x}", bits)`` render of the
    u64 bit pattern — big-endian digits — so unpack ``>d``.
    """
    return [struct.unpack(">d", bytes.fromhex(h))[0] for h in input_hex]


def test_cdylib_is_discoverable() -> None:
    path = find_cdylib()
    assert path.is_file(), path


@pytest.mark.parametrize("name", sorted(DIGESTS))
def test_digest_is_reproduced_bit_exact(name: str) -> None:
    # reference.json records only name→digest; fixture files are
    # <name>.jpg, so the name is the file stem.
    data = fixture_bytes(f"{name}.jpg")
    native = channels(data)
    layout = LAYOUT_GRAY8 if native == 1 else LAYOUT_RGB8
    assert f"{decode_digest(data, layout):016x}" == DIGESTS[name], name


def test_pinned_literal_base_gray() -> None:
    # The drift guard the Rust unit tests re-derive: reference.json
    # digests["base_gray"].
    data = fixture_bytes("base_gray.jpg")
    assert f"{decode_digest(data, LAYOUT_GRAY8):016x}" == "e0ce77f00e2cb066"


def test_native_channels_report() -> None:
    assert channels(fixture_bytes("base_gray.jpg")) == 1
    assert channels(fixture_bytes("base_444.jpg")) == 3


def test_layout_conversions_widen() -> None:
    data = fixture_bytes("base_gray.jpg")
    native = decode_digest(data, LAYOUT_GRAY8)
    rgb = decode_digest(data, LAYOUT_RGB8)
    rgba = decode_digest(data, 4)
    assert len({native, rgb, rgba}) == 3


def test_rgb_to_gray_is_refused() -> None:
    data = fixture_bytes("base_444.jpg")
    with pytest.raises(FfiError) as err:
        decode_digest(data, LAYOUT_GRAY8)
    assert err.value.status == STATUS_REJECTED


def test_malformed_input_is_refused() -> None:
    with pytest.raises(FfiError) as err:
        decode_digest(b"not a jpeg at all", LAYOUT_GRAY8)
    assert err.value.status == STATUS_REJECTED
    with pytest.raises(FfiError) as err:
        decode_digest(b"", LAYOUT_GRAY8)
    assert err.value.status == STATUS_REJECTED


def test_unknown_layout_is_invalid() -> None:
    data = fixture_bytes("base_gray.jpg")
    with pytest.raises(FfiError) as err:
        decode_digest(data, 9)
    assert err.value.status == STATUS_INVALID


@pytest.mark.parametrize("name", sorted(VECTORS))
def test_kernel_vector_is_reproduced(name: str) -> None:
    vector = VECTORS[name]
    input_values = lanes_to_values(vector["input"])
    tol_abs = struct.unpack("<d", bytes.fromhex(vector["tol_abs"]))[0]
    tol_rel = struct.unpack("<d", bytes.fromhex(vector["tol_rel"]))[0]
    expected = lanes_to_values(vector["output"])

    if name == "dequant.zigzag.8x8":
        got = dequant_zigzag(input_values)
    elif name == "dequant.8x8":
        got = dequant(input_values[:64], input_values[64:])
    elif name == "idct.islow.8x8":
        got = idct_islow(input_values[:64], input_values[64:])
    else:
        got = idct_oracle(input_values)

    assert len(got) == len(expected), name
    for i, (g, w) in enumerate(zip(got, expected, strict=True)):
        if vector["exact"]:
            assert struct.pack(">d", g).hex() == vector["output"][i], f"{name} lane {i}"
        else:
            budget = max(tol_abs, tol_rel * abs(w))
            assert math.isclose(g, w, rel_tol=0.0, abs_tol=budget) or abs(g - w) <= budget, (
                f"{name} lane {i}: {g} vs {w}"
            )
