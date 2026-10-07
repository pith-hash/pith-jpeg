# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""pith-jpeg SDK: JPEG decoding and kernel replay through ctypes.

Decodes a native JPEG (baseline SOF0/SOF1 and progressive SOF2,
byte-exact with libjpeg ``-dct int -nosmooth``) through the Rust
cdylib. The cdylib computes the ``fnv1a64`` digest of the decoded
sample bytes internally, so this package stays zero-dependency: one
call per fixture returns the exact digest the ``reference.json``
``digests`` section pins. The four numeric kernels the ``vectors``
section pins (de-zigzag, dequantize, the fixed-point islow IDCT and
the orthonormal ``f64`` oracle) replay through the same boundary over
little-endian ``f64`` bit-pattern buffers.

The cdylib is located through the suite's discovery chain:

1. ``PITH_CDYLIB`` — an explicit cdylib *file* path;
2. ``PITH_CDYLIB_DIR`` — a *directory* scanned for the cdylib names
   (the CD pipeline points this at ``target/release``);
3. the package directory itself (the built wheel ships the cdylib as
   package data);
4. ``<repo root>/target/release`` — the repository working-tree layout,
   so a source checkout runs against a local cargo build with no
   configuration.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

__all__ = [
    "FfiError",
    "LibraryNotFoundError",
    "find_cdylib",
    "decode_digest",
    "channels",
    "dequant_zigzag",
    "dequant",
    "idct_islow",
    "idct_oracle",
    "STATUS_OK",
    "STATUS_INVALID",
    "STATUS_REJECTED",
    "LAYOUT_GRAY8",
    "LAYOUT_RGB8",
    "LAYOUT_RGBA8",
    "BLOCK",
]

#: Status: success.
STATUS_OK = 0
#: Status: a caller argument is invalid — null pointer, unknown layout,
#: wrong buffer length.
STATUS_INVALID = -1
#: Status: the decoder refused the input (malformed JPEG), or the
#: requested conversion is not offered (RGB→gray).
STATUS_REJECTED = -2

#: Output layout: 8-bit grayscale (the ``pith-image`` code).
LAYOUT_GRAY8 = 0
#: Output layout: 8-bit RGB, channel-interleaved (the ``pith-image``
#: code).
LAYOUT_RGB8 = 2
#: Output layout: 8-bit RGBA, channel-interleaved (the ``pith-image``
#: code).
LAYOUT_RGBA8 = 4

#: One 8×8 kernel block: 64 ``f64`` values.
BLOCK = 64

#: Every cdylib file name cargo may drop into the build directory, per
#: platform (windows / linux / macOS).
CDYLIB_NAMES = ("pith_jpeg.dll", "libpith_jpeg.so", "libpith_jpeg.dylib")


class LibraryNotFoundError(OSError):
    """No cdylib was found through the discovery chain."""


class FfiError(Exception):
    """A non-zero status code came back from the cdylib."""

    def __init__(self, op: str, status: int) -> None:
        detail = {
            STATUS_INVALID: "invalid argument",
            STATUS_REJECTED: "input rejected",
        }.get(status, "unknown failure")
        super().__init__(f"{op} failed: {detail} (status {status})")
        #: The raw status code the FFI returned.
        self.status = status


def find_cdylib() -> Path:
    """Locates the cdylib through the suite's discovery chain."""
    explicit = os.environ.get("PITH_CDYLIB")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    env_dir = os.environ.get("PITH_CDYLIB_DIR")
    candidates: list[Path] = []
    if env_dir:
        env_dir_path = Path(env_dir)
        candidates.append(env_dir_path)
        if not env_dir_path.is_absolute():
            # CD and local runs invoke tools from the repository root or
            # from sdk/<lang>; resolve the env value against both.
            candidates.append(Path.cwd() / env_dir_path)
            candidates.append(Path(__file__).resolve().parents[3] / env_dir_path)
    candidates.append(Path(__file__).resolve().parent)  # packaged wheel
    candidates.append(Path(__file__).resolve().parents[3] / "target" / "release")
    for directory in candidates:
        for name in CDYLIB_NAMES:
            p = directory / name
            if p.is_file():
                return p
    raise LibraryNotFoundError(
        "no pith-jpeg cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, "
        "the package directory and <repo>/target/release); "
        "run `cargo build --release` first"
    )


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(find_cdylib()))
        lib.pith_jpeg_decode.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_size_t,  # len
            ctypes.c_uint32,  # layout
            ctypes.POINTER(ctypes.c_uint64),  # out digest
        ]
        lib.pith_jpeg_decode.restype = ctypes.c_int32
        lib.pith_jpeg_channels.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_size_t,  # len
            ctypes.POINTER(ctypes.c_uint32),  # out channels
        ]
        lib.pith_jpeg_channels.restype = ctypes.c_int32
        for op in (
            "pith_jpeg_dequant_zigzag",
            "pith_jpeg_dequant",
            "pith_jpeg_idct_islow",
            "pith_jpeg_idct_oracle",
        ):
            fn = getattr(lib, op)
            fn.argtypes = [
                ctypes.c_void_p,  # input buffer (LE f64 lanes)
                ctypes.c_size_t,  # input length in bytes
                ctypes.c_void_p,  # output buffer (LE f64 lanes)
                ctypes.c_size_t,  # output length in bytes
            ]
            fn.restype = ctypes.c_int32
        _lib = lib
    return _lib


def decode_digest(data: bytes, layout: int) -> int:
    """Decodes a JPEG and returns the ``fnv1a64`` digest of the sample
    bytes in ``layout`` encoding: row-major, channel-interleaved — the
    exact value ``reference.json`` pins for the native layout.

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for
    malformed input and for the refused RGB→gray conversion, and with
    ``status == STATUS_INVALID`` for an unknown layout code.
    """
    digest = ctypes.c_uint64()
    status = _load().pith_jpeg_decode(data, len(data), layout, ctypes.byref(digest))
    if status != STATUS_OK:
        raise FfiError("pith_jpeg_decode", status)
    return digest.value


def channels(data: bytes) -> int:
    """Returns the frame's native channel count (1 or 3) — the digest-
    exact layout is :data:`LAYOUT_GRAY8` for 1 and :data:`LAYOUT_RGB8`
    for 3."""
    out = ctypes.c_uint32()
    status = _load().pith_jpeg_channels(data, len(data), ctypes.byref(out))
    if status != STATUS_OK:
        raise FfiError("pith_jpeg_channels", status)
    return out.value


def _replay(op: str, values: list[float], out_lanes: int) -> list[float]:
    """Feeds ``values`` to one kernel op over little-endian ``f64``
    bit-pattern buffers and returns the decoded output lanes."""
    import struct

    packed = b"".join(struct.pack("<d", v) for v in values)
    out = (ctypes.c_ubyte * (out_lanes * 8))()
    status = getattr(_load(), op)(
        packed, len(packed), out, len(out)
    )
    if status != STATUS_OK:
        raise FfiError(op, status)
    raw = bytes(out)
    return list(struct.unpack(f"<{out_lanes}d", raw))


def dequant_zigzag(payload: list[float]) -> list[float]:
    """Replays the ``dequant.zigzag.8x8`` kernel: 64 DQT-payload
    values in scan order in, the same values in natural (row-major)
    order out (``natural[ZIGZAG[k]] = payload[k]``)."""
    return _replay("pith_jpeg_dequant_zigzag", payload, BLOCK)


def dequant(coefs: list[float], qt: list[float]) -> list[float]:
    """Replays the ``dequant.8x8`` kernel: 64 natural-order
    coefficients and 64 quantizer values in, the 64 element-wise
    products out."""
    return _replay("pith_jpeg_dequant", coefs + qt, BLOCK)


def idct_islow(coefs: list[float], qt: list[float]) -> list[float]:
    """Replays the ``idct.islow.8x8`` kernel: the same input layout as
    :func:`dequant`, the 64 ``u8`` samples of the fixed-point islow
    transcription out as ``f64`` lanes."""
    return _replay("pith_jpeg_idct_islow", coefs + qt, BLOCK)


def idct_oracle(block: list[float]) -> list[float]:
    """Replays the ``idct.oracle.8x8`` kernel: 64 dequantized
    coefficients in, the raw orthonormal ``f64`` DCT-III oracle result
    out. Platform-``libm``-shaped: compare within the recorded
    tolerance, never bit-for-bit."""
    return _replay("pith_jpeg_idct_oracle", block, BLOCK)
