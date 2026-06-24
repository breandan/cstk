#!/usr/bin/env python3
"""Quantize reranker safetensors for browser transfer.

The generated WebGPU module currently expects f32 weight buffers. This script
therefore writes a smaller safetensors file for download, and the HTML harness
expands it back to f32 before calling setupNet().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np


DTYPES: dict[str, tuple[str, int]] = {
    "F32": ("<f4", 4),
    "F16": ("<f2", 2),
    "I8": ("i1", 1),
    "U8": ("u1", 1),
    "I32": ("<i4", 4),
}


def read_safetensors(path: Path) -> tuple[dict[str, Any], bytes, int]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path} is too short to be a safetensors file")

    header_len = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_len
    if data_start > len(raw):
        raise ValueError(f"{path} has a header longer than the file")

    header = json.loads(raw[8:data_start].decode("utf-8"))
    return header, raw, data_start


def tensor_entries(header: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(name, entry) for name, entry in header.items() if name != "__metadata__"]


def tensor_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def load_tensor(raw: bytes, data_start: int, name: str, entry: dict[str, Any]) -> np.ndarray:
    dtype_name = entry["dtype"]
    if dtype_name not in DTYPES:
        raise ValueError(f"{name}: unsupported dtype {dtype_name}")

    dtype, width = DTYPES[dtype_name]
    shape = [int(dim) for dim in entry["shape"]]
    begin, end = [int(offset) for offset in entry["data_offsets"]]
    byte_len = end - begin
    expected = tensor_numel(shape) * width
    if byte_len != expected:
        raise ValueError(f"{name}: {byte_len} data bytes, expected {expected}")
    if data_start + end > len(raw):
        raise ValueError(f"{name}: data offset extends past end of file")

    data = np.frombuffer(raw, dtype=np.dtype(dtype), count=tensor_numel(shape), offset=data_start + begin)
    return data.reshape(shape)


def as_metadata_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"non-finite metadata value: {value}")
    return f"{value:.17g}"


def quantize_q8(name: str, tensor: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    values = tensor.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: cannot quantize tensors containing NaN or Inf")

    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    scale = max_abs / 127.0 if max_abs else 1.0
    quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    restored = quantized.astype(np.float32) * scale
    error = restored - values

    return quantized, {
        "scale": as_metadata_float(scale),
        "max_abs": as_metadata_float(max_abs),
        "max_abs_error": as_metadata_float(float(np.max(np.abs(error))) if error.size else 0.0),
        "mean_abs_error": as_metadata_float(float(np.mean(np.abs(error))) if error.size else 0.0),
    }


def quantize_f16(name: str, tensor: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    values = tensor.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: cannot quantize tensors containing NaN or Inf")

    quantized = values.astype(np.float16)
    restored = quantized.astype(np.float32)
    error = restored - values
    return quantized, {
        "max_abs_error": as_metadata_float(float(np.max(np.abs(error))) if error.size else 0.0),
        "mean_abs_error": as_metadata_float(float(np.mean(np.abs(error))) if error.size else 0.0),
    }


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header: dict[str, Any] = {"__metadata__": metadata}
    chunks: list[bytes] = []
    cursor = 0
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": [int(dim) for dim in shape],
            "data_offsets": [cursor, cursor + len(data)],
        }
        chunks.append(data)
        cursor += len(data)

    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    payload = struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def format_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def quantize_file(input_path: Path, output_path: Path, mode: str) -> None:
    header, raw, data_start = read_safetensors(input_path)
    entries = tensor_entries(header)
    if not entries:
        raise ValueError(f"{input_path} contains no tensors")

    output_tensors: list[tuple[str, str, list[int], bytes]] = []
    metadata: dict[str, str] = {
        "cstk.quantization": "q8_symmetric_per_tensor" if mode == "q8" else "f16",
        "cstk.quantized_from": input_path.name,
        "cstk.original_size_bytes": str(input_path.stat().st_size),
        "cstk.quantizer": Path(__file__).name,
    }
    stats: list[tuple[str, int, int, float, float]] = []

    for name, entry in entries:
        if entry["dtype"] != "F32":
            raise ValueError(f"{name}: expected F32 input tensor, found {entry['dtype']}")

        tensor = load_tensor(raw, data_start, name, entry)
        original_bytes = tensor.nbytes
        if mode == "q8":
            packed, qmeta = quantize_q8(name, tensor)
            out_dtype = "I8"
        elif mode == "f16":
            packed, qmeta = quantize_f16(name, tensor)
            out_dtype = "F16"
        else:
            raise ValueError(f"unsupported mode: {mode}")

        output_tensors.append((name, out_dtype, [int(dim) for dim in entry["shape"]], packed.tobytes(order="C")))
        metadata[f"cstk.quant.{name}.source_dtype"] = entry["dtype"]
        for key, value in qmeta.items():
            metadata[f"cstk.quant.{name}.{key}"] = value

        stats.append((
            name,
            original_bytes,
            packed.nbytes,
            float(qmeta["max_abs_error"]),
            float(qmeta["mean_abs_error"]),
        ))

    write_safetensors(output_path, output_tensors, metadata)

    old_size = input_path.stat().st_size
    new_size = output_path.stat().st_size
    print(f"wrote {output_path}")
    print(f"size: {format_bytes(old_size)} -> {format_bytes(new_size)} ({old_size / new_size:.2f}x smaller)")

    worst = sorted(stats, key=lambda row: row[3], reverse=True)[:8]
    print("largest max abs reconstruction errors:")
    for name, original_bytes, packed_bytes, max_error, mean_error in worst:
        print(
            f"  {name}: {format_bytes(original_bytes)} -> {format_bytes(packed_bytes)}, "
            f"max={max_error:.6g}, mean={mean_error:.6g}"
        )


def main() -> None:
    default_input = Path(__file__).with_name("reranker_2000.safetensors")
    parser = argparse.ArgumentParser(description="Quantize reranker safetensors for web transfer.")
    parser.add_argument("input", nargs="?", type=Path, default=default_input)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["q8", "f16"], default="q8")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}.{args.mode}.safetensors")
    else:
        output_path = args.output.expanduser().resolve()

    quantize_file(input_path, output_path, args.mode)


if __name__ == "__main__":
    main()
