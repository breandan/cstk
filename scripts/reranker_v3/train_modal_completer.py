#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_modal_completer.py
------------------------
Modal harness for train_completer.py.

Typical use:

  # 1) Upload a local data file into the Modal volume.
  modal run train_modal_completer.py --upload-path so_ts_markov.txt

  # 2) Train on H100.
  modal run train_modal_completer.py --steps 20000 --train-file so_ts_markov.txt

  # 3) Download the checkpoint.
  modal run train_modal_completer.py --download completer.pt --out completer.pt
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Optional

import modal


app = modal.App("completer-training-app")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
    .add_local_file("train_completer.py", "/workspace/train_completer.py")
)

data_vol = modal.Volume.from_name("completer-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("completer-ckpts", create_if_missing=True)


@app.function(image=image, volumes={"/data": data_vol}, timeout=60 * 20)
def upload_data(remote_name: str, payload: bytes) -> None:
    remote_path = f"/data/{remote_name}"
    print(f"writing {len(payload)} bytes -> {remote_path}")
    with open(remote_path, "wb") as f:
        f.write(payload)
    data_vol.commit()
    print("upload committed")


@app.function(image=image, volumes={"/ckpts": ckpt_vol}, timeout=60 * 20)
def read_checkpoint(remote_name: str) -> bytes:
    remote_path = f"/ckpts/{remote_name}"
    print(f"reading {remote_path}")
    with open(remote_path, "rb") as f:
        return f.read()


def _script_supports_flag(flag: str) -> bool:
    """Return true iff the mounted train_completer.py advertises this CLI flag."""
    try:
        res = subprocess.run(
            [sys.executable, "/workspace/train_completer.py", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return flag in res.stdout
    except Exception as e:
        print(f"warning: could not inspect train_completer.py --help: {e}", flush=True)
        return False


def _append_if_supported(cmd: list[str], flag: str, value: object | None = None) -> None:
    """Append an optional flag only when the mounted script supports it.

    This keeps the Modal harness from drifting ahead of train_completer.py. It also
    makes it obvious in logs when your local mounted train_completer.py is stale.
    """
    if _script_supports_flag(flag):
        cmd.append(flag)
        if value is not None:
            cmd.append(str(value))
    else:
        print(f"skipping unsupported train_completer.py flag: {flag}", flush=True)


@app.function(
    gpu="H100",
    image=image,
    timeout=60 * 60 * 24,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
)
def train_remote(
        steps: int = 20_000,
        train_file: str = "so_ts_markov.txt",
        val_file: Optional[str] = None,
        tokenizer: str = "char",
        ckpt_name: str = "completer.pt",
        batch_size: int = 128,
        block_size: int = 128,
        dim: int = 192,
        n_heads: int = 4,
        n_layers: int = 3,
        log_every: int = 25,
        data_format: str = "instances",
        max_candidates_per_instance: int = 0,
        max_vocab_scan_samples: int = 250_000,
) -> None:
    os.chdir("/workspace")

    train_path = f"/data/{train_file}"
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found. Upload it first with: "
            f"modal run train_modal_completer.py --upload-path {train_file}"
        )

    cmd = [
        sys.executable,
        "/workspace/train_completer.py",
        "--mode", "train",
        "--train", train_path,
        "--steps", str(steps),
        "--tokenizer", tokenizer,
        "--ckpt-out", f"/ckpts/{ckpt_name}",
        "--batch-size", str(batch_size),
        "--block-size", str(block_size),
        "--dim", str(dim),
        "--n-heads", str(n_heads),
        "--n-layers", str(n_layers),
        "--log-every", str(log_every),
    ]

    # These were added after the first draft, so guard them against stale local
    # train_completer.py versions mounted by Modal.
    _append_if_supported(cmd, "--data-format", data_format)
    _append_if_supported(cmd, "--max-candidates-per-instance", max_candidates_per_instance)
    _append_if_supported(cmd, "--max-vocab-scan-samples", max_vocab_scan_samples)
    _append_if_supported(cmd, "--debug-first-steps", 3)

    if val_file:
        val_path = f"/data/{val_file}"
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"{val_path} not found")
        cmd.extend(["--val", val_path])

    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    ckpt_vol.commit()
    print(f"checkpoint committed -> /ckpts/{ckpt_name}")


@app.local_entrypoint()
def main(
        steps: int = 20_000,
        train_file: str = "so_ts_markov.txt",
        val_file: Optional[str] = None,
        tokenizer: str = "char",
        ckpt_name: str = "completer.pt",
        upload_path: Optional[str] = None,
        download: Optional[str] = None,
        out: Optional[str] = None,
        batch_size: int = 128,
        block_size: int = 128,
        dim: int = 192,
        n_heads: int = 4,
        n_layers: int = 3,
        log_every: int = 25,
        data_format: str = "instances",
        max_candidates_per_instance: int = 0,
        max_vocab_scan_samples: int = 250_000,
) -> None:
    if upload_path:
        p = pathlib.Path(upload_path)
        if not p.exists():
            raise FileNotFoundError(p)
        upload_data.remote(p.name, p.read_bytes())
        return

    if download:
        payload = read_checkpoint.remote(download)
        out_path = pathlib.Path(out or download)
        out_path.write_bytes(payload)
        print(f"wrote {len(payload)} bytes -> {out_path}")
        return

    train_remote.remote(
        steps=steps,
        train_file=train_file,
        val_file=val_file,
        tokenizer=tokenizer,
        ckpt_name=ckpt_name,
        batch_size=batch_size,
        block_size=block_size,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        log_every=log_every,
        data_format=data_format,
        max_candidates_per_instance=max_candidates_per_instance,
        max_vocab_scan_samples=max_vocab_scan_samples,
    )
