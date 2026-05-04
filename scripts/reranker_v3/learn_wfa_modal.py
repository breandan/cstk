import modal
import os, sys, subprocess, pathlib, glob, time

app = modal.App("tinygrad-wfa-training")

data_vol = modal.Volume.from_name("wfa-data", create_if_missing=True)
out_vol  = modal.Volume.from_name("wfa-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ca-certificates", "clang")
    .pip_install("numpy", "safetensors")
    .run_commands("git clone --depth 1 https://github.com/tinygrad/tinygrad.git /opt/tinygrad")
    .env({"PYTHONPATH": "/opt/tinygrad"})
    .add_local_file("learn_wfa.py", "/workspace/learn_wfa.py")
)

@app.function(image=image, volumes={"/data": data_vol})
def upload_data_file(filename: str, contents: bytes):
    allowed = {"so_ts_markov.txt", "so_vs_markov.txt", "so_ts_wfa.txt", "so_vs_wfa.txt"}
    if filename not in allowed:
        raise ValueError(f"filename must be one of {sorted(allowed)}")

    dst = f"/data/{filename}"
    with open(dst, "wb") as f:
        f.write(contents)

    data_vol.commit()
    print(f"✅ Uploaded {filename} ({len(contents)} bytes) to {dst}")

@app.function(
    gpu="A10G",
    timeout=60 * 60 * 24,
    image=image,
    volumes={"/data": data_vol, "/out": out_vol},
)
def train_remote(
        steps: int = 200_000,
        batch_size: int = 32,
        lr: float = 1e-3,
        save_every: int = 10_000,
        train_file: str = "so_ts_wfa.txt",
):
    os.makedirs("/out", exist_ok=True)

    src = f"/data/{train_file}"
    dst = f"/out/{train_file}"

    if not os.path.exists(src):
        print(f"⚠️ Missing {src} in wfa-data volume. Upload it first.")
        return

    if not os.path.exists(dst):
        os.symlink(src, dst)

    print("Artifacts will be written to /out and committed when WFA checkpoints change.")

    last_mtime = 0.0

    def latest_artifact_mtime() -> float:
        paths = []
        paths.extend(glob.glob("/out/wfa_ckpt_*.safetensors"))
        paths.extend(glob.glob("/out/wfa_final.safetensors"))
        if not paths:
            return 0.0
        return max(os.stat(p).st_mtime for p in paths)

    def maybe_commit(force: bool = False):
        nonlocal last_mtime
        mt = latest_artifact_mtime()
        if mt == 0.0:
            return
        if force or mt > last_mtime:
            print("💾 Committing WFA artifacts volume...")
            out_vol.commit()
            last_mtime = mt

    env = os.environ.copy()
    env.pop("CUDA", None)
    env["DEV"] = "CUDA"
    env["DEVICE"] = "CUDA"
    env["PYTHONPATH"] = "/opt/tinygrad"
    env["PYTHONUNBUFFERED"] = "1"

    print("=== nvidia-smi ===")
    subprocess.run(["bash", "-lc", "nvidia-smi"], check=False)

    print("=== tinygrad CUDA smoke test ===")
    subprocess.run(
        [
            sys.executable, "-u", "-c",
            "import os; "
            "from tinygrad.device import Device; "
            "from tinygrad import Tensor; "
            "print('DEV env =', os.environ.get('DEV')); "
            "print('DEVICE env =', os.environ.get('DEVICE')); "
            "print('Device.DEFAULT =', Device.DEFAULT); "
            "x = Tensor.randn(1024, 1024); "
            "y = (x @ x).realize(); "
            "print('ok, realized', y.shape)"
        ],
        env=env,
        check=True,
    )

    cmd = [
        sys.executable, "-u", "/workspace/learn_wfa.py",
        "--data", train_file,
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--save_every", str(save_every),
    ]

    print("🚀 Launching WFA training:")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd="/out",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        maybe_commit()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"WFA training exited with code {rc}")

    maybe_commit(force=True)
    print("✅ WFA training finished and artifacts committed to wfa-artifacts volume.")

@app.local_entrypoint()
def main(
        steps: int = 200_000,
        batch_size: int = 32,
        lr: float = 1e-3,
        save_every: int = 10_000,
        upload_ts: str = "",
        train_file: str = "so_ts_markov.txt",
        run: bool = True,
):
    if upload_ts:
        p = pathlib.Path(upload_ts)
        upload_data_file.remote(train_file, p.read_bytes())

    if run:
        train_remote.remote(
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            save_every=save_every,
            train_file=train_file,
        )