import modal
import os, sys, subprocess, pathlib

app = modal.App("tinygrad-reranker-training")

data_vol = modal.Volume.from_name("ranker-data", create_if_missing=True)
out_vol  = modal.Volume.from_name("ranker-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ca-certificates", "clang", "libvulkan1")
    .pip_install("numpy", "safetensors", "dawn-python")
    .run_commands("git clone --depth 1 https://github.com/tinygrad/tinygrad.git /opt/tinygrad")
    .env({"PYTHONPATH": "/opt/tinygrad"})
    .add_local_file("train_reranker.py", "/workspace/train_reranker.py")
)

@app.function(image=image, volumes={"/data": data_vol})
def upload_data_file(filename: str, contents: bytes):
    if filename not in ("so_ts_markov.txt", "so_vs_markov.txt"):
        raise ValueError("filename must be so_ts_markov.txt or so_vs_markov.txt")
    dst = f"/data/{filename}"
    with open(dst, "wb") as f:
        f.write(contents)
    data_vol.commit()
    print(f"✅ Uploaded {filename} ({len(contents)} bytes) to {dst}")

@app.function(
    gpu="H100",
    timeout=60 * 60 * 24,
    image=image,
    volumes={"/data": data_vol, "/out": out_vol},
)
def train_remote(
        steps: int = 20_000,
        batch_size: int = 8,
        neg_samp: int = 199,
        export_every: int = 100,
        export_docs: int = 64,
        lr: float = 1e-4,
        wd: float = 1e-2,
        grad_clip: float = 5.0,
        seed: int = 0,
):
    os.makedirs("/out", exist_ok=True)

    # Symlink datasets into /out (so train script finds them in CWD)
    for fn in ("so_ts_markov.txt", "so_vs_markov.txt"):
        src = f"/data/{fn}"
        dst = f"/out/{fn}"
        if not os.path.exists(src):
            print(f"⚠️ Missing {src} in ranker-data volume. Upload it first.")
            return
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print("Artifacts will be written to /out and committed when reranker.safetensors changes.")

    safepath = "/out/reranker.safetensors"
    last_mtime = 0.0

    def maybe_commit(force: bool = False):
        nonlocal last_mtime
        try:
            st = os.stat(safepath)
        except FileNotFoundError:
            return
        if force or st.st_mtime > last_mtime:
            print("💾 Committing artifacts volume...")
            out_vol.commit()
            last_mtime = st.st_mtime

    env = os.environ.copy()
    env["CUDA"] = "1"
    env["DEVICE"] = "CUDA"
    env["PYTHONPATH"] = "/opt/tinygrad"
    env["PYTHONUNBUFFERED"] = "1"
    env["WEBGPU_PATH"] = "/usr/local/lib/libwebgpu_dawn.so"
    env["XDG_RUNTIME_DIR"] = "/tmp"

    # --- find Dawn shared lib shipped by dawn-python (pydawn) ---
    import platform, pathlib, glob, pydawn

    root = pathlib.Path(pydawn.__file__).resolve().parent / "lib"
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        libs = sorted(glob.glob(str(root / "libwebgpu_dawn*.so")))
    elif system == "darwin":
        libs = sorted(glob.glob(str(root / "libwebgpu_dawn*.dylib")))
    else:
        raise RuntimeError(f"unsupported OS: {system}")

    print("platform.system  =", platform.system())
    print("platform.machine =", platform.machine())
    print("candidate libs   =", [pathlib.Path(x).name for x in libs])
    assert libs, f"no Dawn shared lib for {system} under {root}"

    want = "x86_64" if machine in ("x86_64","amd64") else ("aarch64" if machine in ("aarch64","arm64") else machine)
    pick = next((p for p in libs if want in pathlib.Path(p).name), libs[0])

    env["WEBGPU_PATH"] = pick
    print("WEBGPU_PATH =", pick)

    print("=== webgpu smoke test ===")
    smoke = "\n".join([
        "import os, ctypes, platform",
        "pick = os.environ['WEBGPU_PATH']",
        "print('platform.system  =', platform.system())",
        "print('platform.machine =', platform.machine())",
        "print('WEBGPU_PATH      =', pick)",
        # don't accidentally try to load a dylib on linux
        "if platform.system().lower() == 'linux':",
        "  assert pick.endswith('.so'), f'expected .so on Linux, got {pick}'",
        "os.environ.setdefault('WEBGPU_BACKEND', 'Null')",
        "ctypes.CDLL(pick)",
        "print('CDLL load: OK')",
        "from tinygrad.device import Device",
        "from tinygrad import Tensor",
        "Device.DEFAULT = 'WEBGPU'",
        "x = (Tensor.randn(4,4) @ Tensor.randn(4,4)).realize()",
        "print('tinygrad WEBGPU realize: OK', x.shape)",
    ])
    subprocess.run([sys.executable, "-u", "-c", smoke], env=env, check=True)

    print("=== nvidia-smi ===")
    subprocess.run(["bash", "-lc", "nvidia-smi"], check=False)

    print("=== tinygrad smoke test ===")
    subprocess.run(
        [sys.executable, "-u", "-c",
         "import os; "
         "from tinygrad.device import Device; "
         "from tinygrad import Tensor; "
         "print('DEVICE env =', os.environ.get('DEVICE')); "
         "print('Device.DEFAULT (before) =', Device.DEFAULT); "
         "Device.DEFAULT = os.environ.get('DEVICE','CUDA'); "
         "print('Device.DEFAULT (after) =', Device.DEFAULT); "
         "x=Tensor.randn(1024,1024); y=(x@x).realize(); "
         "print('ok, realized')"],
        env=env,
        check=True,
    )

    cmd = [
        sys.executable, "-u", "/workspace/train_reranker.py",
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--neg-samp", str(neg_samp),
        "--export-every", str(export_every),
        "--export-docs", str(export_docs),
        "--lr", str(lr),
        "--wd", str(wd),
        "--grad-clip", str(grad_clip),
        "--seed", str(seed),
        "--train-file", "so_ts_markov.txt",
        "--val-file", "so_vs_markov.txt",
    ]

    print("🚀 Launching training:")
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
        raise RuntimeError(f"Training exited with code {rc}")

    maybe_commit(force=True)
    print("✅ Training finished and artifacts committed to ranker-artifacts volume.")

@app.local_entrypoint()
def main(
    steps: int = 20_000,
    upload_ts: str = "",
    upload_vs: str = "",
    run: bool = True,
):
    if upload_ts:
        p = pathlib.Path(upload_ts)
        upload_data_file.remote("so_ts_markov.txt", p.read_bytes())
    if upload_vs:
        p = pathlib.Path(upload_vs)
        upload_data_file.remote("so_vs_markov.txt", p.read_bytes())

    if run:
        train_remote.remote(steps=steps)
