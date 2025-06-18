import modal, subprocess, os, sys, pathlib

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
    .pip_install("numpy")
    .add_local_dir(".", "/workspace", ignore=["*.txt"])
)

data_vol = modal.Volume.from_name("ranker-data")
ckpt_vol = modal.Volume.from_name("ranker-ckpts")

app = modal.App("interaction-ranker")

@app.function(
    gpu="H100",
    image=image,
    timeout=60 * 60 * 3,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
)
def train_remote(steps: int = 20_000):
    os.chdir("/workspace")
    sys.path.insert(0, ".")

    from reranker import modal_entrypt

    for fn in ("so_vs_markov.txt", "so_ts_markov.txt"):
        src = f"/data/{fn}"
        dst = f"/workspace/{fn}"
        if not os.path.exists(dst):
            os.symlink(src, dst)

    modal_entrypt(steps, ckpt_vol)

@app.local_entrypoint()
def main(steps: int = 20_000):
    train_remote.remote(steps)