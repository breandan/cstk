import modal
import os
import sys  # <-- Import the 'sys' module
import pathlib

# Define the Modal application and name it.
app = modal.App("reranker-training-app")

# Define the image for the remote environment.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
    .add_local_file("train_unsupervised.py", "/workspace/train_unsupervised.py")
    .add_local_file("train_reranker.py", "/workspace/train_reranker.py")
)

# Define two persistent storage volumes.
data_vol = modal.Volume.from_name("ranker-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("ranker-ckpts", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": data_vol}
)
def upload_encoder(encoder_data: bytes):
    """
    A remote function that takes the raw bytes of the encoder file
    and writes them to the data volume.
    """
    remote_path = "/data/unsupervised_encoder.pt"
    print(f"Writing {len(encoder_data)} bytes to volume at '{remote_path}'...")
    with open(remote_path, "wb") as f:
        f.write(encoder_data)
    data_vol.commit()
    print("✅ Upload complete.")


@app.function(
    gpu="H100",
    image=image,
    timeout=60 * 60 * 24,
    volumes={"/data": data_vol, "/ckpts": ckpt_vol},
)
def train_remote(steps: int = 20_000):
    """
    The main remote function that runs the reranker training on a GPU.
    """
    os.chdir("/workspace")

    # CORRECTED SECTION: Add the current directory to Python's path
    # This ensures that 'import train_reranker' works correctly.
    sys.path.insert(0, ".")

    # Symlink the dataset files from the volume to the workspace.
    for fn in ("so_vs_markov.txt", "so_ts_markov.txt"):
        src = f"/data/{fn}"
        dst = f"/workspace/{fn}"
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Creating symlink for {fn}...")
            os.symlink(src, dst)
        elif not os.path.exists(src):
            print(f"⚠️ Warning: Data file {src} not found in volume 'ranker-data'.")

    # Now this import will succeed.
    from train_reranker import reranker_modal_entrypoint

    reranker_modal_entrypoint(steps, ckpt_vol)


@app.local_entrypoint()
def main(steps: int = 20_000, upload_path: str = None):
    """
    The local entrypoint for controlling the Modal app from your command line.
    """
    if upload_path:
        local_encoder_path = pathlib.Path(upload_path)
        if not local_encoder_path.exists():
            raise FileNotFoundError(f"Local encoder file not found at: {local_encoder_path}")

        print(f"Reading local file '{local_encoder_path}'...")
        with open(local_encoder_path, "rb") as f:
            encoder_bytes = f.read()

        print(f"Uploading {local_encoder_path.name} to Modal...")
        upload_encoder.remote(encoder_bytes)

    else:
        print("🚀 Starting remote training job...")
        train_remote.remote(steps)