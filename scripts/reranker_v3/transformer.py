import numpy as np
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_state_dict, safe_save, load_state_dict
from extra.models.transformer import Transformer
from extra.export_model import export_model

def make_random_state_dict_like(sd: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, t in sd.items():
        shape = tuple(t.shape)
        dt = t.dtype

        if dt in (dtypes.float16, dtypes.float32):
            arr = (rng.standard_normal(shape).astype(np.float32) * 0.02)
            if dt == dtypes.float16:
                arr = arr.astype(np.float16)
            out[name] = Tensor(arr, dtype=dt)
        elif dt in (dtypes.int32, dtypes.int64):
            arr = rng.integers(low=-3, high=4, size=shape, dtype=np.int32)
            out[name] = Tensor(arr, dtype=dt)
        else:
            raise RuntimeError(f"Unhandled dtype for {name}: {dt}")
    return out

if __name__ == "__main__":
    maxlen = 120
    model = Transformer(syms=10, maxlen=maxlen, layers=4, embed_dim=128, num_heads=4, ff_dim=32)

    sd = get_state_dict(model)
    sd_rand = make_random_state_dict_like(sd, seed=12345)
    load_state_dict(model, sd_rand)

    example_input = Tensor(np.random.randint(0, 10, (1, maxlen)), dtype=dtypes.int32)

    outdir = Path(__file__).resolve().parent
    weights_path = outdir / "transformer.safetensors"
    js_path = outdir / "transformer.js"

    safe_save(sd_rand, weights_path.as_posix())

    prg, inp_sizes, out_sizes, state = export_model(model, "webgpu", example_input)

    js_path.write_text(prg, encoding="utf-8")

    print("Wrote:")
    print(" -", js_path)
    print(" -", weights_path)
    print("inp_sizes =", inp_sizes, "out_sizes =", out_sizes)