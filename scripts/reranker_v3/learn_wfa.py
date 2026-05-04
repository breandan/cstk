#!/usr/bin/env python3
import time, struct, json, argparse, itertools, random
from typing import List
import numpy as np
from tinygrad import Tensor, TinyJit, nn, dtypes

# -------------------------------------------------------------------- #
#  Configuration
# -------------------------------------------------------------------- #
STATES = 256
RANK = 32

# Real emitted symbols only: ASCII 33–126.
# No PAD symbol in the WFA alphabet.
VOCAB_SIZE = 94
CHAR_OFFSET = 33
CHAR_TO_ID = {chr(i): i - CHAR_OFFSET for i in range(CHAR_OFFSET, 127)}
ID_TO_CHAR = {i - CHAR_OFFSET: chr(i) for i in range(CHAR_OFFSET, 127)}

MAX_LEN = 120
DEVICE = "GPU"

class TinyWFA:
    def __init__(self, vocab_size, num_states, rank):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.rank = rank

        # Low-Rank Factorization:
        # fact_L: [S, V, R]
        # fact_R: [R, S]
        # logits: [S, V, S]
        self.fact_L = Tensor.scaled_uniform(num_states, vocab_size, rank)
        self.fact_R = Tensor.scaled_uniform(rank, num_states)

        self.start_logits = Tensor.scaled_uniform(num_states)

        # Per-source-state stop logits.
        # This replaces global final_logits.
        self.stop_logits = Tensor.scaled_uniform(num_states)

    def get_trans_logits(self):
        return self.fact_L.matmul(self.fact_R)

    def get_normalized_weights(self):
        logits = self.get_trans_logits()
        s, v, s2 = logits.shape

        flat_trans = logits.reshape(s, v * s2)
        stop = self.stop_logits.reshape(s, 1)

        # For each source state q, normalize:
        #   all emitted transitions q -a-> q'
        #   plus STOP(q)
        joint = flat_trans.cat(stop, dim=1).log_softmax(1)

        log_trans = joint[:, :v * s2].reshape(s, v, s2)
        log_stop = joint[:, v * s2]

        log_start = self.start_logits.log_softmax(0)

        return log_trans, log_start, log_stop

    def forward(self, x: Tensor, lens: Tensor):
        B, T = x.shape
        log_trans, log_start, log_stop = self.get_normalized_weights()

        alpha = log_start.reshape(1, self.num_states).expand(B, self.num_states)
        trans_by_token = log_trans.permute(1, 0, 2)  # [V, S, S]

        for t in range(T):
            toks = x[:, t]
            step_trans = trans_by_token[toks]        # [B, S, S]

            new_alpha = (alpha.unsqueeze(2) + step_trans).logsumexp(1)

            # Only consume positions t < length.
            # For padded suffixes, keep alpha unchanged.
            active = (lens > t).cast(dtypes.float32).reshape(B, 1)
            alpha = new_alpha * active + alpha * (1.0 - active)

        # Stop exactly after the true sequence length.
        return (alpha + log_stop.reshape(1, self.num_states)).logsumexp(1)

def save_wfa_safetensors(model: TinyWFA, filepath: str):
    print(f"Serialize WFA to {filepath}...")
    log_trans, log_start, log_stop = model.get_normalized_weights()

    transitions = log_trans.numpy()
    start_w = log_start.numpy()
    stop_w = log_stop.numpy()

    sources, targets, labels, scores = [], [], [], []

    vocab_list = [ID_TO_CHAR[i] for i in range(VOCAB_SIZE)]

    PRUNE_THRESHOLD = -10.0

    it = np.nditer(transitions, flags=['multi_index'])
    while not it.finished:
        val = float(it[0])
        if val > PRUNE_THRESHOLD:
            src, v_idx, tgt = it.multi_index
            sources.append(int(src))
            targets.append(int(tgt))
            labels.append(int(v_idx))     # zero-based token id
            scores.append(val)
        it.iternext()

    num_edges = len(sources)
    metadata = {
        "vocab": vocab_list,
        "start_states": list(range(model.num_states)),
        "start_weights": start_w.tolist(),

        # Keep the old metadata names if downstream code expects
        # final_states/final_weights, but semantically these are STOP weights.
        "final_states": list(range(model.num_states)),
        "final_weights": stop_w.tolist(),

        "char_offset": CHAR_OFFSET,
        "token_indexing": "zero_based_ascii_33_126",
        "normalization": "per_state_emit_or_stop",
    }

    header_dict = {
        "__metadata__": metadata,
        "sources": {"dtype": "I32", "shape": [num_edges], "data_offsets": [0,0]},
        "targets": {"dtype": "I32", "shape": [num_edges], "data_offsets": [0,0]},
        "labels":  {"dtype": "I32", "shape": [num_edges], "data_offsets": [0,0]},
        "scores":  {"dtype": "F32", "shape": [num_edges], "data_offsets": [0,0]},
    }

    int_sz, flt_sz = 4, 4
    base = 0
    header_dict["sources"]["data_offsets"] = [base, base + num_edges*int_sz]; base += num_edges*int_sz
    header_dict["targets"]["data_offsets"] = [base, base + num_edges*int_sz]; base += num_edges*int_sz
    header_dict["labels"]["data_offsets"]  = [base, base + num_edges*int_sz]; base += num_edges*int_sz
    header_dict["scores"]["data_offsets"]  = [base, base + num_edges*flt_sz]; base += num_edges*flt_sz

    header_json = json.dumps(header_dict, separators=(',', ':')).encode('utf-8')

    with open(filepath, "wb") as f:
        f.write(struct.pack('<Q', len(header_json)))
        f.write(header_json)
        for x in sources: f.write(struct.pack('<i', x))
        for x in targets: f.write(struct.pack('<i', x))
        for x in labels:  f.write(struct.pack('<i', x))
        for x in scores:  f.write(struct.pack('<f', x))

        total_size = f.tell()

    print(f"Saved {num_edges} edges. Total size: {total_size / 1024:.2f} KB")

# -------------------------------------------------------------------- #
#  Global Shuffle Data Pipeline
# -------------------------------------------------------------------- #
def encode_text(txt: str):
    ids = [CHAR_TO_ID.get(c, 0) for c in txt[:MAX_LEN]]
    return ids, len(ids)

def get_batch_from_list(dataset, idx, batch_size):
    batch_lines = dataset[idx : idx + batch_size]

    # Drop last partial batch for JIT safety
    if len(batch_lines) != batch_size:
        return None

    # Pad value is arbitrary now; positions >= length are ignored.
    tensor_batch = np.zeros((batch_size, MAX_LEN), dtype=np.int32)
    lens = np.zeros((batch_size,), dtype=np.int32)

    for i, txt in enumerate(batch_lines):
        ids, L = encode_text(txt)
        tensor_batch[i, :L] = ids
        lens[i] = L

    return Tensor(tensor_batch), Tensor(lens)

def load_and_shuffle_dataset(path: str):
    print(f"Loading dataset from {path}...", flush=True)
    dataset = []
    t0 = time.time()

    with open(path, 'r', encoding='utf-8') as f:
        for key, group in itertools.groupby(f, key=lambda x: not x.strip()):
            if not key:
                lines = [l.strip() for l in group if l.strip()]
                if len(lines) > 1:
                    dataset.extend(lines[1:])

    print(f"Loaded {len(dataset)} lines in {time.time()-t0:.2f}s. Shuffling...", flush=True)
    t0 = time.time()
    random.shuffle(dataset)
    print(f"Shuffle done in {time.time()-t0:.2f}s.", flush=True)
    return dataset

# -------------------------------------------------------------------- #
#  Training
# -------------------------------------------------------------------- #
@TinyJit
def train_step(x, lens, model, optim):
    log_probs = model.forward(x, lens)
    nll = -log_probs.mean()

    reg = (model.fact_L.abs().mean() + model.fact_R.abs().mean() + model.stop_logits.abs().mean()) * 0.0001

    loss = nll + reg

    optim.zero_grad()
    loss.backward()
    optim.step()

    return nll, loss

def train(args):
    Tensor.training = True
    model = TinyWFA(VOCAB_SIZE, STATES, RANK)

    optim = nn.optim.Adam(
        [model.fact_L, model.fact_R, model.start_logits, model.stop_logits],
        lr=args.lr,
    )

    print(f"Training Factorized WFA (Rank {RANK}) on {DEVICE}", flush=True)
    print(f"Optimizer: Adam | LR: {args.lr}", flush=True)

    full_dataset = load_and_shuffle_dataset(args.data)
    total_samples = len(full_dataset)

    step = 0
    epoch = 0

    train_t0 = time.perf_counter()
    last_log_t = train_t0

    while step < args.steps:
        epoch += 1

        print(f"Epoch {epoch}: reshuffling...", flush=True)
        t0 = time.perf_counter()
        random.shuffle(full_dataset)
        print(f"Epoch {epoch}: reshuffle done in {time.perf_counter() - t0:.3f}s", flush=True)

        for i in range(0, total_samples, args.batch_size):
            step += 1
            if step > args.steps:
                break

            step_t0 = time.perf_counter()

            # -------------------------------
            # Batch construction timing
            # -------------------------------
            t_batch0 = time.perf_counter()
            batch = get_batch_from_list(full_dataset, i, args.batch_size)
            t_batch = time.perf_counter() - t_batch0

            if batch is None:
                continue

            # Patched variable-length version:
            batch_x, batch_lens = batch

            # -------------------------------
            # Training step timing
            # -------------------------------
            print(
                f"Step {step:7d} | entering train_step "
                f"| batch {t_batch:.4f}s",
                flush=True,
            ) if step <= 5 else None

            t_step0 = time.perf_counter()
            nll, loss = train_step(batch_x, batch_lens, model, optim)
            t_train = time.perf_counter() - t_step0

            # -------------------------------
            # Scalar realization timing
            # -------------------------------
            t_real0 = time.perf_counter()
            nll_v = float(nll.numpy())
            loss_v = float(loss.numpy())
            mean_len = float(batch_lens.numpy().mean())
            t_real = time.perf_counter() - t_real0

            step_dt = time.perf_counter() - step_t0

            # Print every step initially, then every 50.
            if step <= 20 or step % 50 == 0:
                now = time.perf_counter()
                since_last = now - last_log_t
                elapsed = now - train_t0
                last_log_t = now

                print(
                    f"Step {step:7d} | Epoch {epoch} | "
                    f"NLL {nll_v:.4f} | "
                    f"NLL/tok {nll_v / max(mean_len, 1.0):.4f} | "
                    f"mean_len {mean_len:.1f} | "
                    f"batch {t_batch:.4f}s | "
                    f"train {t_train:.4f}s | "
                    f"realize {t_real:.4f}s | "
                    f"step {step_dt:.4f}s | "
                    f"since_log {since_last:.2f}s | "
                    f"elapsed {elapsed:.2f}s",
                    flush=True,
                )

            if step % args.save_every == 0:
                t_save0 = time.perf_counter()
                save_wfa_safetensors(model, f"wfa_ckpt_{step}.safetensors")
                print(
                    f"Step {step:7d} | checkpoint save took "
                    f"{time.perf_counter() - t_save0:.3f}s",
                    flush=True,
                )

    t_save0 = time.perf_counter()
    save_wfa_safetensors(model, "wfa_final.safetensors")
    print(f"Final checkpoint save took {time.perf_counter() - t_save0:.3f}s", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="so_ts_wfa.txt")
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--save_every", type=int, default=1_000)
    args = p.parse_args()
    train(args)