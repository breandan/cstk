#!/usr/bin/env python3
import time, struct, json, argparse, itertools, random
from typing import List
import numpy as np
from tinygrad import Tensor, TinyJit, nn, dtypes

# -------------------------------------------------------------------- #
#  Configuration
# -------------------------------------------------------------------- #
STATES = 512
RANK = 64
VOCAB_SIZE = 97       # 0=Pad, 1..96=ASCII
CHAR_OFFSET = 32
MAX_LEN = 120
DEVICE = "GPU"

class TinyWFA:
    def __init__(self, vocab_size, num_states, rank):
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.rank = rank

        # Low-Rank Factorization: L (S, V, R) @ R (R, S) -> (S, V, S)
        self.fact_L = Tensor.scaled_uniform(num_states, vocab_size, rank)
        self.fact_R = Tensor.scaled_uniform(rank, num_states)

        self.start_logits = Tensor.scaled_uniform(num_states)
        self.final_logits = Tensor.scaled_uniform(num_states)

    def get_trans_logits(self):
        return self.fact_L.matmul(self.fact_R)

    def get_normalized_weights(self):
        # 1. Reconstruct Logits
        logits = self.get_trans_logits()

        # 2. Local Normalization (Softmax over target states & vocab)
        s, v, s2 = logits.shape
        log_trans = logits.reshape(s, v * s2).log_softmax(1).reshape(s, v, s2)

        # 3. Normalize Start/Final
        log_start = self.start_logits.log_softmax(0)
        log_final = self.final_logits.log_softmax(0)

        return log_trans, log_start, log_final

    def forward(self, x: Tensor):
        B, T = x.shape
        log_trans, log_start, log_final = self.get_normalized_weights()

        alpha = log_start.reshape(1, self.num_states).expand(B, self.num_states)
        trans_by_token = log_trans.permute(1, 0, 2)

        for t in range(T):
            toks = x[:, t]
            step_trans = trans_by_token[toks]
            alpha = (alpha.unsqueeze(2) + step_trans).logsumexp(1)

        return (alpha + log_final.unsqueeze(0)).logsumexp(1)

def save_wfa_safetensors(model: TinyWFA, filepath: str):
    print(f"Serialize WFA to {filepath}...")
    log_trans, log_start, log_final = model.get_normalized_weights()

    transitions = log_trans.numpy()
    start_w = log_start.numpy()
    final_w = log_final.numpy()

    sources, targets, labels, scores = [], [], [], []
    vocab_list = [chr(i + CHAR_OFFSET) for i in range(VOCAB_SIZE - 1)]

    PRUNE_THRESHOLD = -10.0

    it = np.nditer(transitions, flags=['multi_index'])
    while not it.finished:
        val = it[0]
        if val > PRUNE_THRESHOLD:
            src, v_idx, tgt = it.multi_index
            if v_idx > 0:
                sources.append(int(src))
                targets.append(int(tgt))
                labels.append(int(v_idx))
                scores.append(float(val))
        it.iternext()

    num_edges = len(sources)
    metadata = {
        "vocab": vocab_list,
        "start_states": list(range(model.num_states)), "start_weights": start_w.tolist(),
        "final_states": list(range(model.num_states)), "final_weights": final_w.tolist()
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
def encode_text(txt: str) -> List[int]:
    return [max(1, min(VOCAB_SIZE-1, ord(c) - CHAR_OFFSET + 1)) for c in txt[:MAX_LEN]]

def load_and_shuffle_dataset(path: str):
    print(f"Loading dataset from {path}...")
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        # Group by chunks (separated by blank lines)
        for key, group in itertools.groupby(f, key=lambda x: not x.strip()):
            if not key:
                lines = [l.strip() for l in group if l.strip()]
                # Rule (2): Discard first line, keep rest
                if len(lines) > 1:
                    dataset.extend(lines[1:])

    print(f"Loaded {len(dataset)} lines. Shuffling...")
    random.shuffle(dataset)
    return dataset

def get_batch_from_list(dataset, idx, batch_size):
    # Slice batch from global list
    batch_lines = dataset[idx : idx + batch_size]

    # Drop last partial batch for JIT safety
    if len(batch_lines) != batch_size:
        return None

    tensor_batch = np.zeros((batch_size, MAX_LEN), dtype=np.int32)
    for i, txt in enumerate(batch_lines):
        ids = encode_text(txt)
        tensor_batch[i, :len(ids)] = ids
    return Tensor(tensor_batch)

# -------------------------------------------------------------------- #
#  Training
# -------------------------------------------------------------------- #
@TinyJit
def train_step(x, model, optim):
    log_probs = model.forward(x)
    nll = -log_probs.mean()

    # Regularization: L1 on factors + L1 on Final Weights (to keep them small)
    reg = (model.fact_L.abs().mean() +
           model.fact_R.abs().mean() +
           model.final_logits.abs().mean()) * 0.0001

    loss = nll + reg

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def train(args):
    Tensor.training = True
    model = TinyWFA(VOCAB_SIZE, STATES, RANK)

    # Using SGD with Momentum
    optim = nn.optim.Adam([model.fact_L, model.fact_R, model.start_logits, model.final_logits], lr=args.lr)

    print(f"Training Factorized WFA (Rank {RANK}) on {DEVICE}")
    print(f"Optimizer: SGD(momentum=0.9, nesterov=True) | LR: {args.lr}")

    # 1. Load Everything
    full_dataset = load_and_shuffle_dataset(args.data)
    total_samples = len(full_dataset)

    step = 0
    epoch = 0

    while step < args.steps:
        epoch += 1
        # Re-shuffle every epoch for better generalization
        random.shuffle(full_dataset)

        # Iterate over full dataset
        for i in range(0, total_samples, args.batch_size):
            step += 1
            if step > args.steps: break

            batch_x = get_batch_from_list(full_dataset, i, args.batch_size)
            if batch_x is None: continue # Skip partial batch

            loss = train_step(batch_x, model, optim)

            if step % 50 == 0:
                print(f"Step {step:5d} | Epoch {epoch} | NLL: {loss.numpy():.4f}")

            if step % args.save_every == 0:
                save_wfa_safetensors(model, f"wfa_ckpt_{step}.safetensors")

    save_wfa_safetensors(model, "wfa_final.safetensors")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="so_ts_markov.txt")
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=10_000)
    args = p.parse_args()
    train(args)