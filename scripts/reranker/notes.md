## Dataset

Dataset consists of Syntax and StackOverflow errors and fixes, where each lexical token in Python is mapped to a single character.

First document is the "query" or broken snippet, the second is the true repair, the rest are the most probable candidate repairs drawn by the Markov model.

Each ranking instance consists of up to 1k documents in character format. Each character is ASCII, so 0-128. Python has ~100 unique lexical tokens.

## Objective

Objective is to predict the true repair by scrambling all repairs, scoring each candidate repair, and measuring how close the true repair was to rank 0 in the list.

We will use MRR or Precision@{1,10,100} to evaluate this across a held-out test set.

## Training notes

First tried a learning-to-rank pipeline where the model learns to output scalar value for each document directly. This does not seem to work well.

Two stage pipeline where we train an unsupervised transformer-encoder on next-token prediction, then use the query + document embeddings to produce a score. This can be used in frozen mode, or fine-tuned with ranking. MRR seems better

## Model architecture

Takes as input a concatenated (query, document) pair and outputs a score for each pair. This is used to rank all candidates.

Encoder outputs an embedding, which is fed to a linear model and used to output pairwise score.

## Hyperparameter sweep

Dimension 512, 8 heads, 4 layers seems to be the sweet spot for transformer encoder.

Learning rates: 2e-3, 1e-4

TAU controls the temperature of cross-entropy score.

## Training

```
python train_unsupervised.py
```

Saves `unsupervised_encoder.pt` to same directory.

```
modal run train_modal_reranker.py --upload-path ./unsupervised_encoder.pt

modal run train_modal_reranker.py --steps 20000 2>&1 | tee modal_train.txt

python plot_loss.py plot_loss.txt
```

## Local serving

Start this service before evaluation.

```
modal volume get ranker-ckpts encoder_finetuned_step_500.pt
modal volume get ranker-ckpts reranker_finetuned_step_500.pt
# Update model in serve_local_reranker.py
python serve_local_reranker.py
```

## Evaluate

```
./gradlew pythonBarHillel
```