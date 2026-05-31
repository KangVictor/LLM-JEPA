# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

LLM-JEPA — applying Joint-Embedding Predictive Architecture (JEPA) concepts to large language models. Currently implements SentenceJEPA-Small: a sentence-level JEPA trained from scratch on Wikipedia.

## Architecture

- **Sentence Encoder**: 4-layer BERT-style transformer (hidden 256, 4 heads, FFN 1024, max 48 tokens, mean pooling → 256-dim embedding)
- **Predictor**: 2-layer transformer over sentence embeddings with learned mask token and position embeddings
- **Training**: End-to-end MSE prediction loss + SIGReg regularization. No EMA, no stop-gradient.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Default config
python train.py --config configs/default.yaml

# With ablation overrides
python train.py --config configs/default.yaml --override sigreg.enabled=false
python train.py --config configs/default.yaml --override masking.multi_mask=false
python train.py --config configs/default.yaml --override predictor.num_layers=1
python train.py --config configs/default.yaml --override masking.mask_ratio_min=0.40 masking.mask_ratio_max=0.40
```

## Project Structure

- `src/model.py` — SentenceEncoder, Predictor, SentenceJEPA
- `src/data.py` — Wikipedia streaming dataset + collation
- `src/masking.py` — JEPA mask sampling
- `src/sigreg.py` — SIGReg regularization loss
- `src/logging_utils.py` — Metrics computation + wandb logging
- `train.py` — Main training script
- `configs/default.yaml` — All hyperparameters and ablation switches
