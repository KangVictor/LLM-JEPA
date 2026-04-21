import math
import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):
    """BERT-style transformer encoder that produces one embedding per sentence."""

    def __init__(self, cfg):
        super().__init__()
        enc = cfg["encoder"]
        self.hidden_size = enc["hidden_size"]

        self.token_emb = nn.Embedding(enc["vocab_size"], enc["hidden_size"])
        self.pos_emb = nn.Embedding(enc["max_seq_len"], enc["hidden_size"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc["hidden_size"],
            nhead=enc["num_heads"],
            dim_feedforward=enc["ffn_size"],
            dropout=enc["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=enc["num_layers"]
        )
        self.norm = nn.LayerNorm(enc["hidden_size"])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, S, T) token ids
            attention_mask: (B, S, T) 1=real token, 0=pad
        Returns:
            embeddings: (B, S, H) mean-pooled sentence embeddings
        """
        B, S, T = input_ids.shape
        H = self.hidden_size

        # Flatten to (B*S, T) for efficient encoding
        ids_flat = input_ids.reshape(B * S, T)
        mask_flat = attention_mask.reshape(B * S, T)

        # Token + positional embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(ids_flat) + self.pos_emb(positions)

        # Transformer expects src_key_padding_mask: True = ignore
        pad_mask = mask_flat == 0
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Mean pool over non-padding tokens
        mask_expanded = mask_flat.unsqueeze(-1).float()  # (B*S, T, 1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B*S, 1)
        pooled = (x * mask_expanded).sum(dim=1) / token_counts  # (B*S, H)

        return pooled.reshape(B, S, H)


class Predictor(nn.Module):
    """Transformer predictor over sentence-level embeddings."""

    def __init__(self, cfg):
        super().__init__()
        pred = cfg["predictor"]
        data = cfg["data"]
        self.hidden_size = pred["hidden_size"]

        self.sentence_pos_emb = nn.Embedding(
            data["max_sentences"], pred["hidden_size"]
        )
        self.mask_token = nn.Parameter(torch.randn(pred["hidden_size"]) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred["hidden_size"],
            nhead=pred["num_heads"],
            dim_feedforward=pred["ffn_size"],
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=pred["num_layers"]
        )
        self.norm = nn.LayerNorm(pred["hidden_size"])
        self.proj = nn.Linear(pred["hidden_size"], pred["hidden_size"])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.sentence_pos_emb.weight, std=0.02)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, encoder_embeddings, sentence_mask, mask_indices):
        """
        Args:
            encoder_embeddings: (B, S, H) — already has mask tokens at masked positions
            sentence_mask: (B, S) — True for real sentences, False for padding
            mask_indices: (B, S) — True for masked sentence positions
        Returns:
            predictions: (N_masked, H) — predicted embeddings at masked positions
        """
        B, S, H = encoder_embeddings.shape

        # Add sentence position embeddings
        positions = torch.arange(S, device=encoder_embeddings.device).unsqueeze(0)
        x = encoder_embeddings + self.sentence_pos_emb(positions)

        # Transformer with padding mask (True = ignore)
        pad_mask = ~sentence_mask
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        x = self.proj(x)

        # Extract predictions at masked positions
        predictions = x[mask_indices]  # (N_masked, H)
        return predictions


class SentenceJEPA(nn.Module):
    """Full SentenceJEPA: encoder + predictor, end-to-end (no EMA, no stop-grad)."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = SentenceEncoder(cfg)
        self.predictor = Predictor(cfg)

    def forward(self, input_ids, attention_mask, sentence_mask, mask_indices):
        """
        Args:
            input_ids: (B, S, T)
            attention_mask: (B, S, T)
            sentence_mask: (B, S) — True for real sentences
            mask_indices: (B, S) — True for masked positions
        Returns:
            pred_out: (N_masked, H) predicted embeddings
            targets: (N_masked, H) original encoder embeddings at masked positions
            enc_out: (B, S, H) all encoder embeddings (for SIGReg)
        """
        # Encode all sentences
        enc_out = self.encoder(input_ids, attention_mask)  # (B, S, H)

        # Build predictor input: replace masked positions with learned mask token
        predictor_input = enc_out.clone()
        predictor_input[mask_indices] = self.predictor.mask_token

        # Predict masked embeddings
        pred_out = self.predictor(predictor_input, sentence_mask, mask_indices)

        # Targets: original encoder embeddings at masked positions (NO detach)
        targets = enc_out[mask_indices]

        return pred_out, targets, enc_out
