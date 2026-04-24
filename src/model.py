import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """LeWM-style projection head: Linear -> BatchNorm -> GELU -> Linear."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, sample_mask=None):
        """
        Args:
            x: (..., D) input features
            sample_mask: optional bool mask over leading dims. When provided,
                only valid samples contribute to BatchNorm statistics.
        """
        leading_shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))

        if sample_mask is None:
            projected = self.net(flat)
            return projected.reshape(*leading_shape, self.output_dim)

        valid = sample_mask.reshape(-1)
        projected = flat.new_zeros(flat.size(0), self.output_dim)
        if valid.any():
            projected[valid] = self.net(flat[valid])
        return projected.reshape(*leading_shape, self.output_dim)


class SentenceEncoder(nn.Module):
    """BERT-style transformer encoder that produces one embedding per sentence."""

    def __init__(self, cfg):
        super().__init__()
        enc = cfg["encoder"]
        self.hidden_size = enc["hidden_size"]
        self.embedding_size = enc.get("embedding_size", enc["hidden_size"])

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
        self.projector = ProjectionHead(
            input_dim=enc["hidden_size"],
            hidden_dim=enc.get("projector_hidden_size", 2048),
            output_dim=self.embedding_size,
        )

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
            embeddings: (B, S, D) projected sentence embeddings
        """
        B, S, T = input_ids.shape

        # Flatten to (B*S, T) for efficient encoding
        ids_flat = input_ids.reshape(B * S, T)
        mask_flat = attention_mask.reshape(B * S, T)
        sentence_flat = mask_flat.any(dim=1)

        # Token + positional embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(ids_flat) + self.pos_emb(positions)

        # Transformer expects src_key_padding_mask: True = ignore
        pad_mask = mask_flat == 0
        pad_mask[~sentence_flat] = False
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Mean pool over non-padding tokens
        mask_expanded = mask_flat.unsqueeze(-1).float()  # (B*S, T, 1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B*S, 1)
        pooled = (x * mask_expanded).sum(dim=1) / token_counts  # (B*S, H)
        pooled = pooled.reshape(B, S, self.hidden_size)

        return self.projector(pooled, sentence_flat.reshape(B, S))


class Predictor(nn.Module):
    """Transformer predictor over sentence-level embeddings."""

    def __init__(self, cfg):
        super().__init__()
        enc = cfg["encoder"]
        pred = cfg["predictor"]
        data = cfg["data"]
        self.embedding_size = enc.get("embedding_size", enc["hidden_size"])
        self.hidden_size = pred["hidden_size"]

        self.sentence_pos_emb = nn.Embedding(
            data["max_sentences"], pred["hidden_size"]
        )
        self.mask_token = nn.Parameter(torch.randn(self.embedding_size) * 0.02)
        self.input_proj = (
            nn.Linear(self.embedding_size, pred["hidden_size"])
            if self.embedding_size != pred["hidden_size"]
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred["hidden_size"],
            nhead=pred["num_heads"],
            dim_feedforward=pred["ffn_size"],
            dropout=pred.get("dropout", 0.1),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=pred["num_layers"]
        )
        self.norm = nn.LayerNorm(pred["hidden_size"])
        self.pred_projector = ProjectionHead(
            input_dim=pred["hidden_size"],
            hidden_dim=pred.get("projector_hidden_size", 2048),
            output_dim=self.embedding_size,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.sentence_pos_emb.weight, std=0.02)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_sequence(self, encoder_embeddings, sentence_mask, causal=False):
        """Predict projected sentence embeddings for each sequence position."""
        B, S, _ = encoder_embeddings.shape
        x = self.input_proj(encoder_embeddings)

        positions = torch.arange(S, device=encoder_embeddings.device).unsqueeze(0)
        x = x + self.sentence_pos_emb(positions)

        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=encoder_embeddings.device),
                diagonal=1,
            )

        x = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=~sentence_mask,
        )
        x = self.norm(x)
        return self.pred_projector(x, sentence_mask)

    def forward(self, encoder_embeddings, sentence_mask, mask_indices):
        """
        Args:
            encoder_embeddings: (B, S, D) projected encoder embeddings
            sentence_mask: (B, S) — True for real sentences, False for padding
            mask_indices: (B, S) — True for masked sentence positions
        Returns:
            predictions: (N_masked, D) — predicted embeddings at masked positions
        """
        predictor_input = encoder_embeddings.clone()
        predictor_input[mask_indices] = self.mask_token

        x = self.forward_sequence(predictor_input, sentence_mask, causal=False)
        predictions = x[mask_indices]  # (N_masked, D)
        return predictions


class SentenceJEPA(nn.Module):
    """Full SentenceJEPA: encoder + predictor, end-to-end (no EMA, no stop-grad)."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = SentenceEncoder(cfg)
        self.predictor = Predictor(cfg)

    def forward_masked(self, input_ids, attention_mask, sentence_mask, mask_indices):
        """
        Args:
            input_ids: (B, S, T)
            attention_mask: (B, S, T)
            sentence_mask: (B, S) — True for real sentences
            mask_indices: (B, S) — True for masked positions
        Returns:
            pred_out: (N_masked, D) predicted embeddings
            targets: (N_masked, D) original encoder embeddings at masked positions
            enc_out: (B, S, D) all encoder embeddings (for SIGReg)
        """
        enc_out = self.encoder(input_ids, attention_mask)  # (B, S, D)
        pred_out = self.predictor(enc_out, sentence_mask, mask_indices)
        targets = enc_out[mask_indices]

        return pred_out, targets, enc_out, mask_indices

    def forward_next_sentence(self, input_ids, attention_mask, sentence_mask):
        """Causal next-sentence JEPA objective in projected embedding space."""
        enc_out = self.encoder(input_ids, attention_mask)  # (B, S, D)
        pred_seq = self.predictor.forward_sequence(enc_out, sentence_mask, causal=True)
        pair_mask = sentence_mask[:, :-1] & sentence_mask[:, 1:]

        pred_out = pred_seq[:, :-1][pair_mask]
        targets = enc_out[:, 1:][pair_mask]

        return pred_out, targets, enc_out, pair_mask

    def forward(
        self,
        input_ids,
        attention_mask,
        sentence_mask,
        mask_indices=None,
        mode="masked",
    ):
        if mode == "next_sentence":
            return self.forward_next_sentence(input_ids, attention_mask, sentence_mask)
        if mode == "masked":
            if mask_indices is None:
                raise ValueError("mask_indices is required for masked objective")
            return self.forward_masked(
                input_ids, attention_mask, sentence_mask, mask_indices
            )

        raise ValueError(f"Unknown objective mode: {mode}")
