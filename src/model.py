import torch
import torch.nn.functional as F
import torch.nn as nn


def masked_mean_pool(x, mask, normalize=False):
    """Mean-pool valid sequence positions.

    Args:
        x: (B, S, D) tensor
        mask: (B, S) bool tensor
        normalize: whether to L2-normalize the pooled output

    Returns:
        pooled: (B, D)
    """
    weights = mask.unsqueeze(-1).to(dtype=x.dtype)
    pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
    return pooled


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
        if valid.any():
            valid_projected = self.net(flat[valid])
            projected = valid_projected.new_zeros(flat.size(0), self.output_dim)
            projected[valid] = valid_projected
        else:
            projected = flat.new_zeros(flat.size(0), self.output_dim)
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
        predictor_input[mask_indices] = self.mask_token.to(dtype=predictor_input.dtype)

        x = self.forward_sequence(predictor_input, sentence_mask, causal=False)
        predictions = x[mask_indices]  # (N_masked, D)
        return predictions


class DocumentTransformer(nn.Module):
    """Contextualizes sentence embeddings with paragraph-level self-attention."""

    def __init__(self, cfg):
        super().__init__()
        enc = cfg["encoder"]
        doc = cfg.get("document", {})
        data = cfg["data"]

        self.embedding_size = enc.get("embedding_size", enc["hidden_size"])
        self.hidden_size = doc.get("hidden_size", self.embedding_size)
        self.max_sentences = doc.get("max_sentences", data["max_sentences"])

        self.input_proj = (
            nn.Linear(self.embedding_size, self.hidden_size)
            if self.embedding_size != self.hidden_size
            else nn.Identity()
        )
        self.sentence_pos_emb = nn.Embedding(self.max_sentences, self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=doc.get("num_heads", enc["num_heads"]),
            dim_feedforward=doc.get("ffn_size", enc.get("ffn_size", 1024)),
            dropout=doc.get("dropout", enc.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=doc.get("num_layers", 2)
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_proj = (
            nn.Linear(self.hidden_size, self.embedding_size)
            if self.hidden_size != self.embedding_size
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.sentence_pos_emb.weight, std=0.02)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sentence_embeddings, sentence_mask, position_ids=None):
        """
        Args:
            sentence_embeddings: (B, S, D)
            sentence_mask: (B, S) bool — True for sentences visible to the branch
            position_ids: optional (B, S) original sentence positions

        Returns:
            contextual_sentence_embeddings: (B, S, D)
        """
        B, S, _ = sentence_embeddings.shape
        if S > self.max_sentences:
            raise ValueError(
                f"DocumentTransformer saw {S} sentences but max_sentences="
                f"{self.max_sentences}. Increase document.max_sentences or "
                "data.max_sentences."
            )

        if position_ids is None:
            position_ids = torch.arange(S, device=sentence_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(B, S)
        else:
            position_ids = position_ids.to(device=sentence_embeddings.device)
        position_ids = position_ids.clamp(max=self.max_sentences - 1)

        visible = sentence_mask.unsqueeze(-1).to(dtype=sentence_embeddings.dtype)
        x = self.input_proj(sentence_embeddings * visible)
        x = x + self.sentence_pos_emb(position_ids)

        x = self.transformer(x, src_key_padding_mask=~sentence_mask)
        x = self.norm(x)
        x = self.output_proj(x)
        return x * visible


class ContextQueryPredictor(nn.Module):
    """Predict target contextual sentence embeddings from visible context tokens."""

    def __init__(self, cfg):
        super().__init__()
        enc = cfg["encoder"]
        pred = cfg["predictor"]
        data = cfg["data"]
        doc = cfg.get("document", {})

        self.embedding_size = enc.get("embedding_size", enc["hidden_size"])
        self.hidden_size = pred["hidden_size"]
        self.max_sentences = doc.get("max_sentences", data["max_sentences"])

        self.input_proj = (
            nn.Linear(self.embedding_size, self.hidden_size)
            if self.embedding_size != self.hidden_size
            else nn.Identity()
        )
        self.sentence_pos_emb = nn.Embedding(self.max_sentences, self.hidden_size)
        self.target_query = nn.Parameter(torch.randn(self.hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=pred["num_heads"],
            dim_feedforward=pred["ffn_size"],
            dropout=pred.get("dropout", 0.1),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=pred["num_layers"]
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.pred_projector = ProjectionHead(
            input_dim=self.hidden_size,
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

    def forward(self, context_embeddings, visible_mask, target_mask, position_ids=None):
        """
        Args:
            context_embeddings: (B, S, D) contextual visible sentence outputs
            visible_mask: (B, S) bool — visible context sentence slots
            target_mask: (B, S) bool — target query slots to predict
            position_ids: optional (B, S) original sentence positions

        Returns:
            predictions: (N_targets, D)
        """
        B, S, _ = context_embeddings.shape
        if S > self.max_sentences:
            raise ValueError(
                f"ContextQueryPredictor saw {S} sentences but max_sentences="
                f"{self.max_sentences}. Increase document.max_sentences or "
                "data.max_sentences."
            )

        if position_ids is None:
            position_ids = torch.arange(S, device=context_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(B, S)
        else:
            position_ids = position_ids.to(device=context_embeddings.device)
        position_ids = position_ids.clamp(max=self.max_sentences - 1)

        pos = self.sentence_pos_emb(position_ids)
        context_tokens = self.input_proj(context_embeddings) + pos
        query_tokens = self.target_query.view(1, 1, -1) + pos

        x = torch.cat([context_tokens, query_tokens], dim=1)
        token_mask = torch.cat([visible_mask, target_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=~token_mask)
        x = self.norm(x)

        query_out = x[:, S:]
        pred_all = self.pred_projector(query_out, target_mask)
        return pred_all[target_mask]


class SentenceJEPA(nn.Module):
    """Hierarchical Paragraph-JEPA: sentence encoder + document transformer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SentenceEncoder(cfg)
        self.document_transformer = DocumentTransformer(cfg)
        self.predictor = ContextQueryPredictor(cfg)
        self.detach_target = cfg.get("objective", {}).get("detach_target", False)
        self.freeze_sentence_encoder_flag = cfg["encoder"].get("freeze", False)

        if self.freeze_sentence_encoder_flag:
            self.set_sentence_encoder_trainable(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_sentence_encoder_flag:
            self.encoder.eval()
        return self

    def set_sentence_encoder_trainable(self, trainable=True):
        for p in self.encoder.parameters():
            p.requires_grad = trainable
        self.freeze_sentence_encoder_flag = not trainable
        if not trainable:
            self.encoder.eval()

    def encode_contextual(
        self,
        input_ids,
        attention_mask,
        sentence_mask,
        visible_mask=None,
        position_ids=None,
    ):
        """Encode sentences, then contextualize them at paragraph level."""
        sentence_embeddings = self.encoder(input_ids, attention_mask)
        doc_mask = sentence_mask if visible_mask is None else visible_mask
        contextual = self.document_transformer(
            sentence_embeddings,
            doc_mask,
            position_ids=position_ids,
        )
        return contextual

    def encode_document(
        self,
        input_ids,
        attention_mask,
        sentence_mask,
        normalize=True,
        return_contextual=False,
    ):
        """Inference path: contextualize all sentences and mean-pool."""
        contextual = self.encode_contextual(input_ids, attention_mask, sentence_mask)
        document_embeddings = masked_mean_pool(
            contextual,
            sentence_mask,
            normalize=normalize,
        )
        if return_contextual:
            return document_embeddings, contextual
        return document_embeddings

    def forward_masked_from_sentence_embeddings(
        self,
        sentence_embeddings,
        sentence_mask,
        mask_indices,
        target_contextual=None,
        position_ids=None,
    ):
        """Predict target contextual sentence embeddings for sampled masks."""
        if target_contextual is None:
            target_contextual = self.document_transformer(
                sentence_embeddings,
                sentence_mask,
                position_ids=position_ids,
            )

        visible_mask = sentence_mask & ~mask_indices
        context_contextual = self.document_transformer(
            sentence_embeddings,
            visible_mask,
            position_ids=position_ids,
        )
        pred_out = self.predictor(
            context_contextual,
            visible_mask,
            mask_indices,
            position_ids=position_ids,
        )

        targets = target_contextual[mask_indices]
        if self.detach_target:
            targets = targets.detach()

        document_embeddings = masked_mean_pool(
            target_contextual,
            sentence_mask,
            normalize=False,
        )
        return pred_out, targets, target_contextual, document_embeddings, mask_indices

    def forward_masked(self, input_ids, attention_mask, sentence_mask, mask_indices):
        """
        Args:
            input_ids: (B, S, T)
            attention_mask: (B, S, T)
            sentence_mask: (B, S) — True for real sentences
            mask_indices: (B, S) — True for masked positions
        Returns:
            pred_out: (N_masked, D) predicted embeddings
            targets: (N_masked, D) contextual target embeddings at masked positions
            contextual: (B, S, D) full contextual sentence embeddings
            document_embeddings: (B, D) mean-pooled contextual document embeddings
        """
        sentence_embeddings = self.encoder(input_ids, attention_mask)
        return self.forward_masked_from_sentence_embeddings(
            sentence_embeddings,
            sentence_mask,
            mask_indices,
        )

    def forward_next_sentence(self, input_ids, attention_mask, sentence_mask):
        raise NotImplementedError(
            "next_sentence is not implemented for hierarchical Paragraph-JEPA. "
            "Use objective.mode='masked'."
        )

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
