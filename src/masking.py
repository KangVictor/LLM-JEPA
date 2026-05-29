import torch


def _sample_span_positions(real_positions, num_masks, cfg):
    """Sample up to num_masks positions as one or more contiguous spans."""
    n = real_positions.size(0)
    device = real_positions.device
    selected = torch.zeros(n, dtype=torch.bool, device=device)

    span_min = int(cfg.get("span_len_min", 1))
    span_max = int(cfg.get("span_len_max", max(1, num_masks)))
    num_spans_min = int(cfg.get("num_spans_min", 1))
    num_spans_max = int(cfg.get("num_spans_max", max(1, num_masks)))
    max_spans = max(num_spans_min, num_spans_max)
    num_spans = int(
        torch.randint(num_spans_min, max_spans + 1, (1,), device=device).item()
    )
    num_spans = max(1, min(num_spans, num_masks))

    attempts = 0
    while selected.sum().item() < num_masks and attempts < n * max(4, num_spans):
        remaining = num_masks - int(selected.sum().item())
        span_len_hi = min(span_max, remaining, n)
        span_len_lo = min(span_min, span_len_hi)
        span_len = int(
            torch.randint(span_len_lo, span_len_hi + 1, (1,), device=device).item()
        )
        start = int(torch.randint(0, n - span_len + 1, (1,), device=device).item())
        selected[start : start + span_len] = True
        attempts += 1
        if attempts >= num_spans and selected.sum().item() >= num_masks:
            break

    if selected.sum().item() < num_masks:
        remaining_positions = (~selected).nonzero(as_tuple=True)[0]
        fill_count = min(num_masks - int(selected.sum().item()), remaining_positions.numel())
        if fill_count > 0:
            perm = torch.randperm(remaining_positions.numel(), device=device)[:fill_count]
            selected[remaining_positions[perm]] = True

    return real_positions[selected.nonzero(as_tuple=True)[0][:num_masks]]


def sample_masks(sentence_mask, cfg):
    """Sample sentence positions to mask for JEPA prediction.

    Args:
        sentence_mask: (B, S) bool tensor — True for real sentences
        cfg: masking config dict with keys:
            mask_ratio_min, mask_ratio_max, max_mask_count, multi_mask

    Returns:
        mask_indices: (B, S) bool tensor — True for masked positions
        mask_counts: (B,) int tensor — number of masks per sample
    """
    B, S = sentence_mask.shape
    device = sentence_mask.device

    mask_indices = torch.zeros(B, S, dtype=torch.bool, device=device)
    mask_counts = torch.zeros(B, dtype=torch.long, device=device)

    ratio_min = cfg["mask_ratio_min"]
    ratio_max = cfg["mask_ratio_max"]
    max_count = cfg["max_mask_count"]
    multi = cfg["multi_mask"]
    span_masking = cfg.get("span_masking", False)

    for i in range(B):
        real_positions = sentence_mask[i].nonzero(as_tuple=True)[0]
        n = real_positions.size(0)
        if n == 0:
            continue

        if multi:
            ratio = torch.empty(1, device=device).uniform_(ratio_min, ratio_max).item()
            num_masks = max(1, min(round(n * ratio), max_count))
        else:
            num_masks = 1
        if cfg.get("keep_at_least_one_visible", True) and n > 1:
            num_masks = min(num_masks, n - 1)

        if span_masking:
            selected = _sample_span_positions(real_positions, num_masks, cfg)
        else:
            perm = torch.randperm(n, device=device)[:num_masks]
            selected = real_positions[perm]
        mask_indices[i, selected] = True
        mask_counts[i] = num_masks

    return mask_indices, mask_counts
