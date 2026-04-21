import torch


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

        # Randomly select positions
        perm = torch.randperm(n, device=device)[:num_masks]
        selected = real_positions[perm]
        mask_indices[i, selected] = True
        mask_counts[i] = num_masks

    return mask_indices, mask_counts
