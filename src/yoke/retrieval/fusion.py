"""Reciprocal Rank Fusion (RRF) for merging ranked lists."""


def rrf_merge(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k_rrf: int = 60,
) -> list[tuple[int, float, int | None, int | None]]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results: List of (chunk_id, score) sorted by score desc.
        sparse_results: List of (chunk_id, score) sorted by score desc.
        k_rrf: RRF constant (default 60, per the original paper).

    Returns:
        List of (chunk_id, rrf_score, dense_rank, sparse_rank) sorted by
        rrf_score descending. Ranks are 1-based; None if absent from that list.
    """
    # Build rank maps (1-based)
    dense_rank: dict[int, int] = {
        cid: rank for rank, (cid, _) in enumerate(dense_results, 1)
    }
    sparse_rank: dict[int, int] = {
        cid: rank for rank, (cid, _) in enumerate(sparse_results, 1)
    }

    all_ids = set(dense_rank) | set(sparse_rank)

    scored: list[tuple[int, float, int | None, int | None]] = []
    for cid in all_ids:
        d_rank = dense_rank.get(cid)
        s_rank = sparse_rank.get(cid)
        score = 0.0
        if d_rank is not None:
            score += 1.0 / (k_rrf + d_rank)
        if s_rank is not None:
            score += 1.0 / (k_rrf + s_rank)
        scored.append((cid, score, d_rank, s_rank))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
