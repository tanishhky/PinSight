"""Risk-neutral density extraction.

Pipeline (see docs/research/architecture.md §2):
    raw chain -> quality filter -> IV recompute -> SVI fit ->
    fine-grid reprice -> BL second derivative -> tail extrapolation ->
    normalize -> BKM cross-check.
"""
