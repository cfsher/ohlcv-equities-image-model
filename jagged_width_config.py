#!/usr/bin/env python3
"""Shared feature flags for jagged image-width handling."""

from __future__ import annotations

# Global kill-switch for per-scale jagged image widths (e.g., 32x18 vs 32x15).
# True  -> use per-scale widths from manifest/build spec.
# False -> force legacy fixed-width behavior for all scales.
JAGGED_IMAGE_WIDTHS_ENABLED = True

# Jagged-width 2D fusion strategy (applies only when JAGGED_IMAGE_WIDTHS_ENABLED=True):
# 1 -> option 1: right-pad narrower maps to max width, then channel-concat in 2D.
# 2 -> option 2: right-crop wider maps to min width, then channel-concat in 2D.
# 3 -> option 3: resize every map to JAGGED_OPTION3_TARGET_WIDTH, then channel-concat in 2D.
# 4 -> option 4: like option 1, plus append one valid-width mask channel per scale.
JAGGED_2D_CONCAT_OPTION = 1

# Target width used only for option 3.
JAGGED_OPTION3_TARGET_WIDTH = 15
