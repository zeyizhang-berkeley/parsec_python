"""Tests for additive acceleration backends."""

import os

# The small periodic cells of these tests exercise the sectors of their point groups, which the size floor of the
# Gamma sectors (PARSEC_PERIODIC_SECTOR_FLOOR, 2^24 grid points x states by default) would replace by the full grid:
# it is off here unless a test (or the caller) sets it; test_periodic_sector_floor.py checks the default.
os.environ.setdefault("PARSEC_PERIODIC_SECTOR_FLOOR", "off")
