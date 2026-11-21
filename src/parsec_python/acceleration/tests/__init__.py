"""Tests for additive acceleration backends."""

import os

# The small periodic cells of these tests exercise the sectors of their point groups, which the size floor of the
# Gamma sectors (PARSEC_PERIODIC_SECTOR_FLOOR, 2^24 grid points x states by default) would replace by the full grid:
# it is off here unless a test (or the caller) sets it; test_periodic_sector_floor.py checks the default.
os.environ.setdefault("PARSEC_PERIODIC_SECTOR_FLOOR", "off")
# Their sector counts, weights and projector orbits are those of the plan of the least rows x states: the fixed cost
# of a sector in the choice of the plan of a cell (PARSEC_SECTOR_OVERHEAD) is off here unless a test sets it;
# test_sector_overhead.py checks the default.
os.environ.setdefault("PARSEC_SECTOR_OVERHEAD", "off")
# And they spread the sectors of their small cells over every GPU of the process, as the multi-GPU tests mean to:
# the one GPU of small periodic sector bases (PARSEC_PERIODIC_SECTOR_DEVICES) is off here unless a test sets it;
# test_periodic_sector_devices.py checks the default.
os.environ.setdefault("PARSEC_PERIODIC_SECTOR_DEVICES", "all")
