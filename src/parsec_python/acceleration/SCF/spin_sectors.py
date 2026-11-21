"""Spin-polarized SCF loops in the symmetry sectors (``Spin_Symmetry_Sectors``).

A spin-polarized CuPy run of the accelerated driver can solve both channels in
the representation sectors of the point group of the atoms, as an unpolarized
run does (:class:`~parsec_python.acceleration.Eigensolvers.symmetry.CuPySymmetrySCFEigensolver`,
one channel view per spin).  Every sector orbital belongs to a real
one-dimensional representation, so ``|psi|^2`` is invariant and so are both
channel densities, the total density and every potential: the loop keeps one
value per symmetry orbit for all of them (:class:`SpinSectorReducer`), solves
Hartree on the wedge and evaluates the spin functional on the orbits.

The sectors cannot represent a density that breaks the point group of the
sector representations, and they constrain a state that breaks the larger
point group of the atoms to the directions that group allows.  The full-grid
loop of a partly filled degenerate level (the oxygen atom: one minority p
electron in three degenerate p levels; an open d shell, the Sc atom with one d
electron in five) breaks the symmetry and finds a state that no sector state
holds: at 80 K the sectors keep the symmetric ensemble (one third in each p
orbital, 68 mRy above for the oxygen atom), at integer occupations an
axis-aligned partner (14 mRy above for the O atom of the fixture grid, 14.8 mRy
for the Sc atom).

The sectors can only be wrong where degenerate partners lie in different
sectors: partners within one sector mix there as freely as on the full grid.
A multiplet (:func:`sector_multiplets`) is therefore a run of levels of one
channel whose neighbours lie within :data:`START_MULTIPLET_WIDTH` = 0.01 Ry,
whose whole spread is at most :data:`START_MULTIPLET_SPREAD` = 0.03 Ry and
whose levels lie in at least two sectors.  The width is above the splitting of
physical multiplets by the grid where they are resolved (measured, PBE TM
potentials, h 0.30 bohr: the 3d levels of Sc to Fe split into e_g and t_2g by
2.6-18 mRy, each part exactly degenerate; p levels exact; 0.3 mRy at h 0.20),
and the spread bound keeps out a band of levels: the valence band of a large
cluster and the unconverged top of a first subspace, where the levels of the
first filter lie closer than the width over several eV (the nanodiamonds
C459H204 and C795H300: chains of 159 to 1818 levels through the Fermi level of
the first iteration).  The sector result is kept only where three tests pass
(each raises :class:`SpinSectorsAbandoned`, after which the driver prepares the
run again on the full grid and starts it there):

1. The start Hamiltonian (the first SCF iteration of a run, from the superposed
   atomic densities, which carries the full point group of the atoms on the
   grid): no multiplet of a channel is partly filled
   (:func:`partly_filled_multiplets`): ``n`` electrons in ``d`` levels with
   ``min(n, d - n) > 0.05`` (:data:`SPIN_SECTOR_OCCUPATION_TOLERANCE`: a
   partly filled shell holds a third of an electron or more per level, while
   the Fermi-Dirac tail of a chain of empty levels holds a few 1e-3: nine
   empty levels of C459H204+ at 500 K hold 3.4e-3 electrons).  A partly filled multiplet of fixed occupations
   (``occup.in``) that are equal on all its levels is kept: its density is
   invariant on the full grid as well.  A report counts once the levels are
   converged: the spin loop solves the start Hamiltonian again from its
   subspaces (up to :data:`START_SPECTRUM_PASSES` = 4 times) and abandons the
   sectors when the report stays and the levels up to the Fermi level moved
   by at most :data:`START_LEVEL_CHANGE` = 1e-4 Ry in the last solve (a small
   system: one further solve, its partners agree to 1e-8 Ry).  A report on
   the unconverged levels of a first filter (a large cluster) disappears or
   keeps moving, and the run goes on: tests 2 and 3 decide at the end.
2. At the end of every SCF loop without an exchange operator and of the whole
   run: each channel density is invariant under the operations of the point
   group of the atoms on the grid (the signed permutations of the axes that map
   the atoms and the grid onto themselves, also those outside the sector
   group; :func:`atom_group_images`; for a bulk cell the operations
   ``r -> R r + t`` of the sector group and of the cell group with their
   translations, about the site the sectors of the cell are centred on, a
   defect off the origin included, and the signed permutations about the
   origin, ``PARSEC_SPIN_GUARD_CELL_GROUP``), within
   :data:`DENSITY_SYMMETRY_TOLERANCE` = 0.01 electrons of
   ``int |rho(g r) - rho(r)| dr`` (:func:`density_symmetry_breaking`).  A sector
   state with a partly filled multiplet whose partners an operation of the
   atoms relates (the p or d levels of an atom, the pi levels of a molecule
   along an axis) breaks it by about an electron (V 1.25, Co 1.17, the O atom
   0.5 to 1); an invariant state differs by what the SCF leaves, since the
   sectors enforce only the abelian subgroup: 6e-7 to 2e-5 electrons for small
   systems, 3e-3 to 4e-3 for C459H204 and C185H124 at Convergence_Criterion
   2e-4 Ry, 1.1e-2 to 2.2e-2 at 2e-3 Ry (the looser criterion of the first
   steps of a relaxation with Relaxation_SCF_Upscale).  The test therefore
   tells the two apart by convergence: where a converged reduced loop exceeds
   the tolerance by less than :data:`DENSITY_SYMMETRY_NOISE_BOUND` = 0.2
   electrons, the loop goes on to a criterion ten times tighter (at most
   :data:`DENSITY_SYMMETRY_TIGHTENINGS` = 3 times) and measures again; noise
   shrinks with the residual, a broken state keeps its asymmetry (the loop
   stops tightening once a measure is more than half the one before), and only
   what is left above the tolerance abandons the sectors.  This also catches a
   filling the start did not show (a 4s-3d transfer during the SCF).
3. With Fermi-Dirac occupations, no level of a multiplet (above, or an exact
   one: levels within :data:`MULTIPLET_WIDTH` of each other in two or more
   sectors) holds a fractional occupation at the end, ``min(f, 1 - f) < 0.05``
   (:func:`fractional_occupations`; the symmetric ensemble of a partly filled
   shell holds a third or a half per level).  A fractional single level, or a multiplet within one sector, is
   what the full grid has too.  Fixed occupations are not tested here (test 1
   covers their multiplets; in the weighted sectors ``Occupations.partner_occupations`` refuses, at every step,
   unequal entries of the partners of a weighted state and of degenerate partners that several sectors hold).

The weighted sectors of a subgroup ``H`` of the point group ``G`` of the atoms (``PARSEC_ORBITAL_GROUP_SPIN``,
:class:`~parsec_python.acceleration.Symmetry.PointGroupReduction`) solve one partner of some degenerate levels and
count it ``w`` times; the density of each channel is averaged over ``G`` (the partners no sector holds), so the
potentials keep every operation of ``G`` and test 2 finds no asymmetry there whatever the occupations.  What breaks
``G`` shows in the levels instead: partners of one level stay degenerate, and a partly filled multiplet holds a
fractional occupation of a state of weight ``w > 1``, or of the partners of a level that several sectors of ``H``
hold.  Tests 1 and 3 therefore count a state of weight ``w`` as ``w`` partner levels in distinct sectors
(:func:`partner_levels`): a triplet of Td held by one state of D2d with one electron is a partly filled multiplet
of three levels, as the same triplet in three sectors of D2 is.  A pair of partners with integer but unequal
occupations (one filled, one empty: degenerate within :data:`MULTIPLET_WIDTH`), which the average over ``G``
would turn into the symmetric ensemble, fails test 3 too (``min(f, 1 - f)`` of the mean occupation of the
multiplet).  A run that fails starts again in the abelian sectors.

With ``PARSEC_SPIN_FLIP_SYMMETRY=auto`` (:mod:`.spin_flip`) the loop solves the
up channel of an antiferromagnetic start alone and takes the down channel as its
image under an operation that exchanges the partner labels; the tests see that
down channel as it is: test 1 and 3 its levels (those of the up channel) with
the images of the up channel's sector labels, test 2 its density, the image of
the up density (which passes where the up density passes: the operation maps
the group of the labels onto itself).

What the tests do not see: a partly filled multiplet whose partners no grid
operation of the atoms relates (the e levels of a C3 or C6 axis) and that is
not partly filled at the start, a multiplet that only the first filter of a
large system leaves unresolved and whose sector state ends invariant and
integer, a state that passes through a partly filled multiplet during the SCF
and returns to an invariant integer one, and a symmetric state with integer
occupations that a rotation across a gap would lower (a closed shell in the
sectors has the same exposure).

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import SimpleNamespace

import numpy as np

from parsec_python.Energy.total_energy import _breakdown
from parsec_python.Mixer import make_mixer
from parsec_python.Mixer.anderson import EMPTY_CHANNEL_ELECTRONS, SpinResidualMetrics
from parsec_python.V_xc.ca_lda_spin import SpinXCResult

from .symmetry_fields import SymmetryAndersonMixer, SymmetrySCFReducer, SymmetryWeightedMixer

#: Electrons below which a multiplet counts as filled or empty (tests 1 and 3): above Fermi-Dirac tails.
SPIN_SECTOR_OCCUPATION_TOLERANCE = 5.0e-2
#: Levels of a channel closer than this (Ry) are listed as one multiplet in a report.
MULTIPLET_WIDTH = 1.0e-4
#: Levels of one channel whose neighbours lie closer than this (Ry) form one group (tests 1 and 3, see the module).
START_MULTIPLET_WIDTH = 1.0e-2
#: The largest spread (Ry) of such a group that counts as a multiplet: a longer chain is a band of levels.
START_MULTIPLET_SPREAD = 3.0e-2
#: Further solves of the start Hamiltonian before a partly filled multiplet of test 1 counts (converged levels).
START_SPECTRUM_PASSES = 4
#: Largest change (Ry) of the levels up to the Fermi level between two solves of the start Hamiltonian.
START_LEVEL_CHANGE = 1.0e-4
#: Electrons by which a channel density may differ from its image under an operation of the atoms.
DENSITY_SYMMETRY_TOLERANCE = 1.0e-2
#: Above this many electrons of asymmetry a state is broken, whatever the SCF residual (test 2).
DENSITY_SYMMETRY_NOISE_BOUND = 0.2
#: Times a converged reduced loop may tighten its criterion tenfold to see whether an asymmetry is noise.
DENSITY_SYMMETRY_TIGHTENINGS = 3
#: Sector labels of the partner levels of a weighted state (:func:`partner_levels`): sector times this plus partner.
PARTNER_LABEL_STRIDE = 1024
_RYDBERG_TO_EV = 13.605693122994
_CHANNEL_NAMES = ("up", "down")


class SpinSectorsAbandoned(RuntimeError):
    """A spin-polarized sector loop failed a test of the module.

    ``stage`` names the loop (``the start Hamiltonian``, ``the SCF loop``,
    ``the semilocal start of the Fock loop``, ``the end of the Fock loop``),
    ``report`` what was found.
    """

    def __init__(self, stage: str, report: "FractionalOccupations") -> None:
        self.stage = stage
        self.report = report
        super().__init__(
            f"Spin_Symmetry_Sectors: {report.reason} after {stage} "
            f"({report.measure} = {report.largest:.3e} >= {report.tolerance:.0e}): "
            + "; ".join(report.lines)
        )


@dataclass(frozen=True)
class FractionalOccupations:
    """What a test of the module found: the largest value of its ``measure``, the tolerance and per-item lines."""

    largest: float
    tolerance: float
    lines: tuple[str, ...]
    reason: str = "fractional occupations"
    measure: str = "largest min(f, 1-f)"


def _multiplets(values: np.ndarray, width: float) -> list[list[int]]:
    """Runs of levels (indices, ascending) whose neighbours lie within ``width``."""

    order = np.argsort(values, kind="stable")
    groups: list[list[int]] = []
    for position in order:
        if groups and values[position] - values[groups[-1][-1]] <= width:
            groups[-1].append(int(position))
        else:
            groups.append([int(position)])
    return groups


def sector_multiplets(
    values: np.ndarray,
    sectors: np.ndarray,
    *,
    width: float = START_MULTIPLET_WIDTH,
    spread: float = START_MULTIPLET_SPREAD,
) -> list[list[int]]:
    """The multiplets of one channel that the sectors constrain (indices, ascending).

    A run of levels whose neighbours lie within ``width`` and whose whole
    spread is at most ``spread`` (a longer run is a band: the unconverged top
    of a first subspace, the valence band of a large cluster), with levels in
    at least two sectors: partners within one sector mix there as freely as on
    the full grid.
    """

    sectors = np.asarray(sectors)
    return [
        group
        for group in _multiplets(values, width)
        if len(group) >= 2
        and float(np.ptp(values[group])) <= spread
        and np.unique(sectors[group]).size >= 2
    ]


def partly_filled_multiplets(
    eigenvalues,
    occupations,
    representations,
    *,
    fixed: bool,
    width: float = START_MULTIPLET_WIDTH,
    spread: float = START_MULTIPLET_SPREAD,
    tolerance: float = SPIN_SECTOR_OCCUPATION_TOLERANCE,
):
    """:class:`FractionalOccupations` of the partly filled multiplets of the channels, ``None`` where there are none.

    ``eigenvalues``, ``occupations`` and ``representations`` hold one array
    per channel (Ry; electrons per level; sector labels).  A multiplet of
    :func:`sector_multiplets` with ``d`` levels holding ``n`` electrons is
    partly filled when ``min(n, d - n) > tolerance``; with ``fixed``
    occupations (``occup.in``) one whose occupations are all equal is not (see
    the module).
    """

    largest = 0.0
    lines: list[str] = []
    for index, (values, filled, sectors) in enumerate(zip(eigenvalues, occupations, representations)):
        values = np.asarray(values, dtype=np.float64)
        filled = np.asarray(filled, dtype=np.float64)
        sectors = np.asarray(sectors, dtype=np.int64)
        for group in sector_multiplets(values, sectors, width=width, spread=spread):
            electrons = float(np.sum(filled[group]))
            part = min(electrons, len(group) - electrons)
            if part <= tolerance:
                continue
            if fixed and float(np.ptp(filled[group])) < 1.0e-12:
                continue
            largest = max(largest, part)
            levels = " ".join(f"{values[member] * _RYDBERG_TO_EV:.4f}" for member in group)
            labels = " ".join(_sector_label(sectors[member]) for member in group)
            lines.append(
                f"{_CHANNEL_NAMES[index] if index < 2 else index} channel: {len(group)} levels within "
                f"{width:g} Ry at {levels} eV hold {electrons:.4f} electrons (sectors {labels})"
            )
    if not lines:
        return None
    return FractionalOccupations(
        largest=largest,
        tolerance=float(tolerance),
        lines=tuple(lines),
        reason="a partly filled multiplet",
        measure="largest min(n, d - n) of d levels holding n electrons",
    )


def _sector_label(value) -> str:
    """A sector label of a report: ``s`` or, for a partner level of a weighted state, ``s.k`` (its partner k)."""

    value = int(value)
    if value < PARTNER_LABEL_STRIDE:
        return str(value)
    return f"{value // PARTNER_LABEL_STRIDE}.{value % PARTNER_LABEL_STRIDE + 1}"


def partner_levels(values, filled, sectors, weights):
    """The levels of one channel with every partner of a weighted state as a level of its own.

    ``values`` (Ry), ``filled`` (electrons per state: the occupation times the weight) and ``sectors`` (labels) of
    the states the weighted sectors solved, ``weights`` their weights (``None``: no weights, returned as given).  A
    state of weight ``w`` becomes ``w`` levels of its energy holding ``filled / w`` electrons each, labelled
    ``sector * PARTNER_LABEL_STRIDE + k`` for its ``k``-th partner: partners lie in distinct sectors, as they do in
    the sectors of an abelian group, which is what the multiplet tests of the module ask.
    """

    values = np.asarray(values, dtype=np.float64)
    filled = np.asarray(filled, dtype=np.float64)
    sectors = np.asarray(sectors, dtype=np.int64)
    if weights is None:
        return values, filled, sectors
    repeats = np.rint(np.asarray(weights, dtype=np.float64)).astype(np.int64)
    if repeats.shape != values.shape or np.any(repeats < 1):
        raise ValueError("one positive integer weight per state is needed")
    partners = np.arange(int(repeats.sum()), dtype=np.int64) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    return (
        np.repeat(values, repeats),
        np.repeat(filled / repeats, repeats),
        np.repeat(sectors, repeats) * PARTNER_LABEL_STRIDE + partners,
    )


def _key(matrix: np.ndarray) -> tuple:
    return tuple(int(value) for value in np.rint(matrix).ravel())


def _generators(matrices, seed=()) -> list[np.ndarray]:
    """A subset of ``matrices`` (a group of signed permutations) that generates them together with ``seed``."""

    elements = {_key(np.eye(3)): np.eye(3)}
    generators: list[np.ndarray] = [np.rint(np.asarray(matrix, dtype=np.float64)) for matrix in seed]
    grown = bool(generators)
    while grown:
        grown = False
        for element in list(elements.values()):
            for generator in generators:
                product = element @ generator
                if _key(product) not in elements:
                    elements[_key(product)] = product
                    grown = True
    seeded = len(generators)
    for matrix in matrices:
        matrix = np.rint(np.asarray(matrix, dtype=np.float64))
        if _key(matrix) in elements:
            continue
        generators.append(matrix)
        grown = True
        while grown:
            grown = False
            for element in list(elements.values()):
                for generator in generators:
                    product = element @ generator
                    if _key(product) not in elements:
                        elements[_key(product)] = product
                        grown = True
    return generators[seeded:]


def _grid_images(grid, matrix: np.ndarray):
    """Rows of the images ``matrix r`` of the grid points; ``None`` where the operation does not map the grid onto itself."""

    shift = np.asarray(grid.settings.shift, dtype=np.float64)
    target = (np.asarray(grid.integer_coordinates, dtype=np.float64) + shift) @ np.asarray(matrix).T - shift
    rounded = np.rint(target)
    if np.max(np.abs(target - rounded), initial=0.0) > 1.0e-9:
        return None
    rows = grid.rows_for_integer_coordinates(rounded.astype(np.int64))
    if np.any(rows < 0):
        return None
    return rows


def _periodic_atom_matrices(grid, atoms) -> list[np.ndarray] | None:
    """The signed permutations about the origin that map the atoms of an orthorhombic cell onto themselves modulo
    its lattice.

    ``None`` for a grid without an orthorhombic cell (a cluster: the atoms as they are listed,
    :func:`parsec_python.Forces.symmetry.atom_symmetry_operations`).  A cell lists each atom once, so an atom on the
    face of the cell has its image on the opposite face outside the list: the operations of the listed atoms would
    lose every sign change that moves it there (a vacancy at the origin of a diamond supercell keeps the six
    permutations of the axes of its 24 operations).  These are the operations about the origin alone: a defect off
    the origin keeps few of them (a diamond vacancy at ``(a/2, a/2, 0)`` the swap of x and y), while the sectors of
    the cell are taken about the defect.  Test 2 of a cell therefore takes the operations of the cell group with
    their translations as well (:func:`atom_group_images`, ``PARSEC_SPIN_GUARD_CELL_GROUP``); these remain among
    its candidates, and the group it tests holds every one of them (:func:`_cell_group_images` tests one on its own
    where the group would grow beyond its limit with it).
    """

    from scipy.spatial import cKDTree

    from parsec_python.Forces.symmetry import POSITION_TOLERANCE, cubic_point_operations

    lattice = getattr(getattr(grid, "cell", None), "lattice_vectors", None)
    if lattice is None:
        return None
    lattice = np.asarray(lattice, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-10):
        return None
    sides = np.diag(lattice).copy()
    if np.any(sides <= 0.0):
        return None
    positions = np.mod(np.array([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3), sides)
    symbols = [atom.symbol for atom in atoms]
    members = {symbol: np.flatnonzero([s == symbol for s in symbols]) for symbol in dict.fromkeys(symbols)}
    # a point on the upper face is the point on the lower one: wrapped into [0, side) by the tree's box
    trees = {
        symbol: cKDTree(np.mod(positions[rows], sides) % sides, boxsize=sides) for symbol, rows in members.items()
    }
    found: list[np.ndarray] = []
    for matrix in cubic_point_operations():
        matrix = np.rint(np.asarray(matrix, dtype=np.float64))
        if not np.allclose(np.abs(matrix) @ sides, sides, atol=1.0e-10):
            # a permutation of axes of different lengths does not map the cell onto itself
            continue
        rotated = np.mod(positions @ matrix.T, sides) % sides
        ok = True
        for symbol, rows in members.items():
            distance, nearest = trees[symbol].query(rotated[rows])
            if np.any(distance > POSITION_TOLERANCE) or len(set(np.asarray(nearest).tolist())) != len(rows):
                ok = False
                break
        if ok:
            found.append(matrix)
    return found


_GUARD_CELL_GROUP_SWITCH = "PARSEC_SPIN_GUARD_CELL_GROUP"


def spin_guard_cell_group_setting() -> str:
    """``PARSEC_SPIN_GUARD_CELL_GROUP``: ``auto`` (the default) or ``off``.

    ``auto``: test 2 of a bulk orthorhombic cell (the spin sectors of a cell at the Gamma point,
    ``PARSEC_GAMMA_SPIN_SECTORS``) tests the operations ``r -> R r + t`` that the sectors rely on and the group of
    the cell that contains them (:func:`_cell_group_images`): the elements of the sector group with their
    translations, the operations found from the environments of the atoms or the cell group about its symmetric site
    (:class:`.kpoint_sectors.CellGroup`, the group whose commuting involutions the sectors take) and the signed
    permutations about the origin.  Test 2 of the k-point sectors of a spin-polarized loop
    (``PARSEC_KPOINT_SECTOR_GUARD``: the cell group of the k-point sectors, :class:`.kpoint_sectors.SectorGuardGroup`)
    takes the operations found from the environments as well, those its potential is taken to be invariant under.
    ``off``: the signed permutations about the origin alone (:func:`_periodic_atom_matrices`) on the Gamma sectors,
    the cell group alone on the k-point sectors, as before; both keep few of the operations about a defect off the
    origin of a cell larger than 2 x 2 x 2 diamond cubes.  Clusters are not affected.
    """

    value = os.environ.get(_GUARD_CELL_GROUP_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{_GUARD_CELL_GROUP_SWITCH} must be auto or off, not {value!r}")
    return value


_FALLBACK_SHARED_SWITCH = "PARSEC_SPIN_FALLBACK_SHARED"


def spin_fallback_shared_setting() -> str:
    """``PARSEC_SPIN_FALLBACK_SHARED``: ``auto`` (the default), ``on`` or ``off`` (also ``0``, ``false``, ``no``).

    The layout of the full grid a spin-polarized sector run goes back to where its guard fires
    (``driver._SpinSectorFallback``).  ``off``: both channels on the full grid of the first device of the process
    (the root's in the full-SCF runner), as before, whatever the memory estimate.  ``on``: both channels in one
    sector of the identity, whose basis the GPUs of the process share (``distributed_state``), or the ranks of the
    runner, which all prepare it together; a hybrid's or a meta-GGA's Fock and orbital terms, which keep one device
    per sector basis, keep the layout of ``off``.  ``auto``: ``on`` where the full grid of both channels is estimated
    not to fit the one device of ``off`` (``driver.spin_fallback_plan``), ``off`` elsewhere, so that every fallback
    that fitted runs as before; under ``auto`` and ``on`` a fallback that fits no layout stops before its full grid.
    The same layouts serve a spin cell that the command line solves on the full grid from its start (molecular
    dynamics, manual movements, ``Move_Flag``, ``Ignore_Symmetry``: ``driver.spin_full_grid_plan``), which keeps the
    1x1x1 k-point driver where no layout fits (under ``off`` where one device does not).
    """

    value = os.environ.get(_FALLBACK_SHARED_SWITCH, "auto").strip().lower() or "auto"
    if value in {"0", "false", "no"}:
        value = "off"
    if value in {"1", "true", "yes"}:
        value = "on"
    if value not in ("auto", "on", "off"):
        raise ValueError(f"{_FALLBACK_SHARED_SWITCH} must be auto, on or off, not {value!r}")
    return value


def _is_diagonal(matrix) -> bool:
    matrix = np.asarray(matrix)
    return np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0


def _affine(matrix, shift) -> np.ndarray:
    """``[[R, t], [0, 1]]`` of ``r -> R r + t`` (``t`` in grid steps), integers."""

    affine = np.eye(4, dtype=np.int64)
    affine[:3, :3] = np.rint(np.asarray(matrix, dtype=np.float64)).astype(np.int64)
    affine[:3, 3] = np.asarray(shift, dtype=np.int64)
    return affine


class _AffineClosure:
    """The group that operations ``r -> R r + t`` of a periodic grid generate (``R`` signed permutations, ``t`` in
    grid steps modulo the counts of the grid).

    Elements are kept as arrays and coded as integers: the nine entries of ``R`` (base 3) and the row of ``t``.  A
    generator joins by right products in breadth: the elements so far times the new generator, then every new
    element times every generator (a finite group is the closure of the identity under right products by its
    generators), products of whole arrays at once.  :meth:`add` refuses a generator that would take the group beyond
    ``limit`` elements and leaves it as it was.
    """

    def __init__(self, counts) -> None:
        self.counts = np.asarray(counts, dtype=np.int64).reshape(3)
        self.rotations = np.eye(3, dtype=np.int64)[None]
        self.translations = np.zeros((1, 3), dtype=np.int64)
        self.generators: list[tuple[np.ndarray, np.ndarray]] = []
        # the codes of the elements, ascending
        self.codes = self._codes(self.rotations, self.translations)

    def _codes(self, rotations, translations) -> np.ndarray:
        counts = self.counts
        digits = (np.asarray(rotations, dtype=np.int64).reshape(-1, 9) + 1) @ (3 ** np.arange(9, dtype=np.int64))
        shifts = np.mod(np.asarray(translations, dtype=np.int64).reshape(-1, 3), counts)
        rows = (shifts[:, 0] * counts[1] + shifts[:, 1]) * counts[2] + shifts[:, 2]
        return digits * int(np.prod(counts)) + rows

    @property
    def order(self) -> int:
        return int(self.rotations.shape[0])

    def __contains__(self, affine) -> bool:
        affine = np.asarray(affine, dtype=np.int64)
        code = self._codes(affine[None, :3, :3], affine[None, :3, 3])[0]
        position = int(np.searchsorted(self.codes, code))
        return position < self.codes.size and self.codes[position] == code

    def add(self, affine, limit: int | None = None) -> bool:
        """Join ``affine`` to the generators; ``False`` (and nothing joined) where the group would exceed ``limit``."""

        affine = np.asarray(affine, dtype=np.int64)
        generator = (affine[:3, :3].copy(), np.mod(affine[:3, 3], self.counts))
        generators = self.generators + [generator]
        codes = self.codes
        rotations, translations = [self.rotations], [self.translations]
        size = self.order
        # the elements so far times the new generator, then the new elements times every generator
        frontier, using = (self.rotations, self.translations), [generator]
        while True:
            fresh_rotations, fresh_translations = [], []
            for rotation, translation in using:
                products = frontier[0] @ rotation
                shifts = np.mod(frontier[0] @ translation + frontier[1], self.counts)
                found = self._codes(products, shifts)
                distinct, first = np.unique(found, return_index=True)
                keep = np.sort(first[~np.isin(distinct, codes, assume_unique=True)])
                if keep.size:
                    codes = np.union1d(codes, found[keep])
                    fresh_rotations.append(products[keep])
                    fresh_translations.append(shifts[keep])
                    size += int(keep.size)
                    if limit is not None and size > limit:
                        return False
            if not fresh_rotations:
                break
            frontier = (np.concatenate(fresh_rotations), np.concatenate(fresh_translations))
            rotations.append(frontier[0])
            translations.append(frontier[1])
            using = generators
        self.generators = generators
        self.codes = codes
        self.rotations = np.concatenate(rotations)
        self.translations = np.concatenate(translations)
        return True


#: The most operations of the group test 2 tests on a cell that :func:`_cell_group_images` follows (the operations about
#: the origin of a crystal of many cells whose symmetric site lies elsewhere join its pure translations, n^3 x 4 of
#: them in n x n x n diamond cubes): a candidate that would take it beyond is left out, one about the origin is tested
#: on its own instead.
_GUARD_GROUP_LIMIT = 1 << 15
#: Neighbours of every species in the environment of an atom (:func:`_environment_operations`; the classes are those of
#: ``Symmetry.axis_reflection._atom_environments``, whose ``_ENVIRONMENT_NEIGHBOURS`` this is).
_ENVIRONMENT_NEIGHBOURS = 12
#: The most atoms of one environment from which :func:`_environment_operations` takes its candidates.
_ENVIRONMENT_BUDGET = 32


def _environment_operations(grid, atoms, tolerance: float) -> list[np.ndarray] | None:
    """Affine ``[[R, t], [0, 1]]`` (``t`` in grid steps) of the operations ``r -> R r + t`` of a bulk orthorhombic
    cell that map the atoms onto themselves modulo the lattice and the grid onto itself: one ``t`` for every such
    signed permutation ``R`` and every pure translation, which generate them all; ``None`` where the rarest
    environment of an atom has more than :data:`_ENVIRONMENT_BUDGET` atoms (a crystal of many cells, whose cell
    group :class:`.kpoint_sectors.CellGroup` finds about its symmetric site).

    The cell group of the detection searches the translation of every ``R`` among those that carry the first atom of
    the smallest species onto an atom of it, and gives up after 32 candidates that pass its sampled atoms: under
    ``PARSEC_PERIODIC_DEFECT_CENTRE=off`` (before 2026-11-19) a defect off the origin of a cell larger than 2 x 2 x 2
    diamond cubes loses most of its operations there (the vacancy at ``(a/2, a/2, 0)`` of 3 x 3 x 3 cubes keeps the
    swap of x and y of its 24); under ``auto`` the search takes the candidates of this one where it gives up
    (:func:`..Symmetry.axis_reflection._seeded_translations`, the same classes of environments).  Here the candidates
    carry an atom ``x0`` of the rarest environment onto the atoms of its environment, which every symmetry must: the
    environment of an atom is the sorted distances to its :data:`_ENVIRONMENT_NEIGHBOURS` nearest atoms of every
    species (modulo the lattice), atoms of one species whose lists agree within four atom tolerances (an operation
    within the tolerance changes a distance by at most two); ``x0`` is chosen among the classes of those lists rounded
    to 1e-4 bohr (a vacancy of diamond: its four neighbours).  Each candidate is a whole number of grid steps; those
    that carry 16 sampled atoms of every species near atoms of it are matched with every atom (each within the
    tolerance of a different atom of its species, single-threaded k-d tree queries; two images nearest one atom go to
    :func:`..Symmetry.axis_reflection._atoms_are_invariant_operation`).
    """

    from scipy.spatial import cKDTree

    from ..Symmetry.axis_reflection import (
        _atom_environments,
        _atoms_are_invariant_operation,
        _folded,
        _integer_signed_permutation_parameters,
        _lattice_period,
        _positions_by_species,
        _rarest_class,
        _signed_permutation_operations,
    )

    period = _lattice_period(grid)
    if period is None or not atoms:
        return None
    species = _positions_by_species(atoms)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    steps = period / counts
    names = sorted(species)
    trees = {name: cKDTree(_folded(species[name], period), boxsize=period) for name in names}
    # x0: an atom of the smallest class of rounded environments (the first of the first species among the smallest;
    # the classes of the detection, Symmetry/axis_reflection.py, PARSEC_PERIODIC_DEFECT_CENTRE)
    name, origin, gap = _rarest_class(_atom_environments(species, period, trees))
    alike = np.flatnonzero(gap <= 4.0 * tolerance + 1.0e-9)
    if alike.size > _ENVIRONMENT_BUDGET:
        return None
    x0 = species[name][origin]
    targets = species[name][alike]
    limit = 1.0e-6 if tolerance <= 1.0e-7 else np.maximum(1.0e-6, 2.5 * tolerance / steps)
    radius = 2.0 * (tolerance * (1.0 + 1.0e-9) + 1.0e-12)
    samples = {
        other: species[other][np.unique(np.linspace(0, species[other].shape[0] - 1, 16).astype(np.int64))]
        for other in names
    }

    def sampled(matrix, shifts) -> np.ndarray:
        # whether the sampled atoms of every species land near an atom of it, for all candidates at once
        passing = np.ones(shifts.shape[0], dtype=bool)
        for other in names:
            images = (samples[other] @ matrix.T)[None, :, :] + (shifts * steps)[:, None, :]
            distances, _ = trees[other].query(_folded(images.reshape(-1, 3), period), k=1, distance_upper_bound=radius)
            passing &= np.all(np.isfinite(distances).reshape(shifts.shape[0], -1), axis=1)
        return passing

    def matches(matrix, shift) -> bool:
        # every atom within the tolerance of a partner of its species, the partners all different
        translation = shift * steps
        for other in names:
            images = species[other] @ matrix.T + translation
            distances, nearest = trees[other].query(_folded(images, period), k=1, distance_upper_bound=radius)
            if not np.all(distances <= tolerance):
                return False
            if np.unique(nearest).size != nearest.size:
                # two images nearest one atom: the matching of the detection decides
                return _atoms_are_invariant_operation(species, matrix, tolerance, period, translation)
        return True

    identity = np.eye(3, dtype=np.int64)
    found: list[np.ndarray] = []
    for operation in _signed_permutation_operations():
        matrix = np.asarray(operation, dtype=np.int64)
        if _integer_signed_permutation_parameters(grid, operation, 1.0e-10) is None:
            continue
        candidates = _folded(targets - x0 @ matrix.T, period) / steps
        rounded = np.rint(candidates)
        whole = np.all(np.abs(candidates - rounded) <= limit, axis=1)
        shifts = np.unique(np.mod(rounded[whole].astype(np.int64), counts), axis=0)
        pure = np.array_equal(matrix, identity)
        if pure:
            shifts = shifts[np.any(shifts != 0, axis=1)]
        if not shifts.shape[0]:
            continue
        for shift in shifts[sampled(matrix, shifts)]:
            if not matches(matrix, shift):
                continue
            affine = np.eye(4, dtype=np.int64)
            affine[:3, :3] = matrix
            affine[:3, 3] = shift
            found.append(affine)
            if not pure:
                # one translation per R: the others are this one after a pure translation
                break
    return found


def _cell_group_images(grid, atoms, origin_matrices, sectors=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(affine, rows)`` of the operations that test 2 tests on a bulk orthorhombic cell.

    The candidates, in this order: the elements of the sector group ``sectors`` (``r -> R r + t`` with their
    translations; the sectors keep the channel densities invariant under them by construction, so the generators
    after them are those beyond them, as the sign changes come first on a cluster); the operations of the atoms
    found from their environments (:func:`_environment_operations`: all of them where an environment is rare, a
    defect off the origin of any cell included), or where no environment is rare (a crystal of many cells) the
    operations of the cell group, whose commuting involutions the sectors take (:class:`.kpoint_sectors.CellGroup`:
    about the symmetric site of the crystal, with the half-cell translations of a cell that repeats a smaller one; the
    one the detection of the sectors built where it has these inputs,
    :func:`.kpoint_sectors.recent_cell_group_operations`, else built here);
    either at the atom tolerance of the detection (``PARSEC_SYMMETRY_ATOM_TOLERANCE``) or at
    :data:`parsec_python.Forces.symmetry.POSITION_TOLERANCE` (1e-6 bohr, that of the operations about the origin and
    of the guard of a cluster), whichever is smaller, the translations first, then the sign changes, then the others;
    and the signed permutations about the origin that map the atoms modulo the lattice (``origin_matrices``, those of
    before, to 1e-6 bohr).  Every one maps the atoms onto themselves modulo the lattice, so a channel
    density with the symmetry of the atoms passes; the elements of the sector group are among them, so the
    operations tested are not empty where the sectors are not trivial.  The elements of the sector group need no
    tolerance (the sectors keep the densities invariant under them, however far the atoms are off); an operation
    beyond them that holds the atoms only to a looser tolerance would measure the structure: the sectors average the
    Hamiltonian over their group alone, so the density breaks it by about as much as the atoms are off (a diamond
    vacancy of 2 x 2 x 2 cubes with its atoms 3e-4 bohr off and ``PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-3``: 0.028
    electrons, above the tolerance of the test).  A candidate in the group generated by those taken so far
    (:class:`_AffineClosure`) is skipped, and one that does not map the grid onto itself.  One that would take that
    group beyond :data:`_GUARD_GROUP_LIMIT` operations is left out, unless it is one of ``origin_matrices``: those
    are tested on their own instead (one per coset of those so tested), so that the group tested holds every one of
    them, as under ``PARSEC_SPIN_GUARD_CELL_GROUP=off`` (the inversion about a bond centre of a perfect diamond crystal
    of 6 x 6 x 6 cubes, whose cell group is Td about an atom with its half-cell translations: with it, the 41,472
    operations of the crystal).  ``rows`` are the grid rows of the images ``R r + t`` (``_grid_row_mapping_operation``,
    which wraps the cell), 32-bit below 2^31 points; ``affine`` is ``[[R, t], [0, 1]]`` with ``t`` in grid steps.
    """

    from parsec_python.Forces.symmetry import POSITION_TOLERANCE

    from ..Symmetry.axis_reflection import (
        _RowOrderTable,
        _fast_maps_requested,
        _grid_row_mapping_operation,
        abelian_sector_operations,
        symmetry_atom_tolerance,
    )
    from .kpoint_sectors import CellGroup, recent_cell_group_operations

    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    candidates: list[np.ndarray] = []
    if sectors is not None and int(getattr(sectors, "group_order", 1)) > 1:
        operations = abelian_sector_operations(sectors)
        shifts = getattr(sectors, "translations", None)
        shifts = (
            np.zeros((operations.shape[0], 3), dtype=np.int64)
            if shifts is None
            else np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
        )
        if shifts.shape[0] == operations.shape[0]:
            candidates += [_affine(matrix, shift) for matrix, shift in zip(operations, shifts)]

    def kind(affine) -> int:
        # the translations first, then the sign changes, then the others
        matrix = affine[:3, :3]
        return 0 if np.array_equal(matrix, np.eye(3)) else 1 if _is_diagonal(matrix) else 2

    # the operations beyond the sector group: those of the atoms to 1e-6 bohr at most (see above)
    tolerance = min(symmetry_atom_tolerance(), POSITION_TOLERANCE)
    try:
        found = _environment_operations(grid, atoms, tolerance)
    except ValueError:
        # (a grid the detection refuses: the operations about the origin below)
        found = None
    if found is not None:
        candidates += sorted(found, key=kind)
    else:
        # (where the environments found them all, the cell group is among them) the group the detection of the
        # sectors built for these very atoms and grid, else built here
        recent = recent_cell_group_operations(grid, atoms, atom_tolerance=tolerance)
        if recent is None:
            try:
                cell = CellGroup(grid, atoms, atom_tolerance=tolerance)
            except ValueError:
                cell = None
            recent = None if cell is None or cell.table is None else (cell.matrices, cell.shifts)
        if recent is not None:
            # (the identity first: left out)
            candidates += sorted((_affine(matrix, shift) for matrix, shift in list(zip(*recent))[1:]), key=kind)
    beyond = len(candidates)
    candidates += [
        _affine(matrix, np.zeros(3)) for matrix in sorted(origin_matrices, key=lambda matrix: not _is_diagonal(matrix))
    ]
    closure = _AffineClosure(counts)
    # the operations about the origin covered so far (in the closure or tested on their own): signed permutations
    # with no translation, at most 48
    about_origin = _AffineClosure(counts)
    table = _RowOrderTable(grid) if _fast_maps_requested() else None
    # the maps are kept for the run: 32-bit rows where the grid allows (half the memory of the maps of before)
    index_type = np.int32 if int(grid.size) < np.iinfo(np.int32).max else np.int64
    images: list[tuple[np.ndarray, np.ndarray]] = []
    for position, affine in enumerate(candidates):
        affine[:3, 3] = np.mod(affine[:3, 3], counts)
        origin = position >= beyond
        if affine in closure:
            if origin and affine not in about_origin:
                about_origin.add(affine)
            continue
        if origin and affine in about_origin:
            continue
        rows = _grid_row_mapping_operation(grid, affine[:3, :3], 1.0e-10, table=table, translation=affine[:3, 3])
        if rows is None:
            continue
        # an operation about the origin that the group would not hold within the limit is tested on its own (the
        # closure stays as it was): the group tested holds all those that the test of before tested
        if not closure.add(affine, limit=_GUARD_GROUP_LIMIT) and not origin:
            continue
        if origin:
            about_origin.add(affine)
        images.append((affine, np.asarray(rows).astype(index_type, copy=False)))
        del rows
    return images


def guard_group_order(grid, images) -> int:
    """The order of the group that the operations of test 2 (``images`` of :func:`atom_group_images`) generate on
    ``grid``: modulo the counts of a cell, whose operations are affine; signed permutations on a cluster."""

    if images and np.asarray(images[0][0]).shape == (4, 4):
        closure = _AffineClosure(grid.points_per_axis)
        for affine, _ in images:
            closure.add(affine)
        return closure.order
    elements = {_key(np.eye(3)): np.eye(3)}
    generators = [np.rint(np.asarray(matrix, dtype=np.float64)) for matrix, _ in images]
    grown = True
    while grown:
        grown = False
        for element in list(elements.values()):
            for generator in generators:
                product = element @ generator
                if _key(product) not in elements:
                    elements[_key(product)] = product
                    grown = True
    return len(elements)


def atom_group_images(grid, atoms, sectors=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(matrix, rows)`` of the operations that test 2 of the module needs on ``grid``.

    The signed permutations that map the atoms (by species) and the grid onto
    themselves (the group of the full-grid Hamiltonian), reduced to a set that
    generates that group: the sign changes first (the sectors usually keep the
    densities invariant under them by construction; they are tested all the
    same, at a few operations' cost), then the others.  The grid is asked for
    every candidate: a domain or a grid shift with less symmetry than the
    atoms (a cylinder along y, a box with a shorter y side) keeps a subgroup,
    whose generators need not be among those of the group of the atoms.  An
    operation in a coset ``f H`` of a refused ``f`` is not asked again.
    ``rows`` are the grid rows of the images ``matrix r``.

    A bulk orthorhombic cell (``PARSEC_SPIN_GUARD_CELL_GROUP=auto``) takes the
    operations ``r -> R r + t`` of :func:`_cell_group_images` instead, with
    ``matrix`` the affine ``[[R, t], [0, 1]]`` (``t`` in grid steps):
    ``sectors``, the reduction of the sector orbitals, gives its elements
    first.  ``off`` keeps the signed permutations about the origin of the cell
    (:func:`_periodic_atom_matrices`) in the loop below, as before.
    """

    from parsec_python.Forces.symmetry import atom_symmetry_operations

    def is_diagonal(matrix) -> bool:
        return np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0

    periodic = _periodic_atom_matrices(grid, atoms)
    if periodic is not None and spin_guard_cell_group_setting() == "auto":
        return _cell_group_images(grid, atoms, periodic, sectors)
    matrices = (
        [np.rint(np.asarray(operation.matrix, dtype=np.float64)) for operation in atom_symmetry_operations(atoms)]
        if periodic is None
        else periodic
    )
    # The sign changes first: the generators returned are those beyond their group.
    matrices.sort(key=lambda matrix: not is_diagonal(matrix))
    closure = {_key(np.eye(3)): np.eye(3)}
    generators: list[tuple[np.ndarray, np.ndarray]] = []
    refused: list[np.ndarray] = []
    for matrix in matrices:
        if _key(matrix) in closure:
            continue
        # ``matrix`` in ``f H`` for a refused ``f`` and the grid operations
        # ``H`` found so far does not map the grid either (else ``f`` would).
        if any(_key(other.T @ matrix) in closure for other in refused):
            continue
        rows = _grid_images(grid, matrix)
        if rows is None:
            refused.append(matrix)
            continue
        generators.append((matrix, rows))
        grown = True
        while grown:
            grown = False
            for element in list(closure.values()):
                for generator, _ in generators:
                    product = element @ generator
                    if _key(product) not in closure:
                        closure[_key(product)] = product
                        grown = True
    return generators


def _operation_text(matrix) -> str:
    """A signed permutation as its rows; an affine operation of a cell as ``R + t grid steps``."""

    matrix = np.rint(matrix).astype(int)
    if matrix.shape == (4, 4):
        return f"{matrix[:3, :3].tolist()} + {matrix[:3, 3].tolist()} grid steps"
    return f"{matrix.tolist()}"


def density_symmetry_breaking(densities, grid, atoms=None, tolerance: float = DENSITY_SYMMETRY_TOLERANCE, images=None):
    """:class:`FractionalOccupations` of channel densities that break the point group of ``atoms``, else ``None``.

    Each operation of :func:`atom_group_images` (``images``, else built from
    ``grid`` and ``atoms``) is tested by ``h^3 sum |rho(g r) - rho(r)|`` in
    electrons, per channel.
    """

    if images is None:
        images = atom_group_images(grid, atoms)
    largest = 0.0
    lines: list[str] = []
    for matrix, rows in images:
        broken = 0
        for index, density in enumerate(densities):
            values = np.asarray(density, dtype=np.float64)
            deviation = float(grid.volume_element * np.sum(np.abs(values[rows] - values)))
            largest = max(largest, deviation)
            if deviation >= tolerance:
                broken += 1
                lines.append(
                    f"{_CHANNEL_NAMES[index] if index < 2 else index} channel: the density differs by "
                    f"{deviation:.4f} electrons from its image under the operation "
                    f"{_operation_text(matrix)} of the atoms"
                )
        if broken == 2 and len(densities) == 2:
            # Both channels break it: an antiferromagnetic or broken-symmetry state where the operation
            # carries one channel onto the other (a diagnostic line; the test is the same).
            up, down = (np.asarray(density, dtype=np.float64) for density in densities)
            exchanged = float(grid.volume_element * np.sum(np.abs(up[rows] - down)))
            if exchanged < tolerance:
                lines.append(
                    f"the operation {_operation_text(matrix)} carries one channel onto the other "
                    f"(int |rho_up(g r) - rho_down(r)| = {exchanged:.2e} electrons): an antiferromagnetic or "
                    "broken-symmetry state, which two labels of the species with opposite "
                    "Initial_Spin_Polarization keep in the sectors (PARSEC_SPIN_FLIP_SYMMETRY=auto then solves "
                    "one channel)"
                )
    if largest < tolerance:
        return None
    return FractionalOccupations(
        largest=largest,
        tolerance=float(tolerance),
        lines=tuple(lines),
        reason="a channel density that breaks the point group of the atoms",
        measure="largest int |rho(g r) - rho(r)| (electrons)",
    )


def fractional_occupations(
    channels, tolerance: float = SPIN_SECTOR_OCCUPATION_TOLERANCE, *, unequal_partners: bool = False
):
    """:class:`FractionalOccupations` of the spin ``channels`` of a result, ``None`` where all are integer.

    ``channels`` are :class:`parsec_python.models.SpinChannelResult` (or
    anything with ``eigenvalues`` in Ry, ``occupations`` and
    ``representations``).  Only a fractional occupation that the sectors
    constrain counts: one in a multiplet whose levels lie in two or more
    sectors, exact (levels within :data:`MULTIPLET_WIDTH` of each other) or of
    :func:`sector_multiplets`.  A fractional single level, or a multiplet within
    one sector, is what the full grid has as well.  The line of a multiplet
    gives its levels (eV), occupations and sectors.  ``unequal_partners``
    (the weighted sectors, whose density is averaged over the point group):
    an exact multiplet whose levels hold unequal occupations counts by
    ``min(f, 1 - f)`` of their mean as well.
    """

    largest = 0.0
    lines: list[str] = []
    for index, channel in enumerate(channels):
        occupations = np.asarray(channel.occupations, dtype=np.float64)
        if not occupations.size:
            continue
        values = np.asarray(channel.eigenvalues, dtype=np.float64)
        sectors = np.asarray(getattr(channel, "representations", np.zeros(values.size)), dtype=np.int64)
        fractional = np.minimum(occupations, 1.0 - occupations)
        groups = [
            group for group in _multiplets(values, MULTIPLET_WIDTH) if np.unique(sectors[group]).size >= 2
        ]
        exact = {tuple(group) for group in groups}
        groups += [group for group in sector_multiplets(values, sectors) if tuple(group) not in exact]
        for group in groups:
            measure = float(np.max(fractional[group]))
            if unequal_partners and tuple(group) in exact and float(np.ptp(occupations[group])) > 0.0:
                # (degenerate partners filled unequally, one whole and one empty: their mean is fractional)
                mean = float(np.mean(occupations[group]))
                measure = max(measure, min(mean, 1.0 - mean))
            if measure < tolerance:
                continue
            largest = max(largest, measure)
            levels = " ".join(f"{values[member] * _RYDBERG_TO_EV:.4f}" for member in group)
            filled = " ".join(f"{occupations[member]:.4f}" for member in group)
            labels = " ".join(_sector_label(sectors[member]) for member in group)
            lines.append(
                f"{_CHANNEL_NAMES[index] if index < 2 else index} channel: {len(group)} level(s) at {levels} eV "
                f"hold {float(np.sum(occupations[group])):.4f} electrons (occupations {filled}; sectors {labels})"
            )
    if largest < tolerance:
        return None
    return FractionalOccupations(largest=largest, tolerance=float(tolerance), lines=tuple(lines))


class _ChannelPair:
    """The two channels' orbit values as one vector, for a mixer of :mod:`.symmetry_fields`.

    The interface :class:`~.symmetry_fields.SymmetryAndersonMixer` and
    :class:`~.symmetry_fields.SymmetryWeightedMixer` read of a reducer:
    multiplicity weights of both halves, and plain arrays in and out.
    """

    def __init__(self, reducer: SymmetrySCFReducer) -> None:
        reduction = reducer.reduction
        self.size = 2 * int(reduction.wedge_size)
        weights = np.concatenate((reducer.weights, reducer.weights))
        weights.setflags(write=False)
        self.weights = weights
        self.reduction = SimpleNamespace(
            multiplicities=np.concatenate((reduction.multiplicities, reduction.multiplicities)),
            wedge_size=self.size,
        )

    def wedge_values(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.size,):
            raise ValueError("the two channels' orbit values do not match the wedge")
        return array

    def field(self, values) -> np.ndarray:
        return self.wedge_values(values)

    def expand_values(self, values) -> np.ndarray:
        return self.wedge_values(values)


class SpinSectorReducer:
    """The scalar-field adapter of a spin-polarized loop in the symmetry sectors.

    It is a :class:`~.symmetry_fields.SymmetrySCFReducer` (``reducer``) for the
    fields of one channel, and adds what
    :func:`parsec_python.SCF.spin_polarized.run_spin_scf` asks of a reduced
    loop: the two channels joined for the mixer (up first, as ``spinmix.f90``)
    and split again, PARSEC's residual norms per channel and the spin energy,
    each with the orbit multiplicities as quadrature weights, which gives the
    full-grid sums of invariant fields, and the symmetry guard (see the
    module).  ``hybrid`` says whether the run is a hybrid's Fock loop, which
    names the loop of a failed check.  ``guard`` false: no test (the one
    sector of the identity of a fallback's shared full grid, which holds
    every state of the full grid).
    """

    spin_channels_supported = True

    def __init__(
        self,
        reducer: SymmetrySCFReducer,
        *,
        hybrid: bool,
        tolerance: float = SPIN_SECTOR_OCCUPATION_TOLERANCE,
        fixed_occupations: bool = False,
        structure=None,
        guard: bool = True,
    ) -> None:
        self.reducer = reducer
        # False for one sector of the identity (the shared full grid of a fallback): nothing to test
        self.guard = bool(guard)
        self.hybrid = bool(hybrid)
        self.tolerance = float(tolerance)
        # Occupations of occup.in (no Fermi level): test 3 does not apply.
        self.fixed_occupations = bool(fixed_occupations)
        # ``structure()`` -> ``(grid, atoms)`` of the present structure, for test 2.
        self.structure = structure
        self._pair = _ChannelPair(reducer)
        # ``(iteration, asymmetry, criterion)`` of each test-2 measure at the convergence of a loop.
        self.symmetry_measures: list[tuple[int, float | None, float]] = []

    # ----- one channel: the SymmetrySCFReducer interface ------------------

    @property
    def reduction(self):
        return self.reducer.reduction

    def field(self, values):
        return self.reducer.field(values)

    def from_full(self, values):
        return self.reducer.from_full(values)

    def to_full(self, values) -> np.ndarray:
        return self.reducer.to_full(values)

    def wedge_values(self, values) -> np.ndarray:
        return self.reducer.wedge_values(values)

    def expand_values(self, values) -> np.ndarray:
        return self.reducer.expand_values(values)

    def minimum(self, values) -> float:
        return self.reducer.minimum(values)

    def maximum(self, values) -> float:
        return self.reducer.maximum(values)

    def weighted_dot(self, left, right) -> float:
        return self.reducer.weighted_dot(left, right)

    # ----- two channels ---------------------------------------------------

    def join_channels(self, fields) -> np.ndarray:
        """The orbit values of both channels, up first."""

        up, down = fields
        return np.concatenate((self.reducer.wedge_values(up), self.reducer.wedge_values(down)))

    def split_channels(self, values) -> list:
        """The two channel fields of a joined vector."""

        values = self._pair.wedge_values(values)
        half = self._pair.size // 2
        return [self.reducer.field(values[:half].copy()), self.reducer.field(values[half:].copy())]

    @property
    def weights(self) -> np.ndarray:
        return self.reducer.weights

    def join_blocks(self, fields) -> np.ndarray:
        """Several orbit fields end to end (a meta-GGA's ``V_up, V_down, vtau_up, vtau_down``)."""

        return self.reducer.join_blocks(fields)

    def split_blocks(self, values, count: int) -> list:
        return self.reducer.split_blocks(values, count)

    def block_mixer(self, settings, count: int):
        return self.reducer.block_mixer(settings, count)

    def spin_mixer(self, settings):
        """The mixer of ``settings.method`` for joined channel vectors (:meth:`join_channels`)."""

        if settings.method == "anderson":
            return SymmetryAndersonMixer(self._pair, settings)
        return SymmetryWeightedMixer(self._pair, make_mixer(settings, weights=self._pair.weights))

    def spin_potential_residual_metrics(
        self, input_potentials, output_potentials, densities, volume_element: float, electrons
    ) -> SpinResidualMetrics:
        """:func:`parsec_python.Mixer.spin_potential_residual_metrics` on the orbits."""

        if volume_element <= 0:
            raise ValueError("volume element must be positive")
        weights = self.reducer.weights
        weighted: list[float] = []
        plain: list[float] = []
        for input_potential, output_potential, density, count in zip(
            input_potentials, output_potentials, densities, electrons
        ):
            residual = self.reducer.wedge_values(output_potential) - self.reducer.wedge_values(input_potential)
            if count < EMPTY_CHANNEL_ELECTRONS:
                plain.append(0.0)
                weighted.append(0.0)
                continue
            squared = residual * residual
            plain.append(float(np.sqrt(volume_element * np.dot(weights, squared))))
            density_values = self.reducer.wedge_values(density)
            weighted.append(
                float(np.sqrt(volume_element * np.dot(weights * density_values, squared) / count))
            )
        return SpinResidualMetrics(
            weighted_by_spin=(weighted[0], weighted[1]),
            plain_by_spin=(plain[0], plain[1]),
        )

    def spin_total_energy(
        self,
        eigenvalues,
        occupations,
        densities,
        input_effective_potentials,
        ionic_potential,
        output_hartree_potential,
        output_xc_potentials,
        exchange_correlation_energy: float,
        ion_ion_energy: float,
        volume_element: float,
        hubbard_energy: float | None = None,
        alpha_z_energy: float | None = None,
    ):
        """:func:`parsec_python.Energy.spin_total_energy` with multiplicity-weighted orbit sums."""

        wedge = self.reducer.wedge_values
        weights = self.reducer.weights
        ionic = wedge(ionic_potential)
        band_energy = 0.0
        old_hxc_integral = 0.0
        vxc_integral = 0.0
        for values, filled, density, input_potential, xc_potential in zip(
            eigenvalues, occupations, densities, input_effective_potentials, output_xc_potentials
        ):
            values = np.asarray(values, dtype=float)
            filled = np.asarray(filled, dtype=float)
            if filled.shape != values.shape:
                raise ValueError("eigenvalues and occupations must have the same shape")
            weighted_density = weights * wedge(density)
            band_energy += float(np.dot(filled, values))
            old_hxc_integral += float(volume_element * np.dot(weighted_density, wedge(input_potential) - ionic))
            vxc_integral += float(volume_element * np.dot(weighted_density, wedge(xc_potential)))
        if alpha_z_energy is not None:
            band_energy += float(alpha_z_energy)
        total_density = weights * (wedge(densities[0]) + wedge(densities[1]))
        hartree_integral = float(volume_element * np.dot(total_density, wedge(output_hartree_potential)))
        electron_ion = float(volume_element * np.dot(total_density, ionic))
        electronic = float(band_energy - old_hxc_integral + 0.5 * hartree_integral + exchange_correlation_energy)
        return _breakdown(
            band_energy,
            hartree_integral,
            vxc_integral,
            exchange_correlation_energy,
            electron_ion,
            ion_ion_energy,
            electronic,
            hubbard_energy,
        )

    def spin_hooks(self) -> dict:
        """The hooks of :func:`parsec_python.SCF.spin_polarized.run_spin_scf` for this adapter."""

        return {
            "mixer_factory": self.spin_mixer,
            "residual_metrics_evaluator": self.spin_potential_residual_metrics,
            "total_energy_evaluator": self.spin_total_energy,
            "scalar_field_adapter": self,
        }

    # ----- the guard ------------------------------------------------------

    def _stage(self, final: bool) -> str:
        if final and self.hybrid:
            return "the end of the Fock loop"
        if self.hybrid:
            return "the semilocal start of the Fock loop"
        return "the SCF loop"

    #: Further solves of the start Hamiltonian that test 1 may ask for (see :meth:`start_spectrum_report`).
    start_spectrum_passes = START_SPECTRUM_PASSES
    #: Largest change (Ry) of the levels up to the Fermi level between two such solves that counts as converged.
    start_level_change = START_LEVEL_CHANGE

    def start_spectrum_report(self, eigenvalues, occupations, representations):
        """:func:`partly_filled_multiplets` of the start Hamiltonian's levels (test 1), ``None`` where there are none.

        :func:`parsec_python.SCF.spin_polarized.run_spin_scf` asks after the
        occupations of the first iteration of a loop that starts from the
        atomic densities (one array per channel each).  A report is only a
        candidate until the levels are converged: the loop solves the start
        Hamiltonian again, from its subspaces, up to
        :attr:`start_spectrum_passes` times, and calls :meth:`abandon_start`
        when the report stays and the levels up to the Fermi level moved by at
        most :attr:`start_level_change` in the last solve.  The levels of the
        first filter of a large cluster are not converged near its top (the
        nanodiamond C459H204: seven levels within 0.015 Ry around a first
        Fermi level of +8 eV), and a report on them disappears or keeps moving.
        """

        if not self.guard:
            return None
        if self.sector_weights is not None:
            # Weighted sectors: every partner of a weighted state as a level of its own (see the module).
            expanded = [
                partner_levels(values, filled, sectors, self._weights_of(sectors))
                for values, filled, sectors in zip(eigenvalues, occupations, representations)
            ]
            eigenvalues = [item[0] for item in expanded]
            occupations = [item[1] for item in expanded]
            representations = [item[2] for item in expanded]
        return partly_filled_multiplets(eigenvalues, occupations, representations, fixed=self.fixed_occupations)

    @property
    def sector_weights(self):
        """The weights of the sectors of a subgroup with weights (``PointGroupReduction``), ``None`` otherwise."""

        weights = getattr(self.reducer.reduction, "sector_weights", None)
        return None if weights is None else np.asarray(weights, dtype=np.int64)

    def _weights_of(self, sectors) -> np.ndarray:
        """The weight of every state of the sector labels (from one) of a channel."""

        return self.sector_weights[np.asarray(sectors, dtype=np.int64) - 1]

    def _partner_channel(self, channel):
        """A result channel of weighted sectors with partner levels in distinct sectors (:func:`partner_levels`)."""

        solved = getattr(channel, "sector_states", None)
        if solved is None:
            return channel
        sectors, _, weights = solved
        repeats = np.asarray(weights, dtype=np.int64)
        partners = np.arange(int(repeats.sum()), dtype=np.int64) - np.repeat(np.cumsum(repeats) - repeats, repeats)
        return SimpleNamespace(
            eigenvalues=np.asarray(channel.eigenvalues, dtype=np.float64),
            occupations=np.asarray(channel.occupations, dtype=np.float64),
            representations=np.repeat(np.asarray(sectors, dtype=np.int64), repeats) * PARTNER_LABEL_STRIDE + partners,
        )

    def abandon_start(self, report, passes: int = 0) -> None:
        """Raise :class:`SpinSectorsAbandoned` for a converged report of :meth:`start_spectrum_report`."""

        if passes:
            report = FractionalOccupations(
                largest=report.largest,
                tolerance=report.tolerance,
                lines=report.lines + (f"(the levels of the start Hamiltonian after {passes} further solve(s))",),
                reason=report.reason,
                measure=report.measure,
            )
        raise SpinSectorsAbandoned("the start Hamiltonian", report)

    def check_start_spectrum(self, eigenvalues, occupations, representations) -> None:
        """Raise :class:`SpinSectorsAbandoned` where the levels given have a partly filled multiplet (test 1)."""

        report = self.start_spectrum_report(eigenvalues, occupations, representations)
        if report is not None:
            self.abandon_start(report)

    #: Test 2 (see the module): tolerance, the bound above which no tightening is tried, and the tightenings.
    density_symmetry_tolerance = DENSITY_SYMMETRY_TOLERANCE
    density_symmetry_noise_bound = DENSITY_SYMMETRY_NOISE_BOUND
    density_symmetry_tightenings = DENSITY_SYMMETRY_TIGHTENINGS

    def _grid_and_images(self):
        """The grid of the present structure and the operations of test 2 on it (built once per structure)."""

        grid, atoms = self.structure()
        key = (id(grid), tuple(tuple(np.asarray(atom.position, dtype=float).tolist()) for atom in atoms))
        if getattr(self, "_images_key", None) != key:
            # the sector group: its elements come first among the operations of a cell (a cluster takes none of it)
            self._images = atom_group_images(grid, atoms, sectors=getattr(self.reducer, "reduction", None))
            self._images_key = key
        return grid, self._images

    def density_symmetry_measure(self, densities) -> float | None:
        """The largest ``int |rho(g r) - rho(r)|`` (electrons) of the full-grid channel ``densities`` (test 2).

        ``None`` without a structure.  :func:`parsec_python.SCF.spin_polarized.run_spin_scf`
        asks at the convergence of a loop without an exchange operator and
        tightens the loop's criterion while the measure is above the
        tolerance but below :attr:`density_symmetry_noise_bound` and shrinks
        (see the module).
        """

        if self.structure is None or not self.guard:
            return None
        grid, images = self._grid_and_images()
        largest = 0.0
        for _, rows in images:
            for density in densities:
                values = np.asarray(density, dtype=np.float64)
                largest = max(largest, float(grid.volume_element * np.sum(np.abs(values[rows] - values))))
        return largest

    def check_spin_result(self, result, *, final: bool = False) -> None:
        """Raise :class:`SpinSectorsAbandoned` where ``result`` fails test 2 or 3 of the module.

        ``final``: the check of the whole run (for a hybrid, the state at the
        end of the Fock loop); otherwise that of one SCF loop without an
        exchange operator.
        """

        result = getattr(result, "result", result)
        channels = getattr(result, "spin", None)
        if channels is None or not self.guard:
            return
        if self.structure is not None:
            grid, images = self._grid_and_images()
            report = density_symmetry_breaking(
                [channel.density for channel in channels], grid, tolerance=DENSITY_SYMMETRY_TOLERANCE, images=images
            )
            if report is not None:
                raise SpinSectorsAbandoned(self._stage(final), report)
        if self.fixed_occupations:
            return
        weighted = self.sector_weights is not None
        if weighted:
            channels = [self._partner_channel(channel) for channel in channels]
        report = fractional_occupations(channels, self.tolerance, unequal_partners=weighted)
        if report is not None:
            raise SpinSectorsAbandoned(self._stage(final), report)


class HostSpinXCOnOrbits:
    """A host spin functional for the orbit fields of a sector loop.

    ``evaluate(up, down)`` is a full-grid spin evaluator
    (:meth:`PreparedSinglePointSystem.evaluate_spin_xc` of the present
    structure); both channels are expanded to the full grid, evaluated, and
    the potentials and the energy density reduced to one value per orbit,
    an exact representation change for invariant fields.
    """

    def __init__(self, evaluate, reducer: SymmetrySCFReducer, label: str) -> None:
        self.evaluate = evaluate
        self.reducer = reducer
        self.implementation_label = f"{label} on the full grid, from and to the symmetry orbits"

    def __call__(self, valence_up, valence_down) -> SpinXCResult:
        evaluated = self.evaluate(self.reducer.to_full(valence_up), self.reducer.to_full(valence_down))
        return SpinXCResult(
            potential_up=self.reducer.from_full(evaluated.potential_up),
            potential_down=self.reducer.from_full(evaluated.potential_down),
            energy_density=self.reducer.from_full(evaluated.energy_density),
            total_energy=evaluated.total_energy,
        )


__all__ = [
    "DENSITY_SYMMETRY_TOLERANCE",
    "FractionalOccupations",
    "START_MULTIPLET_WIDTH",
    "atom_group_images",
    "density_symmetry_breaking",
    "guard_group_order",
    "spin_fallback_shared_setting",
    "spin_guard_cell_group_setting",
    "partly_filled_multiplets",
    "partner_levels",
    "HostSpinXCOnOrbits",
    "MULTIPLET_WIDTH",
    "SPIN_SECTOR_OCCUPATION_TOLERANCE",
    "SpinSectorReducer",
    "SpinSectorsAbandoned",
    "fractional_occupations",
]
