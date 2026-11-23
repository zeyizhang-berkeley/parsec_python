"""Spin-flip (magnetic) symmetry of a collinear spin-polarized run (``PARSEC_SPIN_FLIP_SYMMETRY``).

PARSEC 1.5 starts an antiferromagnetic or broken-symmetry state from two labels of one element with opposite
``Initial_Spin_Polarization`` (``Fe_a`` at ``+p``, ``Fe_b`` at ``-p``): ``initchrg.f90`` gives the up channel
``(1 + p)`` and the down channel ``(1 - p)`` times the atomic densities, species by species.  The point group of the
symmetry sectors (:mod:`..Symmetry.axis_reflection`) and the guard of :mod:`.spin_sectors` key on the label, so they
keep the operations that map every label onto itself (``H``).  An operation ``g`` of the structure that maps every
atom of a label onto an atom of its partner label (the label of the same species with the opposite polarization)
maps the up start density onto the down one: ``rho_down(r) = rho_up(g^-1 r)``.  Every term of the Kohn-Sham
potential of a collinear functional is then covariant (the Hartree potential of the total density, which ``g``
leaves invariant, ``V_xc,down[rho_up, rho_down](r) = V_xc,up[rho_down, rho_up]`` at the image point, the ionic and
nonlocal terms of identical species), so ``H_down = U_g H_up U_g^-1``: the down channel has the levels of the up
channel and the images of its orbitals, at every SCF iteration from such a start.  With
``PARSEC_SPIN_FLIP_SYMMETRY=auto`` the spin loop (:func:`parsec_python.SCF.spin_polarized.run_spin_scf`) solves the
up channel alone, in the sectors of ``H`` where the run has them, and takes the down channel from it: its levels
and occupations are those of the up channel, its density the image of the up density (one gather of the orbit
values per iteration), its orbitals the images of the up orbitals (:class:`FlippedOrbitals`, formed only where they
are read) and its sector labels the images of the up labels (:attr:`SpinFlipOperation.sector_map`).  The
nonlocal forces of the down channel are those of the up channel carried by ``g``
(:meth:`SpinFlipOperation.down_forces`).  The eigensolve of a step is halved; so are the subspaces the
sectors keep.

What a run must have for the route (:func:`detect_spin_flip`; otherwise both channels are solved as before and the
backend details say why): a partner for every label with atoms (a label at ``p = 0`` is its own partner; at least
one label starts polarized; species identical in everything else, the pseudopotential file by its contents), an
even electron count (an odd one has no state with equal channels, and the flip would leave half an electron in both
channels' top level), one operation that maps the atoms so and the grid onto itself (on a cluster a signed
permutation of the axes about the origin; on a bulk orthorhombic cell ``r -> R r + t`` with ``t`` a whole number of
grid steps, a pure translation first), that carries the orbits of the sector group onto orbits and every sector onto
a sector, that maps the local ionic potential onto itself within :data:`IONIC_POTENTIAL_TOLERANCE` (relative: point
charges of ``Add_Point_Charges`` or another external term that breaks the operation leave it to the next operation,
or to both channels), and that maps the start density of the up channel onto that of the down channel within
:data:`START_DENSITY_TOLERANCE` electrons.  The spin loop keeps the route only without an exchange operator
(hybrids), a meta-GGA, LDA+U, ``Symmetrize_Charge_Density`` and with the occupations of ``occup.in`` equal in
both channels; it checks the operation against the atoms and the local ionic potential of every loop it runs (a
relaxation that lost it, and the field loops of ``Polarizability``, whose field breaks it, solve both channels),
and a warm start (``Restart_Run``, the structure before) keeps it only where its down potential is the image of its
up potential within :data:`WARM_START_TOLERANCE` Ry.  A loop that leaves the route for one of these reasons says
so in ``parsec.out`` (a :class:`parsec_python.models.SpinSectorFallbackRecord` of ``route`` ``spin_flip``).

The route constrains the state to the magnetic group, as the sectors constrain it to their point group: a lower
state that breaks the flip (a ferrimagnetic one, or one that the up and down channels reach on different paths
from noise) is not found, while a full-grid run from the same flip-symmetric start reaches it only through noise.
From a flip-symmetric start two solved channels keep the levels of each other up to that noise; it can grow into a
moment where a level of the up channel and its image in the down channel are partly filled (Fermi-Dirac
occupations: the symmetric ensemble of a partly filled pair, which unequal fillings lower).  So the spin loop
(:func:`parsec_python.SCF.spin_polarized.run_spin_scf`) leaves the route where its up channel ends with a fractional
occupation (``min(f, 1 - f)`` from ``SCF.spin_polarized.SPIN_FLIP_OCCUPATION_TOLERANCE`` = 0.05, as test 3 of
:mod:`.spin_sectors` keeps the sectors), and where a test of the spin sectors fails in a flip loop (equal levels of
the two channels are those of the operation there, not those two solved channels hold: test 1 then sees a partly
filled multiplet of the start Hamiltonian that ``off`` does not see).  It says so in ``parsec.out``, solves both
channels for the rest of the run (later loops, ionic steps, the systems that the spin guard prepares again), and
solves the loop again from its start in the sectors it started in: the route of ``off``.
Not detected, and the reason the route is not the default: a flip loop that settles with integer occupations in
another minimum than two solved channels reach (A100, PBE, 80 K, against ``off``: Ni2 at 4.06 bohr from
``+0.6 / -0.6``, -138.78451344 against -138.84084192 Ry, 0.77 eV higher, no note; Fe2, Co2, Cr2, Cr4 and MnO
agree with ``off``), and a closed shell that a rotation across its gap would lower, as in the sectors.
``PARSEC_SPIN_FLIP_STABILITY=auto`` (:func:`spin_flip_stability_setting`, opt-in) tests the converged state of every
loop against perturbations that break the operation (the eigenvalues of the linear SCF response of the down channel,
:func:`parsec_python.SCF.spin_polarized._flip_stability`) and leaves the route where one reaches 1.  It does not see
the Ni2 miss: the state of the route is a local minimum there (largest eigenvalue 0.61, 0.86 at 300 K; Cr2 0.95, Cr4
0.36, Co2 0.44, Fe2 0.48, MnO 0.31),
``off`` reaches its own state along its own path (the down channel solved apart from the first iteration on; it ends
unconverged after 150 iterations, the levels of its channels 5.6e-4 Ry apart), and a ferromagnetic state with a
moment of 4 lies below both (-138.85078776 Ry, reached from the state of the route with both channels solved from a
first down solve of its own).  No test of the converged state can make the route match ``off`` there.
A same-label start (one ``Initial_Spin_Polarization`` per species) has no flip operation; a state that breaks the
point group of such atoms fires test 2 of :mod:`.spin_sectors` as before.  Units Ry, bohr.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
import hashlib
import os
from pathlib import Path

import numpy as np

__all__ = [
    "FlippedOrbitals",
    "IONIC_POTENTIAL_TOLERANCE",
    "SpinFlipChannelState",
    "SpinFlipOperation",
    "START_DENSITY_TOLERANCE",
    "WARM_START_TOLERANCE",
    "detect_spin_flip",
    "spin_flip_setting",
    "spin_flip_stability_setting",
    "spin_partner_labels",
]

_SWITCH = "PARSEC_SPIN_FLIP_SYMMETRY"
#: Electrons by which the image of the up start density may differ from the down start density.
START_DENSITY_TOLERANCE = 1.0e-6
#: Distance (bohr) within which the operation must carry every atom onto an atom of its partner label (at most;
#: ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` below it is taken).
ATOM_TOLERANCE = 1.0e-6
#: The operation must map the local ionic potential onto itself within this fraction of ``max(1, max |V_ion|)`` (Ry):
#: the rounding of the sums of the atoms' terms, not an external term (point charges, the field of a polarizability
#: loop) and not partner atoms placed symmetrically only within the atom tolerance.
IONIC_POTENTIAL_TOLERANCE = 1.0e-10
#: Ry within which the down potential of a warm start must be the image of its up potential to keep the route: a
#: warm start of a loop that took the route is one to the rounding, a restart of a broken state is not.
WARM_START_TOLERANCE = 1.0e-8
#: Atoms (spread over the list) whose images a candidate operation must find first, before the full map is formed
#: (a wrong candidate fails about every second probe of random labels, so more probes cost little).
PROBE_ATOMS = 32


def _field_values(values) -> np.ndarray:
    """The host values of a scalar field: the orbit values of a :class:`.symmetry_fields.SymmetryScalarField`."""

    inner = getattr(values, "values", None)
    if inner is not None and hasattr(values, "reduction"):
        return np.asarray(inner, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


#: The value where none is named: ``off`` (a flip loop can settle in another state than two solved channels with
#: integer occupations, Ni2 0.77 eV higher, which no test of the loop sees; see the module).
DEFAULT = "off"


def spin_flip_setting() -> str:
    """``PARSEC_SPIN_FLIP_SYMMETRY``: ``off`` (the default) or ``auto`` (on request).

    ``auto`` solves only the up channel of a collinear spin-polarized run whose labels and start have a spin-flip
    operation (see the module) and takes the down channel from it, leaving the route where the up channel ends with a
    fractional occupation or a test of the spin sectors fails in a flip loop; ``off`` solves both channels.
    """

    value = os.environ.get(_SWITCH, DEFAULT).strip().lower() or DEFAULT
    if value in {"0", "false", "no"}:
        value = "off"
    if value in {"1", "true", "yes", "on"}:
        value = "auto"
    if value not in ("off", "auto"):
        raise ValueError(f"{_SWITCH} must be off or auto, not {value!r}")
    return value


_STABILITY_SWITCH = "PARSEC_SPIN_FLIP_STABILITY"


#: The value of ``PARSEC_SPIN_FLIP_STABILITY`` where none is named: ``off`` (the test found no unstable state in six
#: antiferromagnetic inputs and costs about what the route saves on them; it does not see the Ni2 miss, a stable
#: state of the route: see the module).
STABILITY_DEFAULT = "off"


def spin_flip_stability_setting() -> str:
    """``PARSEC_SPIN_FLIP_STABILITY``: ``off`` (the default) or ``auto`` (on request); used where the flip route is
    taken, read and checked in every preparation.

    ``auto`` tests the state of every converged loop of ``PARSEC_SPIN_FLIP_SYMMETRY=auto`` against perturbations that
    break the operation (the linear response of the down channel, which the route does not solve otherwise:
    ``SCF.spin_polarized._flip_stability``), notes the largest eigenvalue of that response and leaves the route where
    it reaches 1 (an unstable state); ``off`` keeps the state untested, bit for bit as before 2026-11-22.
    """

    value = os.environ.get(_STABILITY_SWITCH, STABILITY_DEFAULT).strip().lower() or STABILITY_DEFAULT
    if value in {"0", "false", "no"}:
        value = "off"
    if value in {"1", "true", "yes", "on"}:
        value = "auto"
    if value not in ("off", "auto"):
        raise ValueError(f"{_STABILITY_SWITCH} must be off or auto, not {value!r}")
    return value


# ----- labels -------------------------------------------------------------


def _file_key(path) -> str:
    """The contents of a pseudopotential file (its SHA-256), else its path."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return f"path:{path}"


def _species_key(specification) -> object:
    """Everything of a species but its polarization and its diagnostic radius, the file by its contents."""

    try:
        names = [item.name for item in fields(specification)]
    except TypeError:
        return repr(specification)
    key = []
    for name in names:
        if name in ("initial_spin_polarization", "atomic_radius"):
            continue
        value = getattr(specification, name)
        key.append((name, _file_key(value) if name == "path" else repr(value)))
    return tuple(key)


def spin_partner_labels(atoms, species) -> tuple[dict[str, str] | None, str]:
    """``({label: partner}, "")`` of the labels with atoms, or ``(None, reason)``.

    The partner of a label at polarization ``p != 0`` is the one other label with atoms of the same species
    (:func:`_species_key`), polarization ``-p`` and as many atoms; a label at ``p = 0`` is its own partner.  At
    least one label must start polarized.
    """

    labels = list(dict.fromkeys(atom.symbol for atom in atoms))
    counts = Counter(atom.symbol for atom in atoms)
    missing = [label for label in labels if label not in species]
    if missing:
        return None, f"no species for the label(s) {', '.join(missing)}"
    polarization = {label: float(getattr(species[label], "initial_spin_polarization", 0.0)) for label in labels}
    keys = {label: _species_key(species[label]) for label in labels}
    partners: dict[str, str] = {}
    for label in labels:
        value = polarization[label]
        if value == 0.0:
            partners[label] = label
            continue
        matches = [
            other
            for other in labels
            if other != label
            and keys[other] == keys[label]
            and polarization[other] == -value
            and counts[other] == counts[label]
        ]
        if len(matches) != 1:
            return None, (
                f"the label {label} (Initial_Spin_Polarization {value:g}) has "
                + ("no" if not matches else "more than one")
                + " partner label of the same species at the opposite polarization with as many atoms"
            )
        partners[label] = matches[0]
    if all(partners[label] == label for label in labels):
        return None, "no label starts spin-polarized (every Initial_Spin_Polarization is 0)"
    return partners, ""


# ----- the operation ----------------------------------------------------


@dataclass(frozen=True)
class SpinFlipChannelState:
    """The eigensolver state of a down channel that was not solved: the images of the up channel's states."""

    operation: "SpinFlipOperation"
    #: Read by the spin loop, which gives such a channel no subspace of its own in a warm start.
    spin_flip_channel = True


@dataclass(frozen=True, eq=False)
class SpinFlipOperation:
    """``r -> R r + t`` that carries the up channel of a flip-symmetric start onto the down channel.

    ``matrix`` is ``R`` (a signed permutation), ``translation`` ``t`` in grid steps (zero on a cluster).
    ``rows[i]`` is the grid row of ``g r_i`` and ``inverse_rows`` that of ``g^-1 r_i``; ``atom_image[a]`` the atom at
    ``g R_a``.  ``orbit_map`` (with a sector reduction of the fields) gives the orbit values of the down channel as
    those of the up channel at ``orbit_map``; ``sector_map[chi]`` (with sectors) the sector of the image of a state of
    sector ``chi`` (0-based).  ``partners`` maps every label to its partner.
    """

    matrix: np.ndarray
    translation: np.ndarray
    rows: np.ndarray
    inverse_rows: np.ndarray
    atom_image: np.ndarray
    partners: dict
    orbit_map: np.ndarray | None = None
    sector_map: np.ndarray | None = None
    tolerance: float = ATOM_TOLERANCE
    period: np.ndarray | None = None
    steps: np.ndarray | None = None
    #: ``PARSEC_SPIN_FLIP_STABILITY`` of the run (:func:`spin_flip_stability_setting`): ``auto`` tests the state of
    #: every converged loop of the route (``SCF.spin_polarized``), ``off`` keeps it untested.
    stability: str = "off"
    _device_rows: dict = field(default_factory=dict, repr=False, compare=False)
    _inverse_orbit_map: list = field(default_factory=list, repr=False, compare=False)

    @property
    def description(self) -> str:
        matrix = np.rint(self.matrix).astype(int).tolist()
        if self.period is None or not np.any(self.translation):
            return f"{matrix}"
        return f"{matrix} + {np.asarray(self.translation, dtype=int).tolist()} grid steps"

    # ----- fields

    def down_field(self, values):
        """The down-channel field of the up-channel field ``values`` (orbit values or a full-grid array)."""

        from .symmetry_fields import SymmetryScalarField

        if isinstance(values, SymmetryScalarField):
            if self.orbit_map is None:
                raise ValueError("this spin-flip operation has no orbit map of the sector fields")
            return SymmetryScalarField(values.reduction, values.values[self.orbit_map])
        array = np.asarray(values, dtype=np.float64)
        if array.shape == self.inverse_rows.shape:
            return np.ascontiguousarray(array[self.inverse_rows])
        if self.orbit_map is not None and array.shape == self.orbit_map.shape:
            return np.ascontiguousarray(array[self.orbit_map])
        raise ValueError("the field matches neither the grid nor the orbits of the spin-flip operation")

    def up_field(self, values):
        """The inverse of :meth:`down_field`: the up-channel field whose image is the down-channel field ``values``
        (``f_up(r) = f_down(g r)``), on the grid or the orbits."""

        from .symmetry_fields import SymmetryScalarField

        if isinstance(values, SymmetryScalarField):
            return SymmetryScalarField(values.reduction, values.values[self._orbit_inverse()])
        array = np.asarray(values, dtype=np.float64)
        if array.shape == self.rows.shape:
            return np.ascontiguousarray(array[self.rows])
        if self.orbit_map is not None and array.shape == self.orbit_map.shape:
            return np.ascontiguousarray(array[self._orbit_inverse()])
        raise ValueError("the field matches neither the grid nor the orbits of the spin-flip operation")

    def _orbit_inverse(self) -> np.ndarray:
        if self.orbit_map is None:
            raise ValueError("this spin-flip operation has no orbit map of the sector fields")
        if not self._inverse_orbit_map:
            # (the orbit map is a permutation of the orbits: _orbit_map)
            self._inverse_orbit_map.append(np.argsort(np.asarray(self.orbit_map, dtype=np.int64), kind="stable"))
        return self._inverse_orbit_map[0]

    def field_deviation(self, up, down=None) -> float:
        """``max |down_field(up) - down|`` over the grid or the orbits; ``down`` is ``up`` where not given."""

        image = _field_values(self.down_field(up))
        target = _field_values(up if down is None else down)
        if image.shape != target.shape:
            raise ValueError("the two fields of a spin-flip comparison have different shapes")
        return float(np.max(np.abs(image - target), initial=0.0))

    def ionic_potential_check(self, ionic_potential) -> tuple[float, float]:
        """``(deviation, bound)``: how far the operation maps the local ionic potential from itself, and the bound
        (:data:`IONIC_POTENTIAL_TOLERANCE` of ``max(1, max |V_ion|)``) it must not exceed for the route."""

        values = _field_values(ionic_potential)
        bound = IONIC_POTENTIAL_TOLERANCE * max(1.0, float(np.max(np.abs(values), initial=0.0)))
        return self.field_deviation(ionic_potential), bound

    def warm_start_check(self, up_potential, down_potential) -> tuple[float, float]:
        """``(deviation, bound)``: how far the down potential of a warm start is from the image of its up potential,
        and :data:`WARM_START_TOLERANCE` (Ry), which it must not exceed for the route."""

        return self.field_deviation(up_potential, down_potential), WARM_START_TOLERANCE

    def down_representations(self, representations) -> np.ndarray:
        """The sector labels (1-based) of the images of up states of ``representations`` (1-based)."""

        labels = np.asarray(representations, dtype=np.int32)
        if self.sector_map is None or labels.size == 0:
            return labels.copy()
        return (np.asarray(self.sector_map, dtype=np.int32)[labels - 1] + 1).astype(np.int32)

    def down_orbitals(self, orbitals):
        """The down-channel orbitals: a host array at once, others as a :class:`FlippedOrbitals`."""

        if isinstance(orbitals, np.ndarray):
            return _gather_rows(orbitals, self.inverse_rows)
        return FlippedOrbitals(orbitals, self)

    def channel_state(self) -> SpinFlipChannelState:
        return SpinFlipChannelState(self)

    # ----- forces

    def down_forces(self, forces) -> np.ndarray:
        """``F_down(g a) = R F_up(a)``: the nonlocal forces of the down channel from those of the up channel."""

        forces = np.asarray(forces, dtype=np.float64)
        result = np.zeros_like(forces)
        result[self.atom_image] = forces @ np.asarray(self.matrix, dtype=np.float64).T
        return result

    # ----- structures

    def atom_deviation(self, atoms) -> float:
        """How far (bohr) the operation carries an atom of ``atoms`` from its image atom of the partner label, at
        most; ``inf`` where the atoms or their labels no longer match."""

        atoms = tuple(atoms)
        if len(atoms) != self.atom_image.size:
            return float("inf")
        positions = np.asarray([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3)
        labels = [atom.symbol for atom in atoms]
        for index, image in enumerate(self.atom_image):
            if self.partners.get(labels[index]) != labels[int(image)]:
                return float("inf")
        moved = positions @ np.asarray(self.matrix, dtype=np.float64).T
        if self.steps is not None:
            moved = moved + np.asarray(self.translation, dtype=np.float64) * self.steps
        difference = moved - positions[self.atom_image]
        if self.period is not None:
            difference = difference - self.period * np.rint(difference / self.period)
        return float(np.max(np.linalg.norm(difference, axis=1), initial=0.0))

    def holds_for(self, atoms, tolerance: float | None = None) -> bool:
        """Whether the operation still carries every atom of ``atoms`` onto its image atom of the partner label."""

        tolerance = self.tolerance if tolerance is None else float(tolerance)
        return bool(self.atom_deviation(atoms) <= tolerance)

    def device_inverse_rows(self, cp):
        device = int(cp.cuda.runtime.getDevice())
        rows = self._device_rows.get(device)
        if rows is None:
            rows = cp.asarray(self.inverse_rows)
            self._device_rows[device] = rows
        return rows


def _gather_rows(values: np.ndarray, rows: np.ndarray, out: np.ndarray | None = None, block: int = 16) -> np.ndarray:
    """``values[rows]`` of a host matrix, a few columns at a time (no temporary of the whole matrix)."""

    values = np.asarray(values)
    if values.ndim == 1:
        return values[rows]
    if out is None:
        out = np.empty((rows.size, values.shape[1]), dtype=values.dtype, order="F")
    for start in range(0, values.shape[1], block):
        stop = min(values.shape[1], start + block)
        out[:, start:stop] = values[rows, start:stop]
    return out


class FlippedOrbitals:
    """The orbitals of the down channel as the images of the up channel's: ``psi_down(r) = psi_up(g^-1 r)``.

    ``source`` are the up orbitals as the eigensolver returned them (sector orbitals, a device matrix); the images
    are formed where they are read, through the interface of ``source`` (``to_full_host``, ``to_full_device``,
    ``get``).
    """

    def __init__(self, source, operation: SpinFlipOperation) -> None:
        self.source = source
        self.operation = operation

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.source.shape)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def sector_handovers(self):
        return getattr(self.source, "sector_handovers", None)

    @property
    def spin_channel(self) -> int:
        return 1

    def with_source(self, source) -> "FlippedOrbitals":
        """The images of ``source``: the up orbitals of a result, which list every partner of a weighted state."""

        return FlippedOrbitals(source, self.operation)

    def release_intermediate_storage(self) -> None:
        """Nothing of its own: the up channel's orbitals release theirs."""

    def _source_host(self) -> np.ndarray:
        export = getattr(self.source, "to_full_host", None)
        if callable(export):
            return export()
        download = getattr(self.source, "get", None)
        if callable(download):
            return download()
        return np.asarray(self.source, dtype=np.float64)

    def to_full_host(self, block_states: int = 8) -> np.ndarray:
        return _gather_rows(self._source_host(), self.operation.inverse_rows)

    def fill_from(self, up: np.ndarray, out: np.ndarray) -> None:
        """Write the images of the host up orbitals ``up`` into ``out`` (the down columns of a joined matrix)."""

        _gather_rows(up, self.operation.inverse_rows, out=out)

    def get(self) -> np.ndarray:
        return self.to_full_host()

    def __array__(self, dtype=None, copy=None):
        values = self.to_full_host()
        return values if dtype is None else values.astype(dtype)

    def to_full_device(self):
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        materialize = getattr(self.source, "to_full_device", None)
        up = materialize() if callable(materialize) else self.source
        with up.device:
            rows = self.operation.device_inverse_rows(cp)
            return cp.asfortranarray(up[rows])


# ----- detection ----------------------------------------------------------


def _grid_period(grid) -> np.ndarray | None:
    from ..Symmetry.axis_reflection import _lattice_period

    return _lattice_period(grid)


def _ordered_matrices(cell: bool) -> list[np.ndarray]:
    """The 48 signed permutations: the identity first on a cell (pure translations), then the sign changes."""

    from ..Symmetry.axis_reflection import _signed_permutation_operations

    identity = np.eye(3, dtype=np.int64)
    matrices = [np.asarray(item, dtype=np.int64) for item in _signed_permutation_operations()]

    def kind(matrix) -> int:
        if np.array_equal(matrix, identity):
            return 0
        if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0:
            return 1
        return 2

    matrices.sort(key=kind)
    if not cell:
        matrices = [matrix for matrix in matrices if not np.array_equal(matrix, identity)]
    return matrices


class _PartnerAtoms:
    """The atoms of every label in a k-d tree (folded into the cell), built once for every candidate operation.

    :meth:`candidates` keeps the candidate translations of one matrix that carry :data:`PROBE_ATOMS` atoms (spread
    over the list) onto atoms of their partner labels, :meth:`image` forms the full map of one candidate.  The
    probes reject a wrong candidate at the cost of a few queries; the full map decides as before.
    """

    def __init__(self, positions, labels, partners, period) -> None:
        from scipy.spatial import cKDTree

        self.positions = positions
        self.labels = labels
        self.partners = partners
        self.period = period
        members: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            members.setdefault(label, []).append(index)
        self.members = {label: np.asarray(rows, dtype=np.int64) for label, rows in members.items()}
        self.trees = {}
        for label, rows in self.members.items():
            points = positions[rows]
            self.trees[label] = (
                cKDTree(points) if period is None else cKDTree(self._folded(points), boxsize=period)
            )
        count = len(labels)
        self.probes = np.unique(np.rint(np.linspace(0, count - 1, num=min(count, PROBE_ATOMS))).astype(np.int64))

    def _folded(self, points):
        from ..Symmetry.axis_reflection import _folded

        return _folded(points, self.period)

    def _query(self, label, points, bound=np.inf):
        tree = self.trees.get(label)
        if tree is None:
            return np.full(len(points), np.inf), np.zeros(len(points), dtype=np.int64)
        query = points if self.period is None else self._folded(points)
        distances, nearest = tree.query(query, k=1, workers=1, distance_upper_bound=bound)
        return np.asarray(distances, dtype=np.float64), np.asarray(nearest, dtype=np.int64)

    def candidates(self, matrix, shifts, tolerance) -> np.ndarray:
        """The rows of ``shifts`` (Cartesian, one per row) whose operation carries every probe atom onto an atom of
        its partner label within ``tolerance``."""

        rotation = np.asarray(matrix, dtype=np.float64).T
        keep = np.arange(len(shifts), dtype=np.int64)
        for probe in self.probes:
            if keep.size == 0:
                break
            label = self.labels[int(probe)]
            moved = self.positions[int(probe)] @ rotation + shifts[keep]
            distances, _ = self._query(self.partners[label], moved, bound=2.0 * tolerance + 1.0e-12)
            keep = keep[distances <= tolerance]
        return keep

    def image(self, matrix, shift, tolerance) -> np.ndarray | None:
        """``image[a]``: the atom of the partner label at ``R x_a + t`` within ``tolerance``, or ``None``."""

        moved = self.positions @ np.asarray(matrix, dtype=np.float64).T + shift
        image = np.full(len(self.labels), -1, dtype=np.int64)
        for label, rows in self.members.items():
            partner = self.partners[label]
            targets = self.members.get(partner)
            if targets is None or targets.size != rows.size:
                return None
            distances, nearest = self._query(partner, moved[rows])
            if np.any(distances > tolerance) or np.unique(nearest).size != nearest.size:
                return None
            image[rows] = targets[nearest]
        return image if np.all(image >= 0) else None


def _translation_candidates(positions, labels, partners, matrix, period, steps, counts, tolerance) -> np.ndarray:
    """The translations (grid steps, modulo the counts) that carry one polarized atom onto the atoms of its partner."""

    polarized = [label for label in dict.fromkeys(labels) if partners[label] != label]
    sizes = Counter(labels)
    label = min(polarized, key=lambda item: (sizes[item], labels.index(item)))
    origin = positions[labels.index(label)]
    targets = positions[[index for index, item in enumerate(labels) if item == partners[label]]]
    candidates = (targets - origin @ np.asarray(matrix, dtype=np.float64).T) / steps
    rounded = np.rint(candidates)
    limit = max(1.0e-6, 2.5 * tolerance / float(np.min(steps)))
    whole = np.all(np.abs(candidates - rounded) <= limit, axis=1)
    return np.unique(np.mod(rounded[whole].astype(np.int64), counts), axis=0)


def _orbit_map(reduction, inverse_rows) -> np.ndarray | None:
    """Orbit values of the image field (``orbit_map`` of :class:`SpinFlipOperation`), or ``None`` where the operation
    does not carry the orbits of ``reduction`` onto orbits."""

    full_to_wedge = np.asarray(reduction.full_to_wedge)
    representatives = np.asarray(reduction.representative_rows)
    images = full_to_wedge[inverse_rows]
    orbit_map = np.ascontiguousarray(images[representatives], dtype=np.int64)
    if not np.array_equal(orbit_map[full_to_wedge], images):
        return None
    if not np.array_equal(np.asarray(reduction.multiplicities)[orbit_map], np.asarray(reduction.multiplicities)):
        return None
    return orbit_map


def _sector_map(decomposition, inverse_rows) -> np.ndarray | None:
    """``sector_map[chi]``: the sector of the images of the states of sector ``chi``, or ``None`` where an image is
    in no sector (the phases of every row are compared)."""

    reduction = decomposition.reduction
    phases = np.asarray(decomposition.phases)
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    representatives = np.asarray(reduction.representative_rows)
    digests = {}
    for representation in range(phases.shape[0]):
        digests.setdefault(hashlib.sha256(np.ascontiguousarray(phases[representation]).tobytes()).digest(), representation)
    mapping = np.full(phases.shape[0], -1, dtype=np.int64)
    for representation in range(phases.shape[0]):
        moved = phases[representation][inverse_rows]
        expected = (moved * moved[representatives][full_to_wedge]).astype(phases.dtype, copy=False)
        target = digests.get(hashlib.sha256(np.ascontiguousarray(expected).tobytes()).digest())
        if target is None or not np.array_equal(phases[target], expected):
            return None
        mapping[representation] = target
    if np.unique(mapping).size != mapping.size:
        return None
    return mapping


def detect_spin_flip(
    grid,
    atoms,
    species,
    *,
    electron_count: float,
    start_densities=None,
    field_reduction=None,
    decomposition=None,
    fixed_occupations=None,
    ionic_potential=None,
) -> tuple[SpinFlipOperation | None, str]:
    """``(operation, "")`` of the spin-flip route of a run (see the module), or ``(None, reason)``.

    ``species`` maps the labels to their :class:`parsec_python.models.SpeciesPotential`; ``start_densities`` are
    the full-grid start densities of the two channels (checked where given); ``field_reduction`` is the orbit map of
    the sector fields and ``decomposition`` the sectors of the orbitals, where the run has them;
    ``fixed_occupations`` the ``(up, down)`` occupations of ``occup.in``, which must be equal; ``ionic_potential``
    the full-grid local ionic potential (point charges included), which the operation must map onto itself
    (:meth:`SpinFlipOperation.ionic_potential_check`; an operation that does not is passed over for the next).
    The candidate operations are tried in a fixed order (the same on every process).
    """

    from ..Symmetry.axis_reflection import _grid_row_mapping_operation, symmetry_atom_tolerance

    atoms = tuple(atoms)
    if not atoms:
        return None, "no atoms"
    count = float(electron_count)
    if abs(count - round(count)) > 1.0e-8 or int(round(count)) % 2:
        return None, f"the electron count {count:g} is not even: no state has equal channels"
    if fixed_occupations is not None:
        up, down = fixed_occupations
        if up is None or down is None or np.asarray(up).shape != np.asarray(down).shape or not np.array_equal(
            np.asarray(up, dtype=float), np.asarray(down, dtype=float)
        ):
            return None, "the occupations of occup.in differ between the channels"
    partners, reason = spin_partner_labels(atoms, species)
    if partners is None:
        return None, reason
    tolerance = min(float(symmetry_atom_tolerance()), ATOM_TOLERANCE)
    try:
        period = _grid_period(grid)
    except ValueError as error:
        return None, f"{error}"
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3)
    labels = [atom.symbol for atom in atoms]
    counts = np.asarray(getattr(grid, "points_per_axis", (1, 1, 1)), dtype=np.int64)
    steps = None if period is None else period / counts
    tried = 0
    reasons = []
    ionic_deviations = []
    partner_atoms = _PartnerAtoms(positions, labels, partners, period)
    for matrix in _ordered_matrices(cell=period is not None):
        if period is None:
            shifts = np.zeros((1, 3), dtype=np.int64)
            cartesian_shifts = np.zeros((1, 3), dtype=np.float64)
        else:
            shifts = _translation_candidates(positions, labels, partners, matrix, period, steps, counts, tolerance)
            cartesian_shifts = shifts * steps
        for candidate in partner_atoms.candidates(matrix, cartesian_shifts, tolerance):
            shift = shifts[candidate]
            image = partner_atoms.image(matrix, cartesian_shifts[candidate], tolerance)
            if image is None:
                continue
            tried += 1
            rows = _grid_row_mapping_operation(
                grid, matrix, 1.0e-10, translation=None if period is None else shift
            )
            if rows is None:
                reasons.append("does not map the grid onto itself")
                continue
            rows = np.asarray(rows, dtype=np.int64)
            inverse = np.empty_like(rows)
            inverse[rows] = np.arange(rows.size, dtype=np.int64)
            orbit_map = None
            if field_reduction is not None:
                orbit_map = _orbit_map(field_reduction, inverse)
                if orbit_map is None:
                    reasons.append("does not carry the orbits of the sector group onto orbits")
                    continue
            sector_map = None
            if decomposition is not None:
                sector_map = _sector_map(decomposition, inverse)
                if sector_map is None:
                    reasons.append("does not carry every sector onto a sector")
                    continue
            operation = SpinFlipOperation(
                matrix=np.asarray(matrix, dtype=np.int64),
                translation=np.asarray(shift, dtype=np.int64),
                rows=rows,
                inverse_rows=inverse,
                atom_image=image,
                partners=dict(partners),
                orbit_map=orbit_map,
                sector_map=sector_map,
                tolerance=tolerance,
                period=period,
                steps=steps,
            )
            if start_densities is not None:
                up, down = (np.asarray(item, dtype=np.float64) for item in start_densities)
                deviation = float(grid.volume_element * np.sum(np.abs(up[inverse] - down)))
                if deviation >= START_DENSITY_TOLERANCE:
                    return None, (
                        f"the operation {operation.description} maps the up start density onto the down one only "
                        f"within {deviation:.3e} electrons"
                    )
            if ionic_potential is not None:
                deviation, bound = operation.ionic_potential_check(ionic_potential)
                if not deviation <= bound:
                    # Point charges or another external term: the next operation may keep it.
                    ionic_deviations.append((deviation, bound))
                    continue
            return operation, ""
    if ionic_deviations:
        deviation, bound = min(ionic_deviations)
        reasons.append(
            "do not map the local ionic potential onto itself (point charges or another external term; "
            f"smallest maximum deviation {deviation:.3e} Ry, allowed {bound:.1e} Ry)"
        )
    if tried == 0:
        return None, "no operation maps every atom onto an atom of its partner label"
    return None, f"the {tried} operation(s) that map the atoms onto their partners " + "; ".join(
        sorted(set(reasons))
    )
