"""The Fock loop of a hybrid functional on the devices of a CuPy calculation.

:func:`parsec_python.SCF.hybrid.run_hybrid_scf` is the loop, unchanged: its
records, energies, stopping rule and result are those of the reference
path.  An accelerated system takes part through :class:`DeviceExchange`
(``system.exchange_device``), whose :meth:`DeviceExchange.fock_engine`
supplies the pieces the loop asks of
:class:`parsec_python.SCF.hybrid.HostExchange`:

- the orbitals of a result stay where the eigensolvers keep them (one
  full-grid block on one device, or the symmetry sectors on theirs);
  ``V_x Psi`` is formed on every device of the process
  (:func:`~parsec_python.acceleration.Hamiltonian.exchange_device.apply_device_exchange`)
  and compressed per sector
  (:func:`~parsec_python.acceleration.Hamiltonian.exchange_device.compress_device_exchange`);
- ``a_x V_ACE`` of a channel is written into the device Hamiltonians that
  the eigensolvers apply (:meth:`DeviceExchange.bind`, called by the
  system's ``hamiltonian`` for the channel being solved): the full-grid
  operator, or the operator of each sector with its block ``-xi_G
  xi_G^T``.  Every product of those operators then holds it: CHEBFF and
  SUBSPACE filters (grouped, see
  :func:`parsec_python.acceleration.Eigensolvers.chebyshev._exchange_block_filter`),
  the Lanczos bounds, the Rayleigh--Ritz products and the residuals;
- the semilocal start runs with the start functional on the device (its
  own XC evaluators), the warm starts between the loops keep the device
  subspaces (``start_new_scf_loop`` of the eigensolvers, as between the
  structures of an ionic loop);
- the energies of an inner iteration and ``E_x(P, P0)`` are formed where
  the orbitals are; only scalars reach the host.

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from parsec_python.SCF.hybrid import FockOperator, HybridDefinition, hybrid_functional_of

from ..Hamiltonian.exchange_device import (
    DevicePairSolvers,
    apply_device_exchange,
    channel_orbitals,
    compress_device_exchange,
)


class WeightedHybridAbandoned(RuntimeError):
    """An unpolarized hybrid in the weighted sectors of a point group found a partly filled degenerate level.

    The weighted sectors give every partner of a level one occupation and keep the density of the full group, so the
    exact exchange cannot split such a level there, while the sectors of the abelian group and the full grid can: from
    the symmetric ensemble, a saddle of the hybrid functional, they reach a lower state (PBE0 at 80 K, SCF 1e-10 Ry:
    the C atom of Oh 0.108 Ry lower, CH4+ of Td 0.026, unpolarized O2 0.089).  ``stage`` names the loop, ``report``
    (:class:`.spin_sectors.FractionalOccupations`) the levels; the driver starts the calculation again in the sectors
    of the abelian group (``driver._SpinSectorFallback``), with a NOTE.
    """

    def __init__(self, stage: str, report) -> None:
        self.stage = stage
        self.report = report
        super().__init__(
            f"weighted sectors of a hybrid: {report.reason} after {stage} "
            f"({report.measure} = {report.largest:.3e} >= {report.tolerance:.0e}): " + "; ".join(report.lines)
        )


#: The reason of :func:`partly_filled_partner_levels`.
PARTLY_FILLED_LEVEL_REASON = "a partly filled degenerate level"


def partly_filled_partner_levels(result, tolerance: float | None = None):
    """The partly filled degenerate levels of an unpolarized result of weighted sectors
    (:class:`.spin_sectors.FractionalOccupations`), ``None`` where there are none, the result has no weighted sectors
    or two channels (the guard of the spin sectors tests them).

    The levels of :func:`.spin_sectors.weighted_partly_filled_levels`, which the NOTE of a semilocal run in the
    weighted sectors reads too (``driver._say_symmetric_level``): every listed partner of a solved state of weight
    ``w`` is a level of its own label, so a run of levels within :data:`.spin_sectors.MULTIPLET_WIDTH` over two or
    more labels, one of whose occupations has ``min(f, 1 - f)`` of ``tolerance``
    (:data:`.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE` by default) or more, is partly filled; degenerate states
    of one label mix as on the full grid and do not count.  The report names :data:`PARTLY_FILLED_LEVEL_REASON` and
    the electrons of both spins of a level.
    """

    from .spin_sectors import SPIN_SECTOR_OCCUPATION_TOLERANCE, weighted_partly_filled_levels

    if getattr(result, "spin", None) is not None:
        return None
    return weighted_partly_filled_levels(
        result,
        SPIN_SECTOR_OCCUPATION_TOLERANCE if tolerance is None else float(tolerance),
        reason=PARTLY_FILLED_LEVEL_REASON,
        both_spins=True,
    )


@dataclass(frozen=True)
class DeviceFockOperator(FockOperator):
    """:class:`parsec_python.SCF.hybrid.FockOperator` whose channels live on devices.

    ``channels`` holds a
    :class:`~parsec_python.acceleration.Hamiltonian.exchange_device.DeviceCompressedExchange`
    per channel (``None`` without electrons); ``exchange`` reads the
    orbitals of an iteration where the eigensolver keeps them.
    """

    exchange: Any = None

    def cross_energy(self, orbitals, occupations) -> float:
        """``E_x(P, P0) = sum_s (w/2) sum_i f_i <q_i|V_ACE[P0]|q_i>`` (Ry, unscaled)."""

        if len(orbitals) != len(self.channels):
            raise ValueError("one orbital block per spin channel of the operator is needed")
        total = 0.0
        for channel, block, weights in zip(self.channels, orbitals, occupations):
            if channel is None:
                continue
            total += channel.band_energy(
                self.exchange.channel(block, weights, channel.spin_weight), scaled=False
            )
        return 0.5 * total


class DeviceExchange:
    """Exact exchange in the device Hamiltonians of a prepared CuPy system.

    ``backend`` is the CuPy backend (its full-grid operator, where the
    orbitals are on the full grid), ``sectors`` the symmetry eigensolver or
    ``None``, ``devices`` the devices of the process (every one solves pair
    densities), ``start_xc`` the ``(unpolarized, spin)`` evaluators of the
    start functional (``None`` entries: the reference functional on the
    host), ``batch_size`` ``Exx_Batch_Size``.
    """

    def __init__(
        self,
        *,
        backend,
        sectors,
        devices,
        start_xc=(None, None),
        multipole_order: int,
        batch_size: int,
        volume_element: float,
    ) -> None:
        self.backend = backend
        self.volume_element = float(volume_element)
        self.sectors = sectors
        self.devices = tuple(int(device) for device in devices)
        self.start_xc = tuple(start_xc)
        self.multipole_order = int(multipole_order)
        self.batch_size = int(batch_size)
        self.expansions: dict = {}
        # the exchange of the weighted sectors of each point group (PARSEC_ORBITAL_GROUP_HYBRID), by reduction
        self.weightings: dict = {}
        self._solvers: dict = {}
        self.bound: object | None = None

    # ----- orbitals --------------------------------------------------------

    def _sector_devices(self):
        if self.sectors is None:
            return None
        return getattr(self.sectors, "_sector_device_ids", None)

    def channel(self, wavefunctions, occupations, spin_weight: float, volume_element: float | None = None):
        """:class:`~parsec_python.acceleration.Hamiltonian.exchange_device.ChannelOrbitals` of an eigensolver's orbitals."""

        if volume_element is None:
            volume_element = self.volume_element
        return channel_orbitals(
            wavefunctions,
            occupations,
            spin_weight,
            volume_element,
            expansions=self.expansions,
            sector_devices=self._sector_devices(),
            weightings=self.weightings,
        )

    @staticmethod
    def _solved_occupations(result):
        """The occupations of the states ``result`` solved: those of the SCF loop, times their weights, for weighted
        sectors (``sector_states``, whose ``occupations`` list every partner), else ``result.occupations``."""

        solved = getattr(result, "sector_states", None)
        if solved is not None:
            return np.asarray(solved[1], dtype=np.float64)
        return result.occupations

    def orbital_sets(self, result):
        """The channels of an SCF result, where its orbitals are: one set or two."""

        channels = getattr(result, "spin", None)
        if channels is None:
            return (self.channel(result.wavefunctions, self._solved_occupations(result), 2.0),)
        return tuple(
            self.channel(channel.wavefunctions, self._solved_occupations(channel), 1.0) for channel in channels
        )

    # ----- the term in the Hamiltonians -------------------------------------

    def operators(self) -> dict:
        """The device operators that hold the term: representation (``None``: full grid) -> operator."""

        if self.sectors is not None:
            return {
                representation: operator
                for representation, operator in enumerate(self.sectors._operators)
                if operator is not None
            }
        operator = getattr(self.backend, "device_operator", None)
        if operator is None:
            operator = getattr(self.backend, "ensure_device_operator")()
        return {None: operator}

    def bind(self, channel) -> None:
        """Write ``a_x V_ACE`` of a channel into the device operators (``None``: remove it)."""

        for representation, operator in self.operators().items():
            projectors = (
                None
                if channel is None or getattr(channel, "released", False)
                else channel.projectors.get(representation)
            )
            if projectors is None:
                operator.bind_exchange(None)
            else:
                operator.bind_exchange(projectors, -float(channel.fraction))
        self.bound = channel

    # ----- the loop ----------------------------------------------------------

    def planned_device_bytes(self, reference, definition: HybridDefinition, states: int) -> dict[int, int]:
        """What the exchange keeps or takes on each device beside the orbitals (an estimate).

        :func:`~parsec_python.acceleration.Hamiltonian.exchange_device.planned_exchange_bytes`
        for the sectors of this system (or its full grid) and ``states``
        orbitals per channel.  The driver hands it to the memory rules of the
        sectors, which then spill the sector states to the host where the
        states and the exchange do not fit together.  The window of the outer
        DIIS (``Exx_Outer_Mixing: diis``) is not in it: it takes what the
        devices have free while the loop runs (:func:`~.fock_device.window_fits`).
        """

        from ..Hamiltonian.exchange_device import estimated_box_points, planned_exchange_bytes

        grid = reference.grid
        kernels = [kernel for kernel, _fraction, _omega in definition.terms]
        parity = self._planned_parity(grid, kernels)
        if self.sectors is not None:
            owned = [
                representation
                for representation, operator in enumerate(self.sectors._operators)
                if operator is not None
            ]
            rows = {
                representation: int(self.sectors.decomposition.sector_size(representation))
                for representation in owned
            }
            places = {
                representation: int(self.sectors._sector_device_ids[representation])
                for representation in owned
            }
        else:
            operator = self.operators()[None]
            rows = {None: int(grid.size)}
            places = {None: int(operator.effective_potential.device.id)}
        spin = bool(getattr(reference.input.scf, "spin_polarized", False))
        counts = planned_exchange_bytes(
            full_size=int(grid.size),
            box_points=estimated_box_points(grid, kernels, self.multipole_order),
            kernels=kernels,
            sector_rows=rows,
            sector_devices=places,
            devices=self.devices,
            states=int(states),
            channels=2 if spin else 1,
            batch_size=self.batch_size,
            parity=parity,
        )
        return counts

    def _planned_parity(self, grid, kernels):
        """The parity classes the sector orbitals will use (:func:`~..exchange_device.planned_parity`), or ``None``."""

        from ..Hamiltonian.exchange_device import planned_parity

        decomposition = getattr(self.sectors, "decomposition", None)
        if decomposition is None:
            return None
        return planned_parity(grid, kernels, self.multipole_order, decomposition.reduction)

    def pair_solvers(self, system, definition: HybridDefinition) -> DevicePairSolvers:
        reference = _periodic_reference(system)
        if reference is not None:
            return self._periodic_pair_solvers(reference, definition)
        terms = tuple(definition.terms)
        if len(terms) == 1:
            kernel, _fraction, omega = terms[0]
            terms = ((kernel, 1.0, omega),)
        key = (terms, self.batch_size)
        solvers = self._solvers.get(key)
        if solvers is None:
            # The host plans are built beside the semilocal start of the loop;
            # the plans of the parity classes only where the sector orbitals
            # will take that path.
            kernels = [kernel for kernel, _weight, _omega in terms]
            solvers = DevicePairSolvers(
                system.grid,
                terms,
                self.multipole_order,
                self.batch_size,
                background=True,
                parity=self._planned_parity(system.grid, kernels) is not None,
            )
            self._solvers[key] = solvers
        return solvers

    def _periodic_pair_solvers(self, reference, definition: HybridDefinition):
        """The cuFFT pair solver of a periodic cell at the Gamma point (real orbitals).

        The kernel and the box of the host solver
        (:func:`parsec_python.SCF.hybrid_periodic.periodic_pair_potentials`,
        what :func:`parsec_python.SCF.hybrid.default_pair_solver` gives the
        reference loop), so the device pair potentials are the host's to
        round-off.
        """

        from parsec_python.SCF.hybrid_periodic import periodic_pair_potentials

        from ..Hamiltonian.bloch_exchange_device import DevicePeriodicBox, DevicePeriodicPairSolvers

        key = ("periodic", tuple(definition.terms), float(definition.fraction), self.batch_size)
        solvers = self._solvers.get(key)
        if solvers is None:
            potentials = periodic_pair_potentials(reference, definition)
            kernel = potentials.kernel
            describe = getattr(kernel, "describe", None)
            solvers = DevicePeriodicPairSolvers(
                kernel,
                DevicePeriodicBox(potentials.box.shape, potentials.fft_shape, getattr(potentials.box, "flat", None)),
                self.batch_size,
                description=str(describe() if callable(describe) else getattr(kernel, "description", "")),
            )
            self._solvers[key] = solvers
        return solvers

    def fock_engine(self, system, definition: HybridDefinition) -> "DeviceFockEngine":
        if _periodic_reference(system) is not None:
            # A periodic cell: its pair densities by FFT whatever Exx_Pair_Solver says (the reference loop's too).
            return DeviceFockEngine(self, system, definition)
        if definition.pair_solver != "kronecker":
            raise NotImplementedError(
                "the device exchange operator solves the pair densities with the Kronecker "
                "solver only (Exx_Pair_Solver: kronecker)"
            )
        return DeviceFockEngine(self, system, definition)

    def start_system(self, system, definition: HybridDefinition):
        """``system`` with the start functional of the hybrid, on the same devices."""

        reference = system.reference
        scf = replace(
            reference.input.scf, xc_functional=definition.start_functional, exact_exchange=None
        )
        start_reference = replace(reference, input=replace(reference.input, scf=scf))
        unpolarized, spin = self.start_xc
        return replace(
            system,
            reference=start_reference,
            xc_evaluator=unpolarized,
            spin_xc_evaluator=spin,
        )


def _periodic_reference(system):
    """The prepared reference system of a periodic cell, or ``None`` (an isolated system).

    ``system`` is an accelerated system (its ``reference``), a reference
    system, or what only names a grid (the exchange service of an MPI rank
    of a cluster, ``SCF/hybrid_mpi.py``, which has no input; the service of a
    periodic cell passes its reference system since 2026-11-22).
    """

    reference = getattr(system, "reference", system)
    problem = getattr(reference, "input", None)
    return reference if getattr(problem, "periodic_cell", None) is not None else None


def _new_loop_state(solver, state, index: int | None = None):
    """The saved subspace of an eigensolver, marked for a new SCF loop on its device."""

    from parsec_python.SCF.single_point import state_for_new_potential

    if index is not None:
        for_channel = getattr(solver, "for_spin_channel", None)
        if callable(for_channel):
            return for_channel(index).start_new_scf_loop()
        return state_for_new_potential(state)
    restart = getattr(solver, "start_new_scf_loop", None)
    return restart() if callable(restart) else state_for_new_potential(state)


class DeviceFockEngine:
    """The pieces of :func:`parsec_python.SCF.hybrid.run_hybrid_scf` on the devices."""

    def __init__(self, exchange: DeviceExchange, system, definition: HybridDefinition) -> None:
        self.device = exchange
        self.definition = definition
        self.solvers = exchange.pair_solvers(system, definition)
        scf = getattr(getattr(system, "input", None), "scf", None)
        # occup.in fills the partners of a weighted state alike (Occupations.weighted_fixed_occupations): its equal
        # fractional occupations are the input's, which the abelian sectors keep too
        self.fixed_occupations = getattr(scf, "fixed_occupations", None) is not None

    def weighted_guard(self, result, stage: str) -> None:
        """Raise :class:`WeightedHybridAbandoned` where an unpolarized ``result`` of weighted sectors has a partly
        filled degenerate level (:func:`partly_filled_partner_levels`).

        :func:`parsec_python.SCF.hybrid.run_hybrid_scf` asks after its semilocal start and after every inner loop.
        Nothing for orbitals of other sectors or of the full grid, for two channels (the guard of the spin sectors
        tests them, ``SCF/spin_sectors.py``) and for the occupations of ``occup.in``.
        """

        if self.fixed_occupations:
            return
        report = partly_filled_partner_levels(result)
        if report is not None:
            raise WeightedHybridAbandoned(stage, report)

    @property
    def description(self) -> str:
        return self.solvers.description

    def take_exchange_paths(self):
        """The paths the builds took since the last call (counts: the parity classes or the full grid of a
        cluster's sector orbitals, the translation classes of a periodic cell's (``PARSEC_CUPY_EXCHANGE_TRANSLATIONS``),
        its pooled block or its fallback to the sector tiles, ``PARSEC_CUPY_EXCHANGE_POOL``), forgotten here; ``None``
        where the pair solvers keep no such record.

        The pair solvers are cached for the run of the driver (``DeviceExchange._solvers``): the Fock loop takes
        the record at its start, after each of its steps and names those builds in its records
        (:func:`parsec_python.SCF.hybrid.run_hybrid_scf`), so a later structure does not inherit them.
        """

        take = getattr(self.solvers, "take_paths", None)
        return take() if callable(take) else None

    def describe_exchange(self, paths) -> str:
        """The pair solvers' description of the builds of ``paths`` (from :meth:`take_exchange_paths`)."""

        describe = getattr(self.solvers, "describe", None)
        return str(describe(paths) if callable(describe) else self.solvers.description)

    def start_system(self, system, definition):
        return self.device.start_system(system, definition)

    def orbital_sets(self, result):
        return self.device.orbital_sets(result)

    @staticmethod
    def _internal(system, values):
        adapter = getattr(system, "scalar_field_adapter", None)
        array = np.asarray(values, dtype=np.float64)
        return array if adapter is None else adapter.from_full(array)

    @staticmethod
    def _public(system, values):
        adapter = getattr(system, "scalar_field_adapter", None)
        return np.asarray(values if adapter is None else adapter.to_full(values), dtype=np.float64)

    def transition_warm_start(self, system, start):
        """The first inner loop: ``V_ion + V_H[rho] + V_xc^semilocal[rho]``, the subspace kept.

        :func:`parsec_python.SCF.hybrid._transition_warm_start` with the
        semilocal part of the hybrid evaluated where the system evaluates
        it, and the device subspaces marked for a new loop.
        """

        from parsec_python.SCF.single_point import SCFWarmStart

        ionic = np.asarray(system.ionic_potential, dtype=np.float64)
        hartree = np.asarray(start.hartree_potential, dtype=np.float64)
        solver = getattr(system, "eigenproblem_solver", None)
        channels = getattr(start, "spin", None)
        # A meta-GGA hybrid starts its first inner loop as every meta-GGA loop:
        # the start functional's potential (PBE) and no orbital term yet.
        evaluator = (
            self.start_system(system, self.definition)
            if getattr(system.input.scf, "is_meta_gga", False)
            else system
        )
        if channels is None:
            xc = evaluator.evaluate_xc(self._internal(system, start.density))
            return SCFWarmStart(
                input_potential=ionic + hartree + self._public(system, xc.potential),
                hartree_potential=hartree,
                eigensolver_state=_new_loop_state(solver, start.eigensolver_state),
            )
        # (Orbit fields where the channels are in the symmetry sectors.)
        xc = evaluator.evaluate_spin_xc(
            self._internal(system, channels[0].density),
            self._internal(system, channels[1].density),
        )
        return SCFWarmStart(
            input_potential=tuple(
                ionic + hartree + self._public(system, potential) for potential in xc.potentials
            ),
            hartree_potential=hartree,
            eigensolver_state=tuple(
                _new_loop_state(solver, channel.eigensolver_state, index)
                for index, channel in enumerate(channels)
            ),
        )

    def next_warm_start(self, system, result):
        from parsec_python.SCF.single_point import meta_gga_warm_start

        from ..driver import _parsec_warm_start

        # A meta-GGA hybrid: the next inner loop takes over the mixed vtau too.
        return meta_gga_warm_start(_parsec_warm_start(system, result, system), result)

    def start_total_energy(self, system, start, exchange_energy: float, fraction: float) -> float | None:
        channels = getattr(start, "spin", None)
        if getattr(system.input.scf, "is_meta_gga", False):
            # tau of the start orbitals on the host full grid (the reference system), where the
            # result holds them; the energy of the start is not reported otherwise.
            from parsec_python.SCF.hybrid import meta_gga_start_semilocal

            semilocal = meta_gga_start_semilocal(system.reference, start)
            if semilocal is None:
                return None
            energies = start.energies
            return float(
                energies.total - energies.exchange_correlation + semilocal + fraction * exchange_energy
            )
        if channels is None:
            semilocal = system.evaluate_xc(self._internal(system, start.density)).total_energy
        else:
            semilocal = system.evaluate_spin_xc(
                self._internal(system, channels[0].density),
                self._internal(system, channels[1].density),
            ).total_energy
        energies = start.energies
        return float(
            energies.total - energies.exchange_correlation + semilocal + fraction * exchange_energy
        )

    def _devices(self, orbitals, columns: str, known) -> tuple:
        """The devices of a build: every device of the process, or one for a small build of a periodic cell."""

        small = getattr(self.solvers, "small_build", None)
        if callable(small) and small(orbitals, columns=columns, known=known):
            return (int(orbitals.sectors[0].device),)
        return self.device.devices

    def exchange(self, sets, *, columns: str = "occupied", known=None):
        return [
            apply_device_exchange(
                orbitals,
                self.solvers,
                self._devices(orbitals, columns, None if known is None else known[index]),
                columns=columns,
                known=None if known is None else known[index],
            )
            for index, orbitals in enumerate(sets)
        ]

    def fock_operator(self, sets, applications, fraction: float) -> DeviceFockOperator:
        reference_energy = float(sum(application.energy for application in applications))
        channels = []
        for application in applications:
            if not application.occupied:
                channels.append(None)
                continue
            channels.append(compress_device_exchange(application, fraction=fraction))
        return DeviceFockOperator(
            channels=tuple(channels),
            fraction=float(fraction),
            reference_energy=reference_energy,
            exchange=self.device,
        )

    def cross_energy(self, operator: DeviceFockOperator, sets) -> float:
        return 0.5 * sum(
            0.0 if channel is None else channel.band_energy(orbitals, scaled=False)
            for channel, orbitals in zip(operator.channels, sets)
        )

    def drop_orbitals(self, sets, result=None) -> None:
        """Let the orbitals of ``sets`` and ``result`` go (views of the device subspaces).

        The operator built from them holds its own ``xi``; the eigensolvers
        keep the subspaces they continue from.  ``result`` keeps its
        energies, fields and levels.
        """

        for orbitals in sets:
            # (and the coset images of weighted sectors, which read the same bases)
            for sector in orbitals.all_sources():
                sector.vectors = None
        if result is not None:
            channels = getattr(result, "spin", None)
            for wavefunctions in (
                [result.wavefunctions]
                if channels is None
                else [channel.wavefunctions for channel in channels]
            ):
                release = getattr(wavefunctions, "release_intermediate_storage", None)
                if callable(release):
                    release()

    def occupied_blocks(self, sets) -> list:
        """Copies of the occupied orbitals of each channel, by sector (``Exx_Outer_Mixing: diis``)."""

        from .fock_device import occupied_blocks

        return occupied_blocks(sets)

    def natural_sets(self, sets, natural):
        """The channels of natural orbitals ``natural`` and the empty states of ``sets``."""

        from .fock_device import build_fits, natural_channels

        if not build_fits(sets):
            raise MemoryError("the channel of the extrapolated density matrix does not fit the devices")

        return natural_channels(sets, natural)


    def release(self, operator: DeviceFockOperator) -> None:
        """Remove the operator from the device Hamiltonians and free its ``xi``."""

        self.device.bind(None)
        for channel in operator.channels:
            if channel is not None:
                channel.release()


def device_exchange_for(
    reference,
    backend,
    sectors,
    *,
    selected_backend: str,
    start_xc=(None, None),
):
    """The :class:`DeviceExchange` of a prepared system, ``None`` without a hybrid.

    A hybrid on another backend than CuPy is refused (the accelerated command
    line sends it to the reference solver).
    """

    scf = getattr(getattr(reference, "input", None), "scf", None)
    if not getattr(scf, "is_hybrid", False):
        return None
    if selected_backend != "cupy":
        raise NotImplementedError(
            f"the self-consistent hybrid functional {scf.xc_functional!r} runs on the CuPy "
            "backend of the accelerated path or on the reference solver (reference_main.py)"
        )
    from parsec_python.V_xc.exact_exchange import pair_hartree_settings

    from ..Eigensolvers.symmetry import selected_device_ids
    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    devices = (
        tuple(sectors.device_ids)
        if sectors is not None
        else selected_device_ids(int(cp.cuda.Device().id))
    )
    hybrid_functional_of(scf)  # the input's overrides are valid
    from ..Hamiltonian.exchange_device import (
        exchange_parity_box_setting,
        exchange_parity_setting,
        exchange_pool_setting,
    )

    from ..Hamiltonian.exchange_translations import exchange_column_update_setting, exchange_translation_setting

    # The switches of the parity classes, of the pooled block of periodic sector orbitals and of their translation
    # classes (PARSEC_CUPY_EXCHANGE_POOL, PARSEC_CUPY_EXCHANGE_TRANSLATIONS and the update of their W by column,
    # PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE, read by the builds of such orbitals only) are refused here when wrong.
    exchange_parity_setting()
    exchange_parity_box_setting()
    exchange_pool_setting()
    exchange_translation_setting()
    exchange_column_update_setting()
    return DeviceExchange(
        backend=backend,
        sectors=sectors,
        devices=devices,
        start_xc=start_xc,
        multipole_order=pair_hartree_settings(reference.input.hartree).multipole_order,
        batch_size=scf.exact_exchange.batch_size,
        volume_element=float(reference.grid.volume_element),
    )


__all__ = [
    "DeviceExchange",
    "DeviceFockEngine",
    "DeviceFockOperator",
    "PARTLY_FILLED_LEVEL_REASON",
    "WeightedHybridAbandoned",
    "device_exchange_for",
    "partly_filled_partner_levels",
]
