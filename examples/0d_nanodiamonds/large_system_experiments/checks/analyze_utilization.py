"""Derive utilization-report arithmetic from curated evidence, using only stdlib.

These phase timers do not reconstruct the platform's GPU/SM/Tensor time series.
The old local-GPU event timers and new synchronous MPI wall timers differ.
"""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / 'data' / name).read_text())


def describe(values):
    return dict(minimum=min(values), median=statistics.median(values),
                maximum=max(values))


def derive():
    profile = read('large_restart_profile.json')
    evidence = read('utilization_evidence.json')
    old = read('old_16gb200_profile.json')
    ranks = profile['per_rank']
    diagonalization = profile['result']['first_iteration_diagonalization_seconds']
    phase_names = list(ranks[0]['distributed_state'][0]['seconds'])
    phases = {name: describe([r['distributed_state'][0]['seconds'][name]
                             for r in ranks]) for name in phase_names}
    layout = [sum(r['distributed_state'][0]['seconds'][name]
                  for name in ('to_rows', 'to_columns')) for r in ranks]
    rank0 = ranks[0]['distributed_state'][0]
    fraction = layout[0] / diagonalization
    audit = evidence['smart_reference_audit']['audit_by_sector_leader']
    audit_seconds = audit['0']['seconds_sum']
    old_rank0 = old['per_rank'][0]['distributed_state'][0]
    # An illustrative buffer model, not a measured MPI call count. Actual
    # environment overrides, changing state counts and QR paths may change it.
    rows, states, group_size = 12575874, 11962, 16
    exchange_bytes, dense_bytes = 1 << 30, 64 << 20
    width = max(1, exchange_bytes // (8 * rows))
    rounds = math.ceil(math.ceil(states / group_size) / width)
    rotation_rows = max(1, dense_bytes // (8 * states))
    return {
        'scope': 'Completed smart reference; not a030 dashboard attribution',
        'ranks_observed': len(ranks),
        'ranks_with_36_ritz_passes': sum(
            r['distributed_state'][0]['ritz_passes'] == 36 for r in ranks),
        'diagonalization_seconds': diagonalization,
        'rank_phase_seconds': phases,
        'rank_layout_seconds': describe(layout),
        'rank_layout_fraction_of_reference_diagonalization': describe(
            [value / diagonalization for value in layout]),
        'rank_solve_seconds': describe([r['command_seconds']['solve'] for r in ranks]),
        'rank0': {
            'layout_seconds': layout[0],
            'layout_fraction_of_diagonalization': fraction,
            'logical_send_bytes': rank0['exchanged_bytes'],
            'logical_send_bytes_per_layout_second': rank0['exchanged_bytes'] / layout[0],
            'logical_rate_is_not_nic_bandwidth': True,
            'phase_timer_sum_seconds': sum(rank0['seconds'].values()),
            'audit_seconds': audit_seconds,
            'audit_fraction_of_diagonalization': audit_seconds / diagonalization,
            'unassigned_after_phase_and_audit_seconds': diagonalization -
                sum(rank0['seconds'].values()) - audit_seconds,
            'unassigned_is_not_proven_cpu_or_idle_time': True,
        },
        'old_16gpu_rank0': {
            'clock': old_rank0['stage_clock'],
            'pipeline': old_rank0['pipeline'],
            'ritz_passes': old_rank0['ritz_passes'],
            'layout_event_seconds': sum(old_rank0['seconds'][k]
                                        for k in ('to_rows', 'to_columns')),
            'reported_overlap_seconds': old_rank0['overlap_seconds'],
            'diagonalization_wall_seconds': old['program']['scf_timings']['diagonalization_seconds'],
            'direct_phase_fraction_comparison_valid': False,
        },
        'buffer_model_assuming_defaults': {
            'rows_per_sector': rows, 'working_states_example': states,
            'ranks_per_sector': group_size, 'exchange_bytes': exchange_bytes,
            'columns_per_exchange_round': width, 'rounds_per_ritz': rounds,
            'alltoallv_calls_per_ritz': 3 * rounds,
            'alltoallv_calls_for_36_ritz_passes': 36 * 3 * rounds,
            'dense_scratch_bytes': dense_bytes, 'rotation_rows_per_tile': rotation_rows,
            'rotation_tiles_per_ritz': math.ceil(math.ceil(rows / group_size) / rotation_rows),
            'audit_columns_per_tile_at_64mib': max(1, dense_bytes // (3 * rows * 8)),
            'measured_call_count': False,
        },
        'hypothetical_layout_speedup_sensitivity': {
            'scope': 'Amdahl illustration using rank0 fraction; not a forecast or measured speedup',
            'layout_2x': 1 / ((1 - fraction) + fraction / 2),
            'layout_4x': 1 / ((1 - fraction) + fraction / 4),
        },
    }


if __name__ == '__main__':
    result = derive()
    (ROOT / 'data' / 'gpu_utilization_analysis.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in
                     ('ranks_observed', 'rank_layout_seconds', 'rank0',
                      'buffer_model_assuming_defaults')}, indent=2))
