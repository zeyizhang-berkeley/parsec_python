"""Regenerate the two report figures from curated data (no cluster access)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'figures'


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    svg = FIGURES / (name + '.svg')
    fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
    # Matplotlib emits trailing spaces in SVG paths; keep generated diffs clean.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(FIGURES / (name + '.png'), bbox_inches='tight', dpi=160)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'parsec-20261115'})
    with (ROOT / 'data/reconstruction_curve.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7.1, 4.2), layout='constrained')
    labels = [('inherit_historical_interrupted', 'Historical inherit (interrupted)', '#64748b', '--'),
              ('smart_same_allocation', 'Smart candidate', '#0969da', '-')]
    for key, label, color, style in labels:
        points = sorted((r for r in rows if r['series'] == key), key=lambda r:int(r['passes']))
        ax.plot([int(r['passes']) for r in points],
                [float(r['maximum_level_change_ry']) for r in points],
                label=label, color=color, linestyle=style, marker='.', markersize=4)
    ax.axhline(1e-4, color='#b54708', linewidth=1.2, label='Level-stability threshold')
    ax.set(yscale='log', xlabel='Extra reconstruction passes',
           ylabel='Maximum change in requested levels (Ry)',
           title='95,648 valence electrons on 64 GB200 GPUs')
    ax.grid(alpha=.18, which='both'); ax.legend(frameon=False, fontsize=9)
    fig.text(.02, -.045, 'Same parent, different historical allocation; level change is not eigenvector residual.',
             fontsize=8, color='#475569')
    save(fig, 'reconstruction_progress')

    data = json.loads((ROOT / 'data/large_restart_profile.json').read_text())
    rank = next(r for r in data['per_rank'] if r['rank'] == 0)
    phases = rank['distributed_state'][0]['seconds']
    order = sorted(phases, key=phases.get)
    labels = {'to_rows':'Column to row layout', 'to_columns':'Row to column layout',
              'gram':'Gram construction','rotate':'Ritz rotation','reduce':'Reduction',
              'dense':'Dense solve','filter':'Polynomial filtering','apply':'Hamiltonian apply'}
    fig, ax = plt.subplots(figsize=(7.1, 4.5), layout='constrained')
    values = [phases[k] for k in order]
    bars = ax.barh([labels.get(k,k) for k in order], values,
                   color=['#0969da' if k in ('to_rows','to_columns') else '#94a3b8' for k in order])
    ax.bar_label(bars, fmt='%.1f s', padding=4, fontsize=9)
    ax.set(xlabel='Accumulated phase time (seconds)', xlim=(0,max(values)*1.23),
           title='Large restart: rank 0 / sector 0 profile')
    ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    fig.text(.02, -.045, 'Audited run. Timers include packing and waits; do not sum ranks as wall time.',
             fontsize=8, color='#475569')
    save(fig, 'restart_profile')


if __name__ == '__main__':
    main()
