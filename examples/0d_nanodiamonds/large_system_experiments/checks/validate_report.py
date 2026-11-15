"""Verify report arithmetic, scientific input/source hashes and local links.

This validates the published evidence package, not the underlying SCF accuracy.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def read(name):
    return json.loads((ROOT / 'data' / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def main():
    provenance = read('provenance.json')
    for name, digest in provenance['modified_source_files'].items():
        check(sha(REPO/name) == digest, 'Source differs from measured snapshot: ' + name)
    for item in read('inputs.json'):
        path = ROOT/item['file']
        check(sha(path) == item['sha256'], 'Input hash differs: ' + item['file'])
        for original, digest in item['input_hashes'].items():
            basename = Path(original).name
            if basename == 'parsec.in':
                check(digest == item['sha256'], 'Input lineage differs')
            else:
                base = '0d_nanodiamonds' if '/cluster/' in original else '3d_diamond_benchmark'
                check(sha(REPO/'examples'/base/basename) == digest, 'Pseudopotential differs: ' + original)
    runs = read('runs.json')['runs']
    ids = [r['record_id'] for r in runs]
    check(len(ids) == len(set(ids)), 'Repeated record ID')
    for r in runs:
        case = r['case_id']
        cluster = re.search(r'C(\d+)H(\d+)', case)
        periodic = re.search(r'^(?:gamma|k\d+)_(\d+)x(\d+)x(\d+)', case)
        if cluster:
            c,h = map(int,cluster.groups()); expected = 4*c+h; atoms = c+h
        elif periodic:
            a,b,c = map(int,periodic.groups()); expected = 32*a*b*c; atoms = 8*a*b*c
        else:
            raise ValueError('Unknown electron-count convention: '+case)
        check(expected == r['electrons'] and atoms == r['atoms'], 'Atom/electron count differs: '+case)
        if r.get('state') == 'accepted':
            check(r.get('converged') is True, 'Accepted row lacks complete SCF evidence: '+case)
        if r.get('peak_gpu_mib') is not None:
            check(r['peak_gpu_gib'] == r['peak_gpu_mib']/1024, 'Memory unit conversion differs')
    with (ROOT/'data/runs.csv').open() as f:
        check([r['record_id'] for r in csv.DictReader(f)] == ids, 'CSV and JSON inventory differ')
    pair = read('full_scf_memory_pair.json')
    check(abs((1-pair['optimized']['peak_gpu_mib']/pair['baseline']['peak_gpu_mib'])
              -pair['peak_memory_reduction_fraction']) < 1e-14, 'Memory reduction differs')
    result = read('large_restart_result.json')
    check(abs(result['result']['energy_ry']-result['parent_energy_ry']
              -result['energy_difference_from_parent_ry']) < 1e-12, 'Energy difference differs')
    check(result['production_adopted'] is False, 'This report has no production-adoption decision')
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            check((path.parent/target.split('#')[0]).exists(), 'Broken report link: '+str(path)+': '+target)
    # These infrastructure fields are deliberately excluded from public evidence.
    private = re.compile(r'hdfs://|merlin-ssh|seed\.byte|/Users/bytedance|/mnt/bn/|PRIVATE KEY|Bearer\s+[A-Za-z0-9]')
    for path in list((ROOT/'data').glob('*')) + list(ROOT.glob('*.md')):
        check(private.search(path.read_text()) is None, 'Private operational value in '+str(path))
    print(f'PASS: {len(runs)} run records, {len(read("inputs.json"))} inputs, '
          f'{len(provenance["modified_source_files"])} measured source files; links and arithmetic verified.')


if __name__ == '__main__':
    main()
