#!/usr/bin/env python3
"""Cross-check the built libraries against the Fortran FHI98PP (optional; Linux or WSL).

For every row of the two manifests with status ok or unmatched: run the Fortran
``psgen`` on the library's own input file and compare the table with the library's
FHI file, then run the Fortran ``pswatch`` with the library's local channel and
compare the ghost verdicts with the library's report.  Standard library only.

    export OPENBLAS_NUM_THREADS=1     # one BLAS thread per job
    python3 crosscheck_fortran.py FORTRAN_ROOT LIBRARY_ROOT WORKDIR [jobs]

``FORTRAN_ROOT`` is the ``Dfhipp`` folder of a built FHI98PP (with ``bin/psgen`` and
``bin/pswatch``), ``LIBRARY_ROOT`` the folder holding ``NCPPs`` and
``CoreHole_NCPPs``, ``WORKDIR`` a scratch folder on a local filesystem.

For Z > 80 the unmodified Fortran writes outside the table of default radii in
``defrtm.f`` (``integer r(4,80)``) and can crash; declare it ``r(4,120)`` in the
build used here.  The ``pswatch`` script is run without ``-q`` because the test
program can die in its kinetic-energy test after printing the ghost analysis.
"""
import csv
import json
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

L = "spdf"
NUM = re.compile(r"[-+]?\d*\.\d+(?:[EeDd]?[-+]\d+)?")
NOEXP = re.compile(r"(?<=\d)([-+]\d{3})$")
CHANNEL_RE = re.compile(r"analysis of kb potentials:\s*([spdfgh]) waves", re.I)
RESULT_RE = re.compile(r"\*\s*(no ghost|one or more ghosts|undetermined|illdefined)", re.I)
STATUS = {"no ghost": "pass", "one or more ghosts": "ghost", "undetermined": "indeterminate",
          "illdefined": "ill_defined"}


def numbers(text):
    return [float(NOEXP.sub(lambda m: "E" + m.group(1), t).replace("D", "E").replace("d", "e"))
            for t in NUM.findall(text)]


def one(args):
    fortran, library, work, row = args
    directory = library / row["directory"]
    name = row["directory"].replace("/", "_")
    out = dict(name=name, state=row["state"], element=row["element"], xc=row["xc"])
    try:
        ini = next((directory / "diagnostics").glob("*.ini"))
        fhi = next((directory / "FHI").glob("*_FHIPP.DAT"))
        report = json.loads(next((directory / "diagnostics").glob("*.report.json")).read_text())
    except StopIteration:
        return dict(out, error="library files missing")
    w = work / name
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)
    (w / "pp.ini").write_text(ini.read_text())
    subprocess.run([str(fortran / "bin/psgen"), "-xv", "-o", "pp", "pp.ini"], cwd=w,
                   text=True, capture_output=True)
    cpi = w / "pp.cpi"
    dat = (w / "pp.dat").read_text(errors="replace") if (w / "pp.dat").is_file() else ""
    if "fhipp - done" not in dat or not cpi.is_file():
        return dict(out, error="fortran psgen failed")
    a, b = numbers(cpi.read_text()), numbers(fhi.read_text())
    if len(a) != len(b):
        return dict(out, error=f"table sizes differ {len(a)} {len(b)}")
    out["cpi_scaled"] = max((abs(x - y) / max(abs(x), 1.0) for x, y in zip(a, b)), default=0.0)
    e = re.findall(r"^\s*total energy\s+([-0-9.]+)", dat, re.M)
    out["dE_ae"] = abs(float(e[0]) - report["ae_energy_hartree"])
    out["dE_ps"] = abs(float(e[1]) - report["pseudo_energy_hartree"])
    lloc = report["selected_local_channel"]
    p = subprocess.run([str(fortran / "bin/pswatch"), "-xv", "-l", str(lloc), "-i", "pp", "pp.ini"],
                       cwd=w, text=True, capture_output=True)
    text = p.stdout
    chans = list(CHANNEL_RE.finditer(text))
    fortran_verdicts = {}
    for i, ch in enumerate(chans):
        end = chans[i + 1].start() if i + 1 < len(chans) else len(text)
        m = RESULT_RE.search(text, ch.end(), end)
        if m:
            fortran_verdicts[L.index(ch.group(1).lower())] = STATUS[m.group(1).lower()]
    mine = next(x for x in report["local_channel_results"] if x["local_channel"] == lloc)
    python_verdicts = {c["angular_momentum"]: c["status"] for c in mine["channels"]}
    out["verdicts"] = len(python_verdicts)
    out["verdicts_agree"] = fortran_verdicts == python_verdicts
    if not out["verdicts_agree"]:
        out["fortran"], out["python"] = fortran_verdicts, python_verdicts
    shutil.rmtree(w, ignore_errors=True)
    return out


if __name__ == "__main__":
    fortran, library, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    rows = []
    for name in ("NCPPs", "CoreHole_NCPPs"):
        with open(library / name / "manifest.csv", newline="") as stream:
            rows += [r for r in csv.DictReader(stream) if r["status"] in ("ok", "unmatched")]
    work.mkdir(parents=True, exist_ok=True)
    results = []
    with Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(fortran, library, work, row) for row in rows]), 1):
            results.append(r)
            if i % 200 == 0:
                print(i, "of", len(rows), file=sys.stderr)
    good = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    print(f"potentials checked {len(results)}; Fortran reproduced the generation in {len(good)}; errors {len(bad)}")
    if good:
        worst = max(good, key=lambda r: r["cpi_scaled"])
        print(f"table: worst |d|/max(|ref|,1) = {worst['cpi_scaled']:.2e} ({worst['name']}); "
              f"within 1e-8: {sum(r['cpi_scaled'] <= 1e-8 for r in good)}")
        print(f"energies: worst all-electron difference {max(r['dE_ae'] for r in good):.1e} Ha, "
              f"pseudo-atom {max(r['dE_ps'] for r in good):.1e} Ha (the Fortran log prints 1e-5)")
        print(f"ghost verdicts: {sum(r['verdicts'] for r in good)} compared, potentials with full agreement "
              f"{sum(r['verdicts_agree'] for r in good)} of {len(good)}")
    for r in bad[:15]:
        print("  error:", r["name"], r["error"])
    for r in [x for x in good if not x["verdicts_agree"]][:15]:
        print("  verdict mismatch:", r["name"], "fortran", r["fortran"], "python", r["python"])
    json.dump(results, open(work / "xcheck.json", "w"))
