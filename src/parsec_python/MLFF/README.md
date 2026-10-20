# Initial structures from machine-learned force fields

A DFT calculation of `parsec_python` starts from the coordinates of the
input. With `Initial_Structure: mlff` it first relaxes them with a
machine-learned interatomic potential (UMA, MACE or any ASE calculator) and
starts from the relaxed structure instead. A relaxation, molecular dynamics
or a single point then begins close to the DFT minimum, which matters most
for structures built by hand or by a cheaper method.

## The quickest way

```text
# parsec.in
Initial_Structure: mlff
MLFF_Model: uma-s-1p2            # the default; or mace-mp-medium-mpa-0, my.model, ...
```

or, without editing the input,

```bash
python src/parsec_python/main.py calculation/parsec.in --mlff            # the default model
```

The same option exists for `reference_main.py` and for the multi-node GPU
runner (`acceleration/benchmarks/mpi_full_scf.py --mlff`, where the root rank
runs the force field and every rank takes its structure).

To look at the relaxed structure without a DFT run:

```bash
python src/parsec_python/mlff.py relax calculation/parsec.in
```

## Installing a model

The force field runs in its own process, with any Python that has NumPy,
ASE and the model package; it need not be the environment of
`parsec_python`. Name it with `MLFF_Python` in the input, `--python` of
`mlff.py`, or the variable `PARSEC_MLFF_PYTHON`; without them the running
interpreter is used.

| Model | `MLFF_Model` | Install | Notes |
|---|---|---|---|
| UMA (Meta FAIR) | `uma-s-1p2` (default), `uma-s-1p2p1`, `uma-s-1p1`, `uma-m-1p1`; eSEN, AllScAIP names of fairchem | `pip install fairchem-core` | Weights gated: accept the licence at <https://huggingface.co/facebook/UMA>, then `hf auth login` (or `HF_TOKEN`). `MLFF_Task: omol` for molecules and clusters. A local checkpoint `*.pt` works without the hub. |
| MACE-OMOL-0 | `mace-omol` | `pip install mace-torch` | Molecules, trained on OMol25 like UMA's `omol` task; Academic Software License. Failed for clusters of a few hundred atoms (below). |
| MACE-OFF23 | `mace-off-small`, `mace-off-medium`, `mace-off-large` | `pip install mace-torch` | Organic molecules (H C N O F P S Cl Br I); Academic Software License. |
| MACE-MP | `mace-mp-<name>`, e.g. `mace-mp-medium-mpa-0` | `pip install mace-torch` | Materials (PBE); names of `mace_mp`. |
| Your own MACE | `path/to/file.model` | `pip install mace-torch` | Relative to the input. |
| Anything else | `python:module:function` or `python:path/file.py:function` | | The function is called with `device`, `task`, `charge`, `spin` and returns an ASE calculator. |

A CUDA build of PyTorch puts the model on the GPU (`MLFF_Device: auto`); on
Windows the PyTorch of PyPI is CPU-only, take the wheel of
<https://download.pytorch.org/whl/> for CUDA. UMA runs without
`torch.compile`, which costs more than a relaxation of some tens of steps
saves; `PARSEC_MLFF_COMPILE=on` compiles it where the C++ compiler of the
platform is found (`g++`, or `cl` on Windows).

Put the environment on a fast file system. Imported from a project file
system on Perlmutter (CFS), fairchem took 40 to 110 s to import and the UMA
model 100 to 250 s to load; from scratch, 5 s and 14 s.

Weights are downloaded on first use (`~/.cache/fairchem`, about 1.1 GB for a
small UMA model; `~/.cache/mace`); a download that stalls for 300 s ends the
run (`PARSEC_MLFF_NETWORK_TIMEOUT`). On clusters whose compute nodes mount the
home directory without file locks (Perlmutter: `OSError: [Errno 524]`),
download them once on a login node with `mlff.py check MODEL`, or set
`FAIRCHEM_CACHE_DIR` to a scratch directory; a small home quota is another
reason to keep these caches on scratch. Check an installation with

```bash
python src/parsec_python/mlff.py models --python /path/to/env/bin/python
python src/parsec_python/mlff.py check uma-s-1p2 --python /path/to/env/bin/python
```

## Keywords

| Keyword | Default | Meaning |
|---|---|---|
| `Initial_Structure` | `input` | `mlff` relaxes the input structure with the force field first. |
| `MLFF_Model` | `uma-s-1p2` | The potential, see the table above. |
| `MLFF_Task` | `omol` | UMA task (`omol`, `omat`, `oc20`, `odac`, `omc`). |
| `MLFF_Fmax` | 0.05 eV/ang | The relaxation stops when the largest force is below it; units as `Force_Min` (eV/ang, Ry/bohr, Hartree/bohr), eV/ang when none is given. |
| `MLFF_Steps` | 500 | Largest number of optimizer steps. |
| `MLFF_Optimizer` | `lbfgs` | ASE optimizer: `lbfgs`, `bfgs` or `fire`. |
| `MLFF_Device` | `auto` | `cuda`, `cpu`, or `auto` (CUDA when PyTorch sees a GPU). |
| `MLFF_Python` | | Interpreter that has the model package; a name on the `PATH` or a path relative to the input. |
| `MLFF_Charge` | `Net_Charges` | Charge in elementary charges, for models that use it (UMA `omol`, MACE-OMOL). |
| `MLFF_Spin` | 1 or 2 | Spin multiplicity 2S+1; by default the smallest one of the electron count. |
| `MLFF_Reuse` | `true` | Take `mlff_structure.xyz` again when it was relaxed from the same input with the same settings. |

## What the run does

1. The input structure is centred as the DFT run centres it.
2. The point group the DFT calculation will use is kept: forces and
   positions are averaged over the operations of the structure that respect
   `Move_Flag`, so that the run keeps its symmetry sectors. Atoms that
   `Move_Flag` fixes stay where they are. Molecular dynamics and
   `Ignore_Symmetry` impose no symmetry, as in the DFT run.
3. The ASE optimizer relaxes the structure to `MLFF_Fmax`.
4. The relaxed coordinates replace those of the input, as an XYZ file of
   `--restart-from` would; the domain stays the one of the input.

It writes beside the input (the runner: into its output directory):

| File | Content |
|---|---|
| `mlff_structure.xyz` | The relaxed structure in angstrom, with the model, energy and largest force; `--restart-from` accepts it. |
| `mlff_relaxation.xyz` | Every optimizer step with its energy (eV) and forces (eV/ang), extended XYZ. |
| `mlff.log` | The table of the ASE optimizer. |

and `parsec.out` reports the model, the number of steps, energy and largest
force before and after, the largest displacement and the time.

The force field is not used when a restart file gives the structure
(`Relax_Restart`, `--restart-from`, `Restart_mode` of molecular dynamics) or
in a `manual` run, and the run says so. Periodic systems are refused.

## What it gains, and which model

Measured on the hydrogen-terminated nanodiamonds of `examples/0d_nanodiamonds`
(CA-LDA, BFGS to `Force_Min` 0.001 Ry/bohr with the SCF criterion 1e-6 Ry,
A100 GPUs, the model environments on scratch). The structures of the example
are GFN2-xTB minima; a crude start was made by expanding each by 3 %.

How close each model brings the expanded structure to the DFT minimum, RMSD
in angstrom, and its time for the whole relaxation, model loading included:

| Model | C35H36 | C185H124 | C795H300 | C1213H412 | Time |
|---|---:|---:|---:|---:|---:|
| `uma-s-1p2` (default) | 0.052 | 0.044 | 0.023 | 0.022 | 20 to 29 s |
| `uma-s-1p2p1` | 0.053 | 0.050 | 0.035 | 0.032 | 19 to 29 s |
| `uma-s-1p1` | 0.053 | 0.052 | 0.028 | 0.095 | 38 to 111 s |
| `uma-m-1p1` | 0.052 | 0.047 | out of memory | out of memory | |
| `mace-mp-medium-mpa-0` | 0.059 | 0.068 | 0.046 | 0.051 | 29 to 54 s |
| `mace-off-medium` | 0.051 | 0.141 | 0.326 | 0.392 | |
| `mace-omol` | 0.052 | refused | refused | out of memory | |
| (the xTB structure) | 0.045 | 0.038 | 0.023 | 0.028 | |
| (the expanded structure) | 0.129 | 0.190 | 0.257 | 0.284 | |

DFT structures to the criterion and the time on 4, 4, 8 and 16 GPUs (the
final energies of a cluster agree within 0.4 meV/atom):

| Cluster | xTB structure | expanded | expanded, then `uma-s-1p2` | time: expanded / `uma-s-1p2` + DFT |
|---|---:|---:|---:|---|
| C35H36 | 13 | 29 | 13 | 33 s / 30 + 22 s |
| C185H124 | 21 | 38 | 18 | 122 s / 19 + 57 s |
| C795H300 | 17 | 27 | 20 | 249 s / 35 + 114 s |
| C1213H412 | 22 | 31 | 16 | 291 s / 26 + 124 s |

- From a crude structure, `uma-s-1p2` takes the DFT run back to the number
  of steps of the xTB structure or fewer; for the larger clusters it saves
  37 to 49 % of the time. From a structure that is already good (the xTB
  minima) a force field changes little.
- The model matters. `mace-off-medium` contracted C795H300 by about 3.5 % and
  formed 1.33 angstrom C-C bonds at its surface, and the DFT run took more
  steps than without it (34 against 27); `mace-omol` gave forces of hundreds
  of eV/angstrom for C185H124 and larger; `uma-m-1p1` needed more than the
  80 GB of an A100 for C795H300.
- Without compilation and with the environment on a fast file system the
  force field costs some tens of seconds; on a project file system with
  `torch.compile`, several minutes.

A structure that a model does not describe is refused before the DFT run:
forces above 50 eV/angstrom or not finite, an energy that rises, or an atom
pushed out of the domain stop the run with a message that names other
models; the trajectory stays in `mlff_relaxation.xyz` for inspection.

## From Python

```python
from parsec_python.Input import parse_parsec_input
from parsec_python.MLFF import relax_initial_structure, apply_mlff_option

translation = apply_mlff_option(parse_parsec_input("calculation/parsec.in"), "uma-s-1p2")
translation, notes = relax_initial_structure(translation, "calculation")
print("\n".join(notes))
```
