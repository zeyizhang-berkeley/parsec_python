"""Point-and-click front end for pseudopotential generation.

Start with ``python -m pp_generation gui`` (or ``python -m pp_generation.gui``).
Only the standard library is needed; the plot tab appears when matplotlib is
installed.  The window builds a :class:`GenerationRequest` and hands it to the
same :func:`generate` used by the command line.
"""

from __future__ import annotations

import math
import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PseudopotentialError
from .fhi_input import L_LABELS, parse
from .models import (
    BackendName,
    ConstructionScheme,
    CoreHole,
    GenerationRequest,
    GenerationResult,
    OutputFormat,
)
from .templates import available_elements, core_shells, template_text

# Rows of the periodic table; "" is an empty cell.
PERIODIC_ROWS = [
    "H . . . . . . . . . . . . . . . . He",
    "Li Be . . . . . . . . . . B C N O F Ne",
    "Na Mg . . . . . . . . . . Al Si P S Cl Ar",
    "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe",
    "Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
    "Fr Ra Ac . . . . . . . . . . . . . . .",
    ". . . Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu .",
    ". . . Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr .",
]

XC_CHOICES = (
    ("pbe", "PBE (GGA)"),
    ("ca", "CA (LDA, Perdew-Zunger)"),
    ("pw92", "PW92 (LDA)"),
)
FORMAT_CHOICES = (
    (OutputFormat.PARSEC, "PARSEC  Element_POTRE.DAT"),
    (OutputFormat.UPF, "UPF v2  (Quantum ESPRESSO, ARES)"),
    (OutputFormat.FHI, "FHI  Element_FHIPP.DAT (.cpi)"),
    (OutputFormat.ABINIT, "abinit  .fhi"),
)
AUTO = "auto"


@dataclass
class FormState:
    """Everything the window lets the user choose, as plain values."""

    element: str = "Si"
    xc: str = "pbe"
    scheme: str = "tm"
    core_hole_shell: str = ""          # "" = ground state
    hole_charge: str = "1.0"
    relativistic: bool = True
    lmax: str = AUTO
    radii: dict[int, str] = field(default_factory=dict)   # per channel, "" = default
    core_radius: str = ""              # "" = template, "0" = no core correction
    local_channel: str = AUTO
    formats: tuple[OutputFormat, ...] = (OutputFormat.PARSEC, OutputFormat.UPF, OutputFormat.FHI)
    output_dir: str = "pp-output"
    backend: str = BackendName.PYFHI98PP.value


def _number(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{what}: {text!r} is not a number") from None


def build_request(state: FormState) -> GenerationRequest:
    """Translate the form into a request; raises ``ValueError`` on bad entries."""

    if not state.formats:
        raise ValueError("select at least one output format")
    hole = None
    if state.core_hole_shell:
        hole = CoreHole(state.core_hole_shell, _number(state.hole_charge, "hole charge"))
    radii = {
        ell: _number(text, f"{L_LABELS[ell]} cutoff radius")
        for ell, text in state.radii.items() if text.strip()
    }
    label = state.element + (f"-{state.core_hole_shell}" if state.core_hole_shell else "")
    return GenerationRequest(
        element=state.element,
        output_dir=Path(state.output_dir).expanduser() / state.xc / label,
        backend=BackendName(state.backend),
        xc=state.xc,
        scheme=ConstructionScheme(state.scheme),
        core_hole=hole,
        formats=tuple(state.formats),
        local_channel=None if state.local_channel == AUTO else L_LABELS.index(state.local_channel),
        relativistic=state.relativistic,
        lmax=None if state.lmax == AUTO else int(state.lmax),
        channel_radii=radii or None,
        core_radius=_number(state.core_radius, "core radius") if state.core_radius.strip() else None,
    )


def summarize(result: GenerationResult) -> str:
    """Plain-text summary shown after a run (also handy from a script)."""

    request = result.request
    lines = [f"{result.prefix}: {request.element}, {request.xc.upper()}, {request.scheme.value.upper()}"]
    if request.core_hole:
        lines.append(
            f"core hole: {request.core_hole.removed_electrons:g} electron(s) removed from "
            f"{request.core_hole.shell}"
        )
    if result.ionic_charge is not None:
        lines.append(
            f"ionic charge {result.ionic_charge:g}, reference valence electrons "
            f"{result.reference_electrons:g}"
        )
    ae, ps = result.ae_energy_hartree, result.pseudo_energy_hartree
    if ae is not None and ps is not None:
        lines.append(f"all-electron total energy  {ae:16.6f} Ha")
        lines.append(f"pseudo-atom total energy   {ps:16.6f} Ha")
        lines.append(
            f"Atomic_Energy_Correction   {2.0 * (ae - ps):16.6f} Ry   (all-electron minus pseudo)"
        )
    if result.selected_local_channel is not None:
        lines.append(
            f"local channel: {L_LABELS[result.selected_local_channel]}   "
            f"ghost-free: {'yes' if result.ghost_free else 'NO'}"
        )
    for candidate in result.local_channel_results:
        margin = candidate.minimum_margin_hartree
        lines.append(
            f"  local {L_LABELS[candidate.local_channel]}: "
            f"{'pass' if candidate.passed else 'FAIL'}"
            + ("" if margin is None else f", smallest margin {margin:.4f} Ha")
        )
    if result.channels:
        lines.append("cutoff radii (bohr): " + "  ".join(
            f"{L_LABELS[x.angular_momentum]} {x.cutoff_radius:.3f}" for x in result.channels))
    if result.kinetic_cutoff_ry:
        ecut = max(result.kinetic_cutoff_ry.values())
        lines.append(
            f"kinetic energy converged to 1 meV at a plane-wave cutoff of {ecut:.0f} Ry "
            f"(equivalent grid spacing pi/sqrt(Ecut) = {math.pi / math.sqrt(ecut):.2f} bohr)"
        )
    lines.extend(f"note: {text}" for text in result.warnings)
    lines.append("files:")
    lines.extend(f"  {name}: {path}" for name, path in sorted(result.artifacts.items()))
    return "\n".join(lines)


def read_cpi_channels(path: Path) -> list[tuple[list[float], list[float], list[float]]]:
    """(r, u, V) per angular channel of a ``.cpi`` table, for the plots."""

    rows = path.read_text().splitlines()
    count = int(rows[0].split()[1])
    position, channels = 11, []
    for _ in range(count):
        size = int(rows[position].split()[0])
        block = [row.split() for row in rows[position + 1: position + 1 + size]]
        channels.append((
            [float(x[1]) for x in block], [float(x[2]) for x in block], [float(x[3]) for x in block],
        ))
        position += 1 + size
    return channels


def run_gui(auto_close_ms: int | None = None, auto_generate: bool = False,
            output_dir: str | None = None) -> int:  # pragma: no cover - needs a display
    """Open the window.  The arguments exist for smoke tests: press Generate once
    after start-up, write below ``output_dir``, and close after ``auto_close_ms``."""

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from .generator import generate

    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except Exception:  # matplotlib is optional
        FigureCanvasTkAgg = Figure = None

    root = tk.Tk()
    root.title("Pseudopotential generator (FHI98PP, Python)")
    root.minsize(1080, 720)
    available = set(available_elements())
    messages: queue.Queue = queue.Queue()

    element = tk.StringVar(value="Si")
    xc = tk.StringVar(value="pbe")
    scheme = tk.StringVar(value="tm")
    kind = tk.StringVar(value="ground")
    shell = tk.StringVar()
    hole_charge = tk.StringVar(value="1.0")
    relativistic = tk.BooleanVar(value=True)
    lmax = tk.StringVar(value=AUTO)
    local = tk.StringVar(value=AUTO)
    core_radius = tk.StringVar()
    out_dir = tk.StringVar(value=output_dir or str(Path.cwd() / "pp-output"))
    radii = {ell: tk.StringVar() for ell in range(4)}
    formats = {fmt: tk.BooleanVar(value=fmt is not OutputFormat.ABINIT) for fmt, _ in FORMAT_CHOICES}
    status = tk.StringVar(value="Choose an element, then Generate.")

    left = ttk.Frame(root, padding=8)
    left.grid(row=0, column=0, sticky="nsew")
    right = ttk.Frame(root, padding=8)
    right.grid(row=0, column=1, sticky="nsew")
    bottom = ttk.Notebook(root)
    bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=(0, 4))
    ttk.Label(root, textvariable=status, anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew", padx=8)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    # --- periodic table
    buttons: dict[str, tk.Button] = {}
    table = ttk.LabelFrame(left, text="Element", padding=6)
    table.pack(fill="both", expand=True)
    template_view = tk.Text(left, height=9, width=64, font=("Courier", 9))

    def refresh_element(*_):
        symbol = element.get()
        for name, button in buttons.items():
            button.configure(relief="sunken" if name == symbol else "raised",
                             bg="#ffd27f" if name == symbol else "#e8eef7")
        try:
            text = template_text(symbol, ConstructionScheme(scheme.get()))
            parsed = parse(text)
        except PseudopotentialError as exc:
            template_view.delete("1.0", "end")
            template_view.insert("1.0", str(exc))
            shell_box.configure(values=[])
            return
        shells = core_shells(parsed)
        shell_box.configure(values=shells)
        if shell.get() not in shells:
            shell.set(shells[0] if shells else "")
        if not shells:
            kind.set("ground")
        hole_radio.configure(state="normal" if shells else "disabled")
        lmax_box.configure(values=[AUTO] + [str(x) for x in range(4)])
        template_view.delete("1.0", "end")
        template_view.insert("1.0", text)
        status.set(f"{symbol}: template lmax = {parsed.lmax}, core shells: {' '.join(shells) or 'none'}")

    for row_index, row in enumerate(PERIODIC_ROWS):
        for column_index, symbol in enumerate(row.split()):
            if symbol == ".":
                continue
            enabled = symbol in available
            button = tk.Button(
                table, text=symbol, width=3, bg="#e8eef7",
                state="normal" if enabled else "disabled",
                command=lambda s=symbol: (element.set(s), refresh_element()),
            )
            button.grid(row=row_index + (1 if row_index >= 7 else 0), column=column_index, padx=1, pady=1)
            buttons[symbol] = button
    ttk.Label(left, text="Template (FHI98PP input) of the selected element").pack(anchor="w", pady=(6, 0))
    template_view.pack(fill="x")

    # --- settings
    box = ttk.LabelFrame(right, text="Settings", padding=8)
    box.pack(fill="both", expand=True)
    row = 0

    def label(text):
        nonlocal row
        ttk.Label(box, text=text).grid(row=row, column=0, sticky="w", pady=2)

    label("Functional")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="w")
    for value, text in XC_CHOICES:
        ttk.Radiobutton(frame, text=text, value=value, variable=xc).pack(side="left", padx=(0, 8))
    row += 1
    label("Scheme")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="w")
    for value, text in (("tm", "Troullier-Martins"), ("hamann", "Hamann")):
        ttk.Radiobutton(frame, text=text, value=value, variable=scheme, command=refresh_element).pack(side="left", padx=(0, 8))
    ttk.Checkbutton(frame, text="scalar-relativistic", variable=relativistic).pack(side="left")
    row += 1
    label("State")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="w")
    ttk.Radiobutton(frame, text="ground state", value="ground", variable=kind).pack(side="left")
    hole_radio = ttk.Radiobutton(frame, text="core hole in", value="hole", variable=kind)
    hole_radio.pack(side="left", padx=(8, 2))
    shell_box = ttk.Combobox(frame, textvariable=shell, width=4, state="readonly")
    shell_box.pack(side="left")
    ttk.Label(frame, text="electrons removed").pack(side="left", padx=(8, 2))
    ttk.Entry(frame, textvariable=hole_charge, width=5).pack(side="left")
    row += 1
    label("Channels")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="w")
    ttk.Label(frame, text="lmax").pack(side="left")
    lmax_box = ttk.Combobox(frame, textvariable=lmax, width=5, state="readonly", values=[AUTO, "0", "1", "2", "3"])
    lmax_box.pack(side="left", padx=(2, 8))
    ttk.Label(frame, text="local").pack(side="left")
    ttk.Combobox(frame, textvariable=local, width=5, state="readonly",
                 values=[AUTO, "s", "p", "d", "f"]).pack(side="left", padx=(2, 0))
    row += 1
    label("Cutoff radii (bohr)")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="w")
    for ell in range(4):
        ttk.Label(frame, text=L_LABELS[ell]).pack(side="left")
        ttk.Entry(frame, textvariable=radii[ell], width=6).pack(side="left", padx=(2, 8))
    ttk.Label(frame, text="core").pack(side="left")
    ttk.Entry(frame, textvariable=core_radius, width=6).pack(side="left", padx=(2, 0))
    row += 1
    ttk.Label(box, text="(empty = element default; core = radius of the nonlinear core correction, 0 switches it off)",
              foreground="#555").grid(row=row, column=1, sticky="w")
    row += 1
    label("Output formats")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="w")
    for index, (fmt, text) in enumerate(FORMAT_CHOICES):
        ttk.Checkbutton(frame, text=text, variable=formats[fmt]).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 10))
    row += 1
    label("Output folder")
    frame = ttk.Frame(box)
    frame.grid(row=row, column=1, sticky="ew")
    ttk.Entry(frame, textvariable=out_dir, width=46).pack(side="left")
    ttk.Button(frame, text="Browse", command=lambda: out_dir.set(
        filedialog.askdirectory(initialdir=out_dir.get()) or out_dir.get())).pack(side="left", padx=4)
    row += 1
    go = ttk.Button(box, text="Generate")
    go.grid(row=row, column=1, sticky="w", pady=(10, 0))

    # --- results
    summary = tk.Text(bottom, height=14, font=("Courier", 10))
    bottom.add(summary, text="Summary")
    ghost = ttk.Treeview(bottom, columns=("local", "channel", "verdict", "margin", "message"), show="headings", height=8)
    for name, width in (("local", 60), ("channel", 70), ("verdict", 110), ("margin", 110), ("message", 560)):
        ghost.heading(name, text={"local": "local", "channel": "projector", "verdict": "verdict",
                                  "margin": "margin (Ha)", "message": "FHI98PP analysis"}[name])
        ghost.column(name, width=width, anchor="w")
    bottom.add(ghost, text="Ghost-state test")
    figure = canvas = None
    if Figure is not None:
        plot_frame = ttk.Frame(bottom)
        figure = Figure(figsize=(9, 3.6), tight_layout=True)
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        bottom.add(plot_frame, text="Plots")
    log = tk.Text(bottom, height=14, font=("Courier", 9))
    bottom.add(log, text="Log")

    def collect() -> FormState:
        return FormState(
            element=element.get(), xc=xc.get(), scheme=scheme.get(),
            core_hole_shell=shell.get() if kind.get() == "hole" else "",
            hole_charge=hole_charge.get(), relativistic=relativistic.get(), lmax=lmax.get(),
            radii={ell: var.get() for ell, var in radii.items()}, core_radius=core_radius.get(),
            local_channel=local.get(),
            formats=tuple(fmt for fmt, var in formats.items() if var.get()),
            output_dir=out_dir.get(),
        )

    def show(result: GenerationResult):
        summary.delete("1.0", "end")
        summary.insert("1.0", summarize(result))
        ghost.delete(*ghost.get_children())
        for candidate in result.local_channel_results:
            if not candidate.channels:
                ghost.insert("", "end", values=(L_LABELS[candidate.local_channel], "-", "pass", "-",
                                                "no projector: purely local potential"))
            for channel in candidate.channels:
                ghost.insert("", "end", values=(
                    L_LABELS[candidate.local_channel], channel.label, channel.status.value,
                    "-" if channel.margin_hartree is None else f"{channel.margin_hartree:.5f}",
                    channel.message,
                ))
        cpi = result.artifacts.get("cpi")
        if figure is not None and cpi is not None and Path(cpi).is_file():
            figure.clear()
            left_axis, right_axis = figure.subplots(1, 2)
            for ell, (r, u, v) in enumerate(read_cpi_channels(Path(cpi))):
                left_axis.plot(r, u, label=L_LABELS[ell])
                right_axis.plot(r, v, label=L_LABELS[ell])
            left_axis.set(xlim=(0, 6), xlabel="r (bohr)", title="pseudo wavefunctions u(r)")
            right_axis.set(xlim=(0, 6), xlabel="r (bohr)", title="ionic pseudopotentials (Ha)")
            low = min(min(v) for _, _, v in read_cpi_channels(Path(cpi)))
            right_axis.set_ylim(max(low, -30.0) * 1.05, 2.0)
            left_axis.legend()
            right_axis.legend()
            canvas.draw()
        bottom.select(0)

    def worker(request: GenerationRequest):
        try:
            messages.put(("done", generate(request)))
        except (PseudopotentialError, ValueError, OSError) as exc:
            messages.put(("error", str(exc)))
        except Exception:  # unexpected: show the traceback instead of dying silently
            messages.put(("error", traceback.format_exc()))

    def start():
        try:
            request = build_request(collect())
        except ValueError as exc:
            messagebox.showerror("Input", str(exc))
            return
        go.configure(state="disabled")
        status.set(f"Generating {request.element} ({request.xc}) ... this takes from seconds to a minute.")
        log.insert("end", f"request: {request}\n")
        threading.Thread(target=worker, args=(request,), daemon=True).start()

    def poll():
        try:
            what, payload = messages.get_nowait()
        except queue.Empty:
            root.after(150, poll)
            return
        go.configure(state="normal")
        if what == "done":
            show(payload)
            status.set(f"Done: {payload.prefix} written to {payload.request.output_dir}")
            log.insert("end", summarize(payload) + "\n\n")
        else:
            status.set("Generation failed; see the Log tab.")
            log.insert("end", payload + "\n\n")
            bottom.select(log)
            messagebox.showerror("Generation failed", payload.strip().splitlines()[-1])
        root.after(150, poll)

    go.configure(command=start)
    refresh_element()
    root.after(150, poll)
    if auto_generate:
        root.after(400, start)
    if auto_close_ms is not None:
        root.after(auto_close_ms, root.destroy)
    root.mainloop()
    return 0


def main() -> int:  # pragma: no cover
    return run_gui()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
