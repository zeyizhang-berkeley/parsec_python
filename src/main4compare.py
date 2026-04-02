import json
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import scipy.sparse.linalg as spla
from scipy.io import loadmat
from Laplacian.fd3d import fd3d
from Laplacian.nuclear import nuclear
from V_ion.pseudoNL_original import pseudoNL
from V_ion.pseudoNL_ML4Den import pseudoNL_ML4Den
from V_xc.exc_nspn import exc_nspn
from V_ion.nelectrons import nelectrons
from Eigensolvers.first_filt import first_filt
from Eigensolvers.chefsi1 import chefsi1
from Eigensolvers.lanczos import lanczos
from Eigensolvers.chsubsp import chsubsp
from Eigensolvers.occupations import occupations
from Eigensolvers.pcg import pcg
from Mixer.mixer import mixer, reset_mixer
import matplotlib.pyplot as plt

# Variables and definitions
# A      = sparse matrix representing the discretization of the Laplacian
# nev    = number of eigenvalues - this is the number of occupied states
# Domain = struct containing information on the physical domain.
# Atoms  = struct containing information on the atoms.
# tol    = tolerance parameter for stopping scf iteration.
# maxtis = maximum number of SCF iterations allowed.
# fid    = output file id
# rho    = final charge density found
# lam    = eigenvalues computed - their number may be larger than nev
# W      = set of wave functions.

# Defaut Technical Parameters
fd_order = 8            # order of finite difference scheme {8}
maxits   = 40           # max SCF iterations                {40}
tol      = 1.e-05       # tolerance for SCF iteration.      {1.e-03}
Fermi_temp  =  500.0    # Smear out fermi level             {500.0}
A0_ANG = 0.529177210903  # 1 Bohr in Angstrom
last_input_file_path = None

# Global variables
CG_prec = 0             # whether to precondition CG
poldeg = 10             # polynomial degree for chebyshev
diagmeth = 3            # method for diagonalization
'''
diagmeth == 0 --> Lanczos 1st step and chebyshev filtering thereafter
diagmeth == 1 --> Lanczos all the time
diagmeth == 2 --> Full-Chebyshev subspace iteration first step chebyshev filtering thereafter.
diagmeth == 3 --> First step filtering of random initial vectors, then chebyshev subspace filtering thereafter
                  (this uses less memory and faster than diagonalization)
'''
adaptiveScheme = 0
'''
adaptiveScheme == 0    Do not use an adaptive scheme
adaptiveScheme == 1    Allow for the changing of parameters used by lanczos and chefsi1 to increase speed, 
                       slight risk of longer time to converge
'''

Ry = 13.605698066  # Unit in eV

# imports a bunch of common flag settings (TODO: implement other things later)
def RSDFTsettings():
    pass  # Implement any settings logic here

# Check validity of global variables
def check_global_variables():
    global CG_prec, poldeg, diagmeth, adaptiveScheme

    if (CG_prec != 0) and (CG_prec != 1):
        CG_prec = 0
    if poldeg < 1:
        poldeg = 1
    if (adaptiveScheme != 0) and (adaptiveScheme != 1):
        adaptiveScheme = 0
    if (diagmeth < 0) or (diagmeth > 3):
        diagmeth = 3

# Call the RSDFT settings and validity check
RSDFTsettings()
check_global_variables()

DEFAULT_SETTINGS = {
    'fd_order': fd_order,
    'maxits': maxits,
    'tol': tol,
    'Fermi_temp': Fermi_temp,
    'CG_prec': CG_prec,
    'poldeg': poldeg,
    'diagmeth': diagmeth,
    'adaptiveScheme': adaptiveScheme,
}


def apply_settings_from_dict(settings_overrides):
    """
    Override global solver settings using the provided dictionary.
    Only known keys are applied; unknown keys are ignored.
    """
    if not settings_overrides:
        return {}

    global fd_order, maxits, tol, Fermi_temp, CG_prec, poldeg, diagmeth, adaptiveScheme

    applied = {}
    def _parse_tol(val):
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        # Allow integer-like shorthand: 5 -> 1e-5
        if fval >= 1 and abs(fval - round(fval)) < 1e-12:
            return 10 ** (-int(round(fval)))
        return fval

    mapping = {
        'fd_order': ('fd_order', int),
        'maxits': ('maxits', int),
        'tol': ('tol', _parse_tol),
        'Fermi_temp': ('Fermi_temp', float),
        'fermi_temp': ('Fermi_temp', float),
        'CG_prec': ('CG_prec', int),
        'cg_prec': ('CG_prec', int),
        'poldeg': ('poldeg', int),
        'diagmeth': ('diagmeth', int),
        'adaptiveScheme': ('adaptiveScheme', int),
        'adaptive_scheme': ('adaptiveScheme', int),
    }

    for key, (target, caster) in mapping.items():
        if key in settings_overrides and settings_overrides[key] is not None:
            try:
                new_value = caster(settings_overrides[key])
                if target == 'fd_order':
                    fd_order = new_value
                elif target == 'maxits':
                    maxits = new_value
                elif target == 'tol':
                    tol = new_value
                elif target == 'Fermi_temp':
                    Fermi_temp = new_value
                elif target == 'CG_prec':
                    CG_prec = new_value
                elif target == 'poldeg':
                    poldeg = new_value
                elif target == 'diagmeth':
                    diagmeth = new_value
                elif target == 'adaptiveScheme':
                    adaptiveScheme = new_value
                applied[target] = new_value
            except (TypeError, ValueError):
                print(f'Warning: Could not parse value for {key}, keeping default.')

    check_global_variables()
    if applied:
        print('Applied solver setting overrides:')
        for name, value in applied.items():
            print(f'  {name}: {value}')
    return applied


def _prompt_value_with_default(prompt_text, cast_func, current_value):
    """Utility for interactive overrides."""
    raw = input(f'{prompt_text} [{current_value}]: ').strip()
    if raw == '':
        return None
    try:
        return cast_func(raw)
    except ValueError:
        print('  Invalid input, keeping current value.')
        return None


def prompt_for_settings_overrides():
    """
    Optional interactive prompts for overriding solver/grid defaults
    when the user is providing manual input.
    """
    print('\n ------------------------')
    choice = input('Override default solver/grid settings? (y/N): ').strip().lower()
    if choice not in ['y', 'yes']:
        return {}

    overrides = {}
    overrides['tol'] = _prompt_value_with_default('SCF tolerance', float, tol)
    overrides['maxits'] = _prompt_value_with_default('Max SCF iterations', int, maxits)
    overrides['fd_order'] = _prompt_value_with_default('Finite-difference order', int, fd_order)
    overrides['Fermi_temp'] = _prompt_value_with_default('Fermi temperature (K)', float, Fermi_temp)
    overrides['poldeg'] = _prompt_value_with_default('Polynomial degree for Chebyshev', int, poldeg)
    overrides['diagmeth'] = _prompt_value_with_default('Diagonalization method (0-3)', int, diagmeth)
    overrides['CG_prec'] = _prompt_value_with_default('Use CG preconditioner? (0/1)', int, CG_prec)
    overrides['adaptiveScheme'] = _prompt_value_with_default('Use adaptive scheme? (0/1)', int, adaptiveScheme)

    # Geometry / eigenvalue-related knobs (optional)
    overrides['nev'] = _prompt_value_with_default('Number of eigenvalues (blank = auto)', int, 'auto')
    overrides['grid_spacing'] = _prompt_value_with_default('Grid spacing h [Bohr] (blank = auto)', float, 'auto')
    overrides['sphere_radius'] = _prompt_value_with_default('Sphere radius [Bohr] (blank = auto)', float, 'auto')

    # Remove Nones to keep downstream logic simple
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return overrides


# Load element information from CSV file
def load_elements():
    """
    Load the element data from 'elements_new.csv' assuming there is no header,
    and that the first column contains the element type.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'elements_new.csv')
    # Load the CSV file without headers
    elem = pd.read_csv(csv_path, header=None)

    # Assign column names (assuming we know the structure)
    elem.columns = ['Element', 'AtomicNumber', 'Z', 'Zvalue', 'h', 'r', 'R'] + [f'Property{i}' for i in range(1, 5)]
    # TODO: Add more descriptive names for columns if known
    N_elements = elem.shape[0]  # This gives the number of elements (rows)
    # print(N_elements)
    return elem, N_elements

# Call the function to load the element data
elem, N_elements = load_elements()

def element_exists(typ, elem):
    """Check if the element exists in the element data."""
    return typ in elem['Element'].values  # Assuming elem is a pandas DataFrame

# Main function to get all the settings for RSDFT (manual or file)
def get_atoms(elem):
    global last_input_file_path
    # Ask the user for input mode
    print(' ********************** ')
    print('  DATA INPUT FOR RSDFT  ')
    print(' ********************** ')
    print('------------------------')

    # Needed variables
    in_data = 0
    Atoms = []
    n_atom = []
    Z_charge = 0.0
    density_method = "sad"
    ml_file_path = None
    settings_overrides = {}

    while (in_data != 1) and (in_data != 2):
        try:
            in_data = int(input('Input data mode: 1 for manual, 2 for file: '))
        except ValueError:
            print('Please enter 1 or 2')

    # Case 1: Manual input
    if in_data == 1:
        last_input_file_path = None
        Atoms, n_atom, Z_charge, density_method, ml_file_path = manual_input_species(elem)
        if n_atom is None or Atoms is None:
            return None, None, None, None, None, {}
        settings_overrides = prompt_for_settings_overrides()
        
    # Case 2: File input
    elif in_data == 2:
        Atoms, n_atom, Z_charge, density_method, ml_file_path, settings_overrides = load_atomic_data_from_file()
        if n_atom is None or Atoms is None:
            return None, None, None, None, None, {}

    # Normalize density settings
    if density_method not in ['sad', 'ml']:
        density_method = 'sad'

    return Atoms, n_atom, Z_charge, density_method, ml_file_path, settings_overrides

def manual_input_species(elem):
    """Handles manual input for atomic species."""
    # Input number of atomic species
    at = 0
    while at < 1: # must have at least one element
        try:
            at = int(input('Input number of different atomic species: '))
            if at < 1:
                print('Please enter a number >= 1')
        except ValueError:
            print('Invalid input. Please enter a valid number.')
    Atoms = []  # atom information for each element
    n_atom = [] # number of atoms for each element
    # Loop through each atomic species
    typ = ''
    for i in range(at):
        correctElementName = False
        while not correctElementName:
            typ = input('Element of species, e.g., Mg, C, O, only first 18 elements supported: ')
            correctElementName = element_exists(typ, elem)
            if not correctElementName:
                print('    Element not recognized')

        print('  Coordinates should be in atomic units ')
        print('  Example: atoms at (0,0,0) should be entered as 0 0 0 on each line ')
        print('  Terminate with /, i.e., 0 0 0 / for the last entry ')

        # Input atomic coordinates
        errorFreeCoordinates = 0
        xyz = []
        while (errorFreeCoordinates == 0):
            readxyz = ''  # Initialize an empty string to collect all input
            while '/' not in readxyz:
            # Keep appending input to the string until '/' is encountered
                readxyz += ' ' + input('  Input coordinates: ')

            # Process the string: remove the '/' and convert to a list of numbers
            try:
                # Split by spaces and remove any empty strings caused by multiple spaces
                coordinates = [float(coord) for coord in readxyz.split('/')[-2].strip().split()]

                # Ensure the number of coordinates is not zero and divisible by 3
                if (len(coordinates) == 0) or (len(coordinates) % 3 != 0):
                    print('Error: The number of coordinates must be nonzero and divisible by 3!')
                    return None, None, None, None, None
                else:
                    # Reshape the list into (n, 3) NumPy array
                    xyz = np.array(coordinates).reshape(-1, 3)
                    errorFreeCoordinates = 1

            except ValueError:
                print('Error: Invalid input detected. Please ensure all coordinates are numbers.')
                return None, None, None, None, None
        # Store the number of atoms and the atomic data
        n_atom.append(xyz.shape[0])
        Atoms.append({'typ': typ, 'coord': xyz})

    # Input charge information
    try:
        Z_charge = float(input(' How many electrons should be added/removed from the system? '))
        # Changing from number of electrons to charge (negative for added electrons)
        Z_charge = -Z_charge
    except ValueError:
        print('Error: Invalid input detected. Use default Z_charge = 0')
        Z_charge = 0

    # ADD DENSITY METHOD CHOICE HERE
    print('\n ********************** ')
    print(' DENSITY INITIALIZATION METHOD')
    print(' ********************* ')
    print('------------------------')

    method_choice = 0
    while method_choice not in [1, 2]:
        try:
            print('Choose density initialization method:')
            print('1. Superposition of atomic densities (traditional) but still use the ML grids')
            print('2. Machine learning predicted density (.npy file)')
            method_choice = int(input('Enter your choice (1 or 2): '))
        except ValueError:
            print('Please enter 1 or 2')

    if method_choice == 1:
        density_method = 'sad'
        ml_file_path = input('Enter path to ML density .npy file: ')
    else:
        density_method = 'ml' # Get ML density file path
        ml_file_path = input('Enter path to ML density .npy file: ')

    # for Drew: Add manual input for doing CDFT? return a bool variable?(at least for a simple test, can add other options)

    return Atoms, n_atom, Z_charge, density_method, ml_file_path


def load_atomic_data_from_file():
    """
    Handles file input for atomic species and optional solver/settings overrides.
    Supported formats:
      - .dat : simple atom list (legacy)
      - .mat : expects variable AtomsInMolecule with fields typ, coord
      - .json: atoms plus optional settings (preferred)
      - .in/ .inp : custom format with $system and $settings blocks
    """
    global last_input_file_path
    file_name = input('What is the name of the file to load from?: ').strip()
    last_input_file_path = None

    # Check if the file name is too short
    if len(file_name) <= 4:
        print('File name is too short, must include file extension')
        return None, None, None, None, None, {}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        file_name,
        os.path.join('SavedMolecules', file_name),
        os.path.join(base_dir, file_name),
        os.path.join(base_dir, 'SavedMolecules', file_name),
    ]
    file_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            file_path = path
            break

    if file_path is None:
        print(f'File not found: tried {candidate_paths}')
        return None, None, None, None, None, {}

    last_input_file_path = file_path

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    Atoms = []
    n_atom = []
    Z_charge = 0.0
    density_method = 'sad'
    ml_file_path = None
    settings_overrides = {}
    unit_setting = None

    def _parse_atoms_from_lines(lines):
        grouped = {}
        order = []
        for raw in lines:
            line = raw.strip()
            if line == '' or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 4:
                print('Atom line must be: TYPE x y z')
                return None, None
            typ = parts[0]
            try:
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                print('Could not parse coordinates; must be numeric.')
                return None, None
            if typ not in grouped:
                grouped[typ] = []
                order.append(typ)
            grouped[typ].append(coords)

        Atoms_local = []
        n_atom_local = []
        for typ in order:
            coord_arr = np.array(grouped[typ], dtype=float)
            Atoms_local.append({'typ': typ, 'coord': coord_arr})
            n_atom_local.append(coord_arr.shape[0])
        return Atoms_local, n_atom_local

    def _prune_empty(overrides_dict):
        if not overrides_dict:
            return {}
        cleaned = {}
        for k, v in overrides_dict.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == '':
                continue
            cleaned[k] = v
        return cleaned

    # Handle .dat files
    if file_extension == '.dat':
        try:
            with open(file_path, 'r') as file:
                at = int(file.readline().strip())
                for _ in range(at):
                    typ = file.readline().strip()
                    readxyz = np.array([float(x) for x in file.readline().strip().split()])
                    if len(readxyz) % 3 != 0:
                        print('Coordinate count in .dat must be divisible by 3.')
                        return None, None, None, None, None, {}
                    n_at = len(readxyz) // 3
                    xyz = np.reshape(readxyz, (n_at, 3))
                    n_atom.append(n_at)
                    Atoms.append({'typ': typ, 'coord': xyz})
        except FileNotFoundError:
            print(f"File not found: {file_path}.")
            return None, None, None, None, None, {}
        except ValueError:
            print(f"Could not parse .dat file: {file_path}")
            return None, None, None, None, None, {}

    # Handle .mat files
    elif file_extension == '.mat':
        # Load .mat file
        mat_data = loadmat(file_path)

        if 'AtomsInMolecule' in mat_data:
            Atoms, n_atom = AtomsInMoleculeToAtomsConverter(mat_data['AtomsInMolecule'])
        else:
            print('File does not contain the correct variable: AtomsInMolecule')
            return None, None, None, None, None, {}

    # Handle simple text files with lines: "TYPE x y z"
    elif file_extension == '.txt':
        with open(file_path, 'r') as f:
            lines = f.readlines()
        Atoms, n_atom = _parse_atoms_from_lines(lines)
        if Atoms is None:
            return None, None, None, None, None, {}

    # Handle .in / .inp files with sections
    elif file_extension in ['.in', '.inp']:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        section = None
        atom_lines = []
        settings_lines = []
        for raw in lines:
            line = raw.strip()
            if line == '' or line.startswith('#'):
                continue

            token = line.lower()
            if token == '$system':
                section = 'system' if section != 'system' else None
                continue
            if token == '$settings':
                section = 'settings' if section != 'settings' else None
                continue

            if section == 'system':
                atom_lines.append(line)
            elif section == 'settings':
                settings_lines.append(line)
            else:
                # ignore lines outside recognized sections
                continue

        Atoms, n_atom = _parse_atoms_from_lines(atom_lines)
        if Atoms is None:
            return None, None, None, None, None, {}

        # Parse settings block: key = value, blank values mean default
        for entry in settings_lines:
            if '=' not in entry:
                continue
            key, val = entry.split('=', 1)
            key = key.strip()
            val = val.strip()
            if val == '':
                continue

            key_lower = key.lower()
            if key_lower in ['z_charge', 'charge']:
                try:
                    Z_charge = float(val)
                except ValueError:
                    print(f'Warning: could not parse Z_charge "{val}", using default 0.')
            elif key_lower in ['density_method', 'density']:
                density_method = val.lower()
            elif key_lower in ['ml_file_path', 'density_file']:
                ml_file_path = val
            elif key_lower in ['unit']:
                unit_setting = val
            else:
                settings_overrides[key] = val

        # Resolve ML density path relative to the input file directory if needed
        if ml_file_path:
            ml_file_path = os.path.expanduser(ml_file_path)
            if not os.path.isabs(ml_file_path):
                ml_file_path = os.path.join(os.path.dirname(file_path), ml_file_path)
        settings_overrides = _prune_empty(settings_overrides)

    # Handle .json files
    elif file_extension == '.json':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f'Could not read JSON: {exc}')
            return None, None, None, None, None, {}

        atoms_block = data.get('atoms') or data.get('Atoms')
        if not atoms_block:
            print('JSON file missing "atoms" list.')
            return None, None, None, None, None, {}

        # Allow "atoms" as a block of lines (string) or list of strings
        if isinstance(atoms_block, str):
            Atoms, n_atom = _parse_atoms_from_lines(atoms_block.strip().splitlines())
            if Atoms is None:
                return None, None, None, None, None, {}
        elif isinstance(atoms_block, list) and len(atoms_block) > 0 and isinstance(atoms_block[0], str):
            Atoms, n_atom = _parse_atoms_from_lines(atoms_block)
            if Atoms is None:
                return None, None, None, None, None, {}
        else:
            for entry in atoms_block:
                typ = entry.get('typ') or entry.get('type') or entry.get('element')
                coords = entry.get('coord') or entry.get('coords') or entry.get('coordinates')
                if typ is None or coords is None:
                    print('Each atom entry must have typ and coord/coords.')
                    return None, None, None, None, None, {}

                typ = str(typ)
                coord_arr = np.asarray(coords, dtype=float)
                coord_arr = np.reshape(coord_arr, (-1, 3))
                n_atom.append(coord_arr.shape[0])
                Atoms.append({'typ': typ, 'coord': coord_arr})

        try:
            Z_charge = float(data.get('Z_charge', data.get('charge', 0.0)))
        except (TypeError, ValueError):
            print('Warning: invalid Z_charge in JSON, defaulting to 0.')
            Z_charge = 0.0
        density_method = data.get('density_method', density_method)
        if isinstance(density_method, str):
            density_method = density_method.lower()
        else:
            density_method = 'sad'
        if density_method not in ['sad', 'ml']:
            density_method = 'sad'

        ml_file_path = data.get('ml_file_path') or data.get('density_file')
        if ml_file_path:
            ml_file_path = os.path.expanduser(ml_file_path)
            if not os.path.isabs(ml_file_path):
                ml_file_path = os.path.join(os.path.dirname(file_path), ml_file_path)

        settings_overrides = data.get('settings', data.get('params', {})) or {}
        if isinstance(settings_overrides, dict):
            unit_setting = settings_overrides.get('unit', unit_setting)
        # Also accept top-level override keys for convenience
        for key in ['nev', 'grid_spacing', 'h', 'sphere_radius', 'radius']:
            if key in data and key not in settings_overrides:
                settings_overrides[key] = data[key]
        settings_overrides = _prune_empty(settings_overrides)
        if unit_setting:
            settings_overrides['unit'] = unit_setting

    # Handle unrecognized file extensions
    else:
        print('File extension not recognized (supported: .dat, .mat, .json, .txt, .in /.inp)')
        return None, None, None, None, None, {}

    if unit_setting:
        settings_overrides['unit'] = unit_setting

    return Atoms, n_atom, Z_charge, density_method, ml_file_path, settings_overrides


def AtomsInMoleculeToAtomsConverter(AtomsInMolecule):
    """Convert data from .mat file format to usable Python structures."""
    Atoms = []
    n_atom = []

    flat_atoms = np.array(AtomsInMolecule).ravel()
    for atom in flat_atoms:
        typ = None
        coord = None

        if isinstance(atom, dict):
            typ = atom.get('typ') or atom.get('type') or atom.get('element')
            coord = atom.get('coord') or atom.get('coords') or atom.get('coordinates')
        elif isinstance(atom, np.void) and atom.dtype.fields:
            if 'typ' in atom.dtype.fields:
                typ = atom['typ']
            if 'coord' in atom.dtype.fields:
                coord = atom['coord']
        else:
            try:
                typ = atom['typ']
                coord = atom['coord']
            except Exception:
                pass

        if typ is None or coord is None:
            print('AtomsInMolecule entry missing typ or coord.')
            return [], []

        if isinstance(typ, np.ndarray):
            typ = typ.item()
        typ = str(typ)

        coord = np.asarray(coord, dtype=float)
        try:
            coord = np.reshape(coord, (-1, 3))
        except ValueError:
            print('Could not reshape coordinates to (N,3) in .mat file.')
            return [], []
        n_atom.append(coord.shape[0])
        Atoms.append({'typ': typ, 'coord': coord})

    return Atoms, n_atom

# Usage example:
# Assuming elem is a pandas DataFrame loaded with element data from 'elements_new.csv'
Atoms, n_atom, Z_charge, density_method, ml_file_path, settings_overrides = get_atoms(elem)
if not Atoms:
    raise SystemExit('No atoms loaded, exiting RSDFT')

print('Atoms', Atoms)
print('n_atom', n_atom)
print('Z_charge', Z_charge)

# Apply unit conversion before using overrides
settings_overrides = settings_overrides or {}

def has_diagmeth_override(overrides):
    """Check whether the user supplied a diagonalization method override."""
    return any(str(k).lower() == 'diagmeth' for k in (overrides or {}))


def resolve_ml_density_path(ml_path):
    """Return an absolute path to the ML density file if it exists, else None."""
    if not ml_path:
        return None

    ml_path = os.path.expanduser(ml_path)
    candidates = []
    if os.path.isabs(ml_path):
        candidates.append(ml_path)
    else:
        candidates.append(os.path.abspath(ml_path))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(base_dir, ml_path))
        if last_input_file_path:
            candidates.append(os.path.join(os.path.dirname(last_input_file_path), ml_path))

    for path in candidates:
        if os.path.exists(path):
            return path

    print(f'Warning: ML density file {ml_path} not found. Tried: {candidates}')
    return None

ml_file_path = resolve_ml_density_path(ml_file_path)
if ml_file_path is None:
    raise SystemExit('ML density file is required to set grid spacing and domain; please provide a valid .npy path.')

def convert_units_to_bohr(Atoms, settings_overrides):
    """Convert coordinates and certain overrides to Bohr based on 'unit'."""
    defaults_to_ang = last_input_file_path is not None
    unit_raw = settings_overrides.get('unit')
    if unit_raw is None:
        unit_raw = 'ang' if defaults_to_ang else 'bohr'
    unit_norm = str(unit_raw).lower()
    is_ang = unit_norm in ['ang', 'angstrom', 'a']
    is_bohr = unit_norm in ['bohr', 'au', 'a.u.']

    factor = 1.0
    if is_ang:
        factor = 1.0 / A0_ANG
    elif is_bohr:
        factor = 1.0
    else:
        # Unknown unit: assume default (Ang if file input, Bohr if manual)
        factor = 1.0 / A0_ANG if defaults_to_ang else 1.0

    if factor != 1.0:
        for atom in Atoms:
            atom['coord'] = atom['coord'] * factor

    # Convert overrides (grid spacing, radius) if present
    converted_overrides = settings_overrides.copy()
    for key in ['grid_spacing', 'h', 'sphere_radius', 'radius']:
        if key in converted_overrides:
            try:
                converted_overrides[key] = float(converted_overrides[key]) * factor
            except (TypeError, ValueError):
                pass

    # Remove unit from overrides to avoid confusion downstream
    converted_overrides.pop('unit', None)
    return Atoms, converted_overrides, unit_norm

Atoms, settings_overrides, unit_used = convert_units_to_bohr(Atoms, settings_overrides)

# If using ML initialization and no explicit override, prefer diagmeth=2 (chsubsp)
if (density_method == 'ml') and (not has_diagmeth_override(settings_overrides)):
    diagmeth = 2

# Apply any solver overrides provided via manual input or file
apply_settings_from_dict(settings_overrides)

# Determine output file name based on input file (if provided)
method_tag = str(density_method).lower() if density_method else 'sad'
if method_tag not in ['sad', 'ml']:
    method_tag = 'sad'

method_suffix = f"{method_tag}_diagmeth{diagmeth}"
output_file = f'./rsdft_parameter_{method_suffix}.out'
if last_input_file_path:
    base = os.path.splitext(os.path.basename(last_input_file_path))[0]
    out_dir = os.path.dirname(last_input_file_path) or '.'
    output_file = os.path.join(out_dir, f'{base}_{method_suffix}.out')

# Keep a raw copy of the original coordinates for ML shifting
# Atoms_raw = [ { 'typ': atom['typ'], 'coord': atom['coord'].copy() } for atom in Atoms ]
# print(Atoms,n_atom,Z_charge)
print(f"Successfully loaded {len(Atoms)} atomic species.")


def calculate_grid_spacing(Atoms, elem, N_elements, n_atom, Z_charge, ml_file_path, h_override=None, nev_override=None):
    """
    Calculate the smallest grid spacing hmin and the number of eigenvalues (nev).

    Parameters:
    Atoms (list of dictionaries): List of atomic species with type and coordinates.
    elem (pd.DataFrame): DataFrame containing element information (including grid spacing).
    n_atom (list): List containing the number of atoms for each species.
    Z_charge (float): Charge state of the system.
    h_override (float, optional): User-specified grid spacing.
    nev_override (int, optional): User-specified number of eigenvalues.

    Returns:
    hmin (float): Smallest grid spacing.
    nev (int): Number of eigenvalues (number of states).
    """
    N_types = len(Atoms)

    zelec = 0.0
    hmin = 100.0  # Large initial value to find the minimum grid spacing

    if ml_file_path is None or not os.path.exists(ml_file_path):
        raise SystemExit('ML density file is required to determine grid spacing.')

    # Always use the ML grid spacing for consistency across SAD and ML densities
    density_ang3 = np.load(ml_file_path)
    A0_ANG = 0.529177210903  # 1 Bohr = 0.529177210903 Å
    nx_ml, ny_ml, nz_ml = density_ang3.shape
    box_length_ml_ang = 10.0
    grid_spacing_ml = float(box_length_ml_ang) / float(nx_ml)
    hmin = grid_spacing_ml / A0_ANG

    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']  # Get the atomic symbol

        # Look for matching element data in the elem DataFrame
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                Z = elem['Z'].iloc[i] * n_atom[at_typ]  # Number of electrons for this species
                zelec += Z  # Add the electrons from this species

    # Check for valid electron count
    ztest = zelec - Z_charge
    if ztest < 0:
        print('Problem with charge state. Negative number of electrons.')
        return None, None, None, None

    # Calculate the smallest grid spacing and number of eigenvalues
    if h_override is not None:
        try:
            h = float(h_override)
        except (TypeError, ValueError):
            print('Warning: invalid grid spacing override, using auto value.')
            h = hmin
    else:
        h = hmin

    if nev_override is not None:
        try:
            nev = max(1, int(nev_override))
        except (TypeError, ValueError):
            print('Warning: invalid nev override, using auto value.')
            nev = max(16, round(0.7 * zelec + 0.5))
    else:
        nev = max(16, round(0.7 * zelec + 0.5))  # At least 16 eigenvalues are required

    return h, nev, zelec, ztest

nev_override = settings_overrides.get('nev')
h_override = settings_overrides.get('grid_spacing', settings_overrides.get('h'))
radius_override = settings_overrides.get('sphere_radius', settings_overrides.get('radius'))

h, nev, zelec, ztest = calculate_grid_spacing(
    Atoms, elem, N_elements, n_atom, Z_charge, ml_file_path,
    h_override=h_override, nev_override=nev_override
)

if h is None or nev is None:
    raise SystemExit('Failed to determine grid spacing or number of states.')

if h is not None and nev is not None:
    print(f"Grid spacing (h): {h}")
    print(f"Number of eigenvalues (nev): {nev}")


def estimate_radius_and_grid(Atoms, elem, N_elements, h, ml_file_path, radius_override=None):
    """
    Estimate the spherical radius and calculate grid sizes based on atom positions.

    Parameters:
    Atoms (list of dictionaries): List of atomic species with type and coordinates.
    elem (pd.DataFrame): DataFrame containing element information (including atomic size).
    h (float): Grid spacing (smallest h from previous calculations).
    radius_override (float, optional): User-specified spherical radius.

    Returns:
    Domain (dict): A dictionary containing grid size information and the spherical radius.
    """
    xyz = []
    rmax = 0.0  # Initialize max radius
    natoms = 0.0  # Initialize number of atoms
    rsize = 0.0  # Initialize radius size
    sph_rad = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0
    N_types = len(Atoms)  # Number of atomic species

    if ml_file_path is None or not os.path.exists(ml_file_path):
        raise SystemExit('ML density file is required to determine domain radius and grid.')

    # Always use the ML grid size and implied box for the domain
    density_ang3 = np.load(ml_file_path)
    A0_ANG = 0.529177210903  # 1 Bohr = 0.529177210903 Å
    nx_ml, ny_ml, nz_ml = density_ang3.shape
    nx = nx_ml
    ny = ny_ml
    nz = nz_ml
    box_length_ml_ang = 10.0
    sph_rad = ( box_length_ml_ang / 2 ) / A0_ANG

    if radius_override is not None:
        try:
            sph_rad = float(radius_override)
            nx = int(2 * sph_rad / h) + 1
            nx = 2 * ((nx + 1) // 2)
            sph_rad = 0.5 * h * (nx - 1)
            ny = nx
            nz = nx
        except (TypeError, ValueError):
            print('Warning: invalid radius override, keeping auto radius.')

    # Create the Domain dictionary
    Domain = {'radius': sph_rad, 'nx': nx, 'ny': ny, 'nz': nz, 'h': h}

    return Domain, N_types

# def recenter_atoms(Atoms):
#     all_coords = np.vstack([atom['coord'] for atom in Atoms])
#     center = np.mean(all_coords, axis=0)
#     for atom in Atoms:
#         atom['coord'] -= center
#     return Atoms

# # Recenter Domain atoms for grid generation only for SAD; ML keeps original coords to match density file
# if density_method == 'sad':
#     Atoms = recenter_atoms(Atoms)
#     print("recenter atoms", Atoms)

# Call the function
Domain, N_types = estimate_radius_and_grid(Atoms, elem, N_elements, h, ml_file_path, radius_override=radius_override)
if Domain:
    print(f"Domain: {Domain}")


def write_rsdft_parameter_output(filename, nev, Atoms, n_atom, Domain, h, poldeg, fd_order, bohr_to_ang=A0_ANG):
    """
    Writes detailed atom data and grid information to a file.

    Parameters:
    filename (str): The output file path.
    nev (int): Number of eigenvalues (states).
    Atoms (list): List of atomic species with type and coordinates.
    n_atom (list): List of number of atoms for each species.
    Domain (dict): Dictionary containing domain and grid size information.
    h (float): Grid spacing.
    poldeg (int): Polynomial degree used in calculations.
    fd_order (int): Finite difference order.
    """
    # Open the file for writing
    with open(filename, 'w') as fid:
        # Write number of states
        fid.write(f" Number of states: \t{nev}\n\n")
        fid.write("Atom data:\n -------------\n")

        # Total number of atom types
        at = len(Atoms)
        fid.write(f" Total # of atom types is {at}\n")
        atom_count = 0

        # Loop over atom types to write details
        for at_typ in range(at):
            xyz = Atoms[at_typ]['coord']
            fid.write(f" There are {n_atom[at_typ]} {Atoms[at_typ]['typ']} atoms\n")
            fid.write(" and their coordinates are:\n\n")
            fid.write("\t{:<12s}\t{:<12s}\t{:<12s}\n".format("x [a.u.]", "y [a.u.]", "z [a.u.]"))

            # Count total number of atoms and write the coordinates
            atom_count += n_atom[at_typ]
            for i in range(n_atom[at_typ]):
                fid.write(f"\t{xyz[i, 0]:12.6f}\t{xyz[i, 1]:12.6f}\t{xyz[i, 2]:12.6f}\n")
            fid.write('\n')

            # Print Angstrom version
            xyz_ang = xyz * bohr_to_ang
            fid.write("\t{:<12s}\t{:<12s}\t{:<12s}\n".format("x [Å]", "y [Å]", "z [Å]"))
            for i in range(n_atom[at_typ]):
                fid.write(f"\t{xyz_ang[i, 0]:12.6f}\t{xyz_ang[i, 1]:12.6f}\t{xyz_ang[i, 2]:12.6f}\n")
            fid.write('\n')

        # Final summary of atom data
        fid.write(' --------------------------------------------------\n')
        fid.write(f' Total number of atoms :         {atom_count}\n\n')

        # Write grid and domain information
        fid.write(f' Number of states:               {nev:10d} \n')
        fid.write(f' h grid spacing :                {h:10.5f} a.u.   ({h*bohr_to_ang:10.5f} Å)\n')
        fid.write(f' Hamiltonian size :              {Domain["nx"] * Domain["ny"] * Domain["nz"]:10d}  \n')
        fid.write(f' Sphere Radius :                 {Domain["radius"]:10.5f} a.u.   ({Domain["radius"]*bohr_to_ang:10.5f} Å)\n')
        fid.write(f' # grid points in each direction {Domain["nx"]:10d}  \n')
        fid.write(f' Polynomial degree used :        {poldeg:10d}  \n')
        fid.write(f' Finite difference order :       {fd_order:10d}  \n')
        fid.write(' --------------------------------------------------\n')

# Write the output to the file
write_rsdft_parameter_output(output_file, nev, Atoms, n_atom, Domain, h, poldeg, fd_order)

print(' ')
print('******************')
print('     OUTPUT       ')
print('******************')
print(' ')
print(' Working.....constructing Laplacian matrix...')

# Construct the Laplacian operator
nx, ny, nz = Domain['nx'], Domain['ny'], Domain['nz']
#nx, ny, nz = 2, 2, 2
start_time = time.time()
A = (1 / (h * h)) * fd3d(nx, ny, nz, fd_order)
Laplacian_time = time.time() - start_time
print(Laplacian_time)

# Initialize variables
n = A.shape[0]  # Size of the matrix
Hpot = np.zeros(n)  # Initialize the potential vector Hpot
pot = Hpot.copy()  # Initialize the potential vector pot
err = 10 + tol  # Set initial error value
its = 0  # Initialize iteration counter

print(' Working.....setting up ionic potential...')
start_time = time.time()
E_nuc0 = nuclear(Domain, Atoms, elem, N_elements)
Enuc_time = time.time() - start_time
print(' Enuc time', Enuc_time)
# print("Nucleus repulsion energy:", enuc)

# Step 1: Calculate initial charge density and potentials
print(' Working.....setting up diagonal part of ionic potential...')
start_time = time.time()

if density_method == 'sad':
    print('Using SAD method...')
    from V_ion.pseudoDiag import pseudoDiag
    rho0, hpot0, Ppot = pseudoDiag(Domain, Atoms, elem, N_elements)
else: # ML method
    print(f'Using ML-predicted density from: {ml_file_path}')

    from V_ion.pseudoDiag_ML4Den_poisson import pseudoDiag_ML4Den

    PRE = []
    if CG_prec:
        print('Calling ilu0 ...')
        # Perform an incomplete LU decomposition
        # spilu() from scipy can be used similarly to MATLAB's luinc
        PRE = spla.spilu(A) # PRE will be the equivalent of MATLAB's LU structure
        print('done.')

    # Call ML-based initialization
    # print('Atoms_raw', Atoms_raw)
    rho0, hpot0, Ppot = pseudoDiag_ML4Den( Domain, Atoms, elem, N_elements, ml_file_path, A, CG_prec, PRE )

pseudoDiag_time = time.time() - start_time
print(' pseudoDiag time: ', pseudoDiag_time)

# Step 2: Renormalize if the charge state is not neutral
if Z_charge != 0:
    scaling_factor = ztest / zelec
    rho0 *= scaling_factor
    hpot0 *= scaling_factor

# Step 3: Calculate Hartree energy (in eV)
hpsum0 = np.sum(rho0 * hpot0) * Ry

# Step 4: Write the initial Hartree energy to the output file

with open(output_file, 'a') as fid:
    fid.write(f" Initial Hartree energy (eV) = {hpsum0:10.5f}  \n")

# count # atoms for stats
n_atoms = 0
for at in Atoms:
    n_atoms += at['coord'].shape[0]

# Compute the non-local part of the pseudopotential
print(' Working.....setting up nonlocal part of ionic potential...')
start_time = time.time()
if density_method == 'sad':
    print('Using SAD method...')
    vnl = pseudoNL(Domain, Atoms, elem, N_elements)
else:
    print(f'Using ML-predicted density from: {ml_file_path}')
    vnl = pseudoNL_ML4Den(Domain, Atoms, elem, N_elements)

pseudoNL_time = time.time() - start_time
print(' pseudoNL time: ', pseudoNL_time)

# Set h from the Domain object
h = Domain['h']
rad = Domain['radius']

# Screening from Gaussian density
# Transposing and dividing by h^3 as per the original MATLAB code
rhoxc = np.transpose(rho0) / (h**3)

# Assuming `exc_nspn` is a function that computes XC potential
print(' Working.....setting up exchange and correlation potentials...')
start_time = time.time()
XCpot, exc = exc_nspn(Domain, rhoxc, output_file)
exc_time = time.time() - start_time
print(' exc time: ', exc_time)

# Transpose the result back to match the original code's xcpot = XCpot'
xcpot = np.transpose(XCpot)

#Calculate the number of electrons
Nelec = nelectrons(Atoms, elem, N_elements)

# Adjust for any charge in the system (if Z_charge is non-zero)
if Z_charge != 0:
    Nelec -= Z_charge

# At this stage, pot is calculated as the sum of Ppot, hpot0, and 0.5 * xcpot
pot = Ppot + hpot0 + 0.5 * xcpot

# SCF LOOP
# when 'preconditioning' is used fall ilu0
PRE = []
if CG_prec:
    print('Calling ilu0 ...')
    # Perform an incomplete LU decomposition
    # spilu() from scipy can be used similarly to MATLAB's luinc
    PRE = spla.spilu(A)  # PRE will be the equivalent of MATLAB's LU structure
    print('done.')

# Clear persistent variables in mixer (equivalent to MATLAB's "clear mixer")
reset_mixer()

# SCF LOOP starts here
with open(output_file, 'a') as fid:
    fid.write('\n----------------------------------\n\n')

# Precompute the constant matrix for efficiency
halfAPlusvnl = 0.5 * A + vnl  # A and vnl do not change in the loop

# Initialize variables used in the adaptive scheme
if adaptiveScheme != 0 and sum(n_atom) <= 2:
    degreeAdaptiveModifier = 0.75
    mAdaptiveModifier = 0.95
else:
    degreeAdaptiveModifier = 1
    mAdaptiveModifier = 1

# SCF Loop
W = []
lam = []
occup = []
rho = 0
while err > tol and its <= maxits:
    its += 1
    print(f'  Working ... SCF iter # {its} ... ')

    # Redefine Hamiltonian
    B = halfAPlusvnl + sp.diags(pot, 0, shape=(n, n))

    # Diagonalization method defined
    start_time = time.time()

    if diagmeth == 1 or (its == 1 and diagmeth == 0):
        print('Calling lanczos...')
        v = np.random.randn(n, 1)
        W, lam = lanczos(B, nev + 15, v, nev + (500 * mAdaptiveModifier), 1e-5)

    elif its == 1 and diagmeth == 2:
        print('Calling chsubsp...')
        W, lam = chsubsp(poldeg * degreeAdaptiveModifier, nev + 15, B)

    elif its == 1 and diagmeth == 3:
        print('Calling first_filt...')
        W, lam = first_filt(nev + 15, B, poldeg)

    else:
        print('Calling chebsf...')
        W, lam = chefsi1(W, lam, poldeg * degreeAdaptiveModifier, nev, B)

    diag_time = time.time() - start_time

    # Print results to file
    with open(output_file, 'a') as fid:
        fid.write(f'\n\n SCF iter # {its}  ... \n')
        fid.write(f'Diagonalization time [sec] :\t{diag_time}\n\n')

    # Get occupation factors and Fermi level
    Fermi_level, occup = occupations(lam[:nev], Fermi_temp, Nelec, 1e-6)

    # Print eigenvalues and occupations
    with open(output_file, 'a') as fid:
        fid.write('   State  Eigenvalue [Ry]     Eigenvalue [eV]\n\n')
    for i in range(nev):
        eig = lam[i] * 2 * Ry
        ry = eig / Ry
        occ = occup[i]
        with open(output_file, 'a') as fid:
            fid.write(f'{i + 1:5d}   {ry:15.10f}   {eig:18.10f}  {occ:5.2f}\n')

    # Get charge density
    # rho = (W(:,1:nev) .* W(:,1:nev)) *2* occup ; changed...
    # (W(:,1:nev)*W(:,1:nev)) to W(:,1:nev).^2
    rho = (W[:, :nev] ** 2) @ (2 * occup)  # Element-wise square and matrix multiplication

    hrhs = (4 * np.pi / h ** 3) * (rho - rho0)

    rho = rho / h ** 3

    # Trigger timer for Hartree potential calculation
    start_time = time.time()

    if CG_prec:
        print("with CG_prec")
        Hpot, _ = pcg(A, hrhs, Hpot, 200, 1e-6, PRE, 'precLU')  # Preconditioned CG
    else:
        print("no CG_prec")
        Hpot, _ = pcg(A, hrhs, Hpot, 200, 1e-6)  # Standard CG

    #print("new Hpot:", Hpot.shape)
    hart_time = time.time() - start_time
    with open(output_file, 'a') as fid:
        fid.write(f'\nHartree potential time [sec]: \t{hart_time}\n\n')

    # Get exchange-correlation potential
    XCpot, exc = exc_nspn(Domain, rho, output_file)
    #print("new XCpot:", XCpot.shape)

    # Compute the new potential
    potNew = Ppot + 0.5 * XCpot + Hpot + hpot0

    # Compute error
    errNew = np.linalg.norm(potNew - pot) / np.linalg.norm(potNew)

    # Adaptive scheme adjustments
    if adaptiveScheme == 0 or errNew > 1 or errNew > 2 * err:
        degreeAdaptiveModifier = 1
        mAdaptiveModifier = 1
    elif errNew > err:
        degreeAdaptiveModifier = min(1.1, degreeAdaptiveModifier + 0.2)
        mAdaptiveModifier = min(1.1, degreeAdaptiveModifier + 0.05)
    elif 3 * errNew < err:
        degreeAdaptiveModifier = max(0.5, degreeAdaptiveModifier - 0.1)
        mAdaptiveModifier = max(0.9, degreeAdaptiveModifier - 0.025)

    err = errNew

    # Print SCF error to output files
    with open(output_file, 'a') as fid:
        fid.write(f'   ... SCF error = {err:10.2e}\n')

        # Per-iteration energy components
        Esum_iter = np.sum(lam[:nev] * occup[:nev])
        Esum_iter0 = 4 * Esum_iter
        Hsum_iter0 = np.sum(rho * (Hpot + hpot0)) * h**3
        Vxcsum_iter0 = np.sum(rho * XCpot) * h**3
        Excsum_iter0 = exc
        E_elec_iter0 = Esum_iter0 - Hsum_iter0 + Excsum_iter0 - Vxcsum_iter0
        E_total_iter0 = E_elec_iter0 + E_nuc0

        Esum_iter_eV = Ry * Esum_iter0
        Hsum_iter_eV = Ry * Hsum_iter0
        Excsum_iter_eV = Ry * Excsum_iter0
        E_total_iter_eV = Ry * E_total_iter0

        fid.write("   Energy components this iter:\n")
        fid.write(f"     Sum of eigenvalues      = {Esum_iter_eV:10.5f}  eV   = {Esum_iter0:10.5f}  Ry  \n")
        fid.write(f"     Hartree energy          = {Hsum_iter_eV:10.5f}  eV   = {Hsum_iter0:10.5f}  Ry  \n")
        fid.write(f"     Exchange-corr. energy   = {Excsum_iter_eV:10.5f}  eV   = {Excsum_iter0:10.5f}  Ry  \n")
        fid.write(f"     Ion-ion repulsion       = {Ry * E_nuc0:10.5f}  eV   = {E_nuc0:10.5f}  Ry  \n")
        fid.write(f"     Total electronic energy = {E_total_iter_eV:10.5f}  eV   = {E_total_iter0:10.5f}  Ry  \n")
        fid.write(f"     Electronic energy/atom  = {E_total_iter_eV/n_atoms:10.5f}  eV   = {E_total_iter0/n_atoms:10.5f}  Ry  \n")
    print(f'   ... SCF error = {err:10.2e}\n')

    # Call mixer to update potential
    pot, _ = mixer(pot, potNew - pot)
    #print("new pot:", pot.shape)

print("SCF loop completed.")

# Check convergence status and display messages
if err > tol:
    print("          ")
    print("**************************")
    print(" !!THE SYSTEM DID NOT CONVERGE!!")
    print("          ")
    print(" !!THESE ARE THE VALUES FROM THE LAST ITERATION!!")
    print("**************************")
    print("         ")
else:
    print("          ")
    print("**************************")
    print(" CONVERGED SOLUTION!! ")
    print("**************************")
    print("         ")


# Print Eigenvalues and Occupations
print("   State  Eigenvalue [Ry]     Eigenvalue [eV]  Occupation ")
for i in range(nev):
    eig = lam[i] * 2 * Ry
    ry = eig / Ry
    occ = 2 * occup[i]
    print(f"{i + 1:5d}   {ry:15.4f}   {eig:18.3f}  {occ:10.2f}")

# Total Energy Calculations
Esum = np.sum(lam[:nev] * occup[:nev])
Esum0 = 4 * Esum

# Hartree Potential Sum
Hsum0 = np.sum(rho * (Hpot + hpot0)) * h**3

# Exchange-Correlation Sum
Vxcsum0 = np.sum(rho * XCpot) * h**3
Excsum0 = exc

# Total Electronic Energy
E_elec0 = Esum0 - Hsum0 + Excsum0 - Vxcsum0

# Add Nuclear-Nuclear Repulsion Term
E_total0 = E_elec0 + E_nuc0

# Convert to eV
Esum = Ry * Esum0
Hsum = Ry * Hsum0
Excsum = Ry * Excsum0
E_nuc = Ry * E_nuc0
E_total = Ry * E_total0

# Print Results to File and Console
with open(output_file, 'a') as fid:
    fid.write("\n\n")
    fid.write(" Total Energies \n\n")
    fid.write(f" Sum of eigenvalues      = {Esum:10.5f}  eV   = {Esum/Ry:10.5f}  Ry  \n")
    fid.write(f" Hartree energy          = {Hsum:10.5f}  eV   = {Hsum/Ry:10.5f}  Ry  \n")
    fid.write(f" Exchange-corr. energy   = {Excsum:10.5f}  eV   = {Excsum/Ry:10.5f}  Ry  \n")
    fid.write(f" Ion-ion repulsion       = {E_nuc:10.5f}  eV   = {E_nuc/Ry:10.5f}  Ry  \n")
    fid.write(f" Total electronic energy = {E_total:10.5f}  eV   = {E_total0:10.5f}  Ry  \n")
    fid.write(f" Electronic energy/atom  = {E_total/n_atoms:10.5f}  eV   = {E_total0/n_atoms:10.5f}  Ry  \n")

# Print Results to Console
print("\n Total Energies \n\n")
print(f" Sum of eigenvalues      = {Esum:10.5f}  eV   = {Esum/Ry:10.4f}  Ry  ")
print(f" Hartree energy          = {Hsum:10.5f}  eV   = {Hsum/Ry:10.4f}  Ry  ")
print(f" Exchange-corr. energy   = {Excsum:10.5f}  eV   = {Excsum/Ry:10.4f}  Ry  ")
print(f" Ion-ion repulsion       = {E_nuc:10.5f}  eV   = {E_nuc/Ry:10.4f}  Ry  ")
print(f" Total electronic energy = {E_total:10.5f}  eV   = {E_total0:10.4f}  Ry  ")
print(f" Electronic energy/atom  = {E_total/n_atoms:10.5f}  eV   = {E_total0/n_atoms:10.4f}  Ry  ")

# Free memory (reset persistent variables in the mixer)
reset_mixer()

# Output Results to Binary File
with open("./wfn.dat", "wb") as wfnid:
    little_big_test = 26
    wfnid.write(np.array(little_big_test, dtype=np.uint32).tobytes())

    wfnid.write(np.array(Domain['radius'], dtype=np.float64).tobytes())
    wfnid.write(np.array(Domain['h'], dtype=np.float64).tobytes())

    pot_length = len(pot)
    wfnid.write(np.array(pot_length, dtype=np.uint32).tobytes())
    wfnid.write(np.array(pot, dtype=np.float64).tobytes())

    rho_length = len(rho)
    wfnid.write(np.array(rho_length, dtype=np.uint32).tobytes())
    wfnid.write(np.array(rho, dtype=np.float64).tobytes())

    w_length = len(W)
    wfnid.write(np.array(w_length, dtype=np.uint32).tobytes())
    wfnid.write(np.array(nev, dtype=np.uint32).tobytes())
    for i in range(nev):
        wfnid.write(np.array(W[:, i], dtype=np.float64).tobytes())

    # Write the atomic structure in `wfn.dat`
    wfnid.write(np.array(N_types, dtype=np.uint32).tobytes())
    for atom in Atoms:
        xyz = atom['coord']
        wfnid.write(np.array(len(xyz), dtype=np.uint32).tobytes())
        for j in range(len(xyz)):
            wfnid.write(np.array(xyz[j, :], dtype=np.float64).tobytes())

# Close output file
fid.close()
