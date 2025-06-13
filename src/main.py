import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import scipy.sparse.linalg as spla
from scipy.io import loadmat
from fd3d import fd3d
from nuclear import nuclear
from pseudoDiag import pseudoDiag
from pseudoNL import pseudoNL
from exc_nspn import exc_nspn
from nelectrons import nelectrons
from first_filt import first_filt
from chefsi1 import chefsi1
from lanczos import lanczos
from chsubsp import chsubsp
from occupations import occupations
from pcg import pcg
import msecant1

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
fd_order = 2            # order of finite difference scheme {8}
maxits   = 40           # max SCF iterations                {40}
tol      = 1.e-04       # tolerance for SCF iteration.      {1.e-03}
Fermi_temp  =  500.0    # Smear out fermi level             {500.0}

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

# Load element information from CSV file
def load_elements():
    """
    Load the element data from 'elements_new.csv' assuming there is no header,
    and that the first column contains the element type.
    """
    # Load the CSV file without headers
    elem = pd.read_csv('elements_new.csv', header=None)

    # Assign column names (assuming we know the structure)
    elem.columns = ['Element', 'AtomicNumber', 'Z', 'Zvalue', 'h', 'r', 'R'] + [f'Property{i}' for i in range(1, 5)]

    N_elements = elem.shape[0]  # This gives the number of elements (rows)
    # print(N_elements)
    return elem, N_elements

# Call the function to load the element data
elem, N_elements = load_elements()

def element_exists(typ, elem):
    """Check if the element exists in the element data."""
    return typ in elem['Element'].values  # Assuming elem is a pandas DataFrame


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
                    return None
                else:
                    # Reshape the list into (n, 3) NumPy array
                    xyz = np.array(coordinates).reshape(-1, 3)
                    errorFreeCoordinates = 1

            except ValueError:
                print('Error: Invalid input detected. Please ensure all coordinates are numbers.')
                return None
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

    return Atoms, n_atom, Z_charge


def load_atomic_data_from_file():
    """Handles file input for atomic species."""
    file_name = input('What is the name of the file to load from?: ')

    # Check if the file name is too short
    if len(file_name) <= 4:
        print('File name is too short, must include file extension')
        return None # stop execution and return to command window

    # check in the saved molecules folder
    file_name = os.path.join('SavedMolecules', file_name)

    # Extract the file extension
    file_extension = file_name[-4:]

    Atoms = []

    # Handle .dat files
    if file_extension == '.dat':
        try:
            with open(file_name, 'r') as file:
                at = int(file.readline().strip())
                for i in range(at):
                    typ = file.readline().strip()
                    readxyz = np.array([float(x) for x in file.readline().strip().split()])
                    n_atom = len(readxyz) // 3
                    xyz = np.reshape(readxyz, (n_atom, 3))
                    Atoms.append({'typ': typ, 'coord': xyz})
        except FileNotFoundError:
            print(f"File not found: {file_name}. File must be in SavedMolecules folder.")
            return None

    # Handle .mat files
    elif file_extension == '.mat':
        if not os.path.exists(file_name):
            print('File not found in directory, SavedMolecules')
            return None

        # Load .mat file
        mat_data = loadmat(file_name)

        if 'AtomsInMolecule' in mat_data:
            Atoms = AtomsInMoleculeToAtomsConverter(mat_data['AtomsInMolecule'])
        else:
            print('File does not contain the correct variable: AtomsInMolecule')
            return None

    # Handle unrecognized file extensions
    else:
        print('File extension not recognized')
        return None

    return Atoms


def AtomsInMoleculeToAtomsConverter(AtomsInMolecule):
    """Placeholder function for converting data from .mat file format to usable Python structures."""
    Atoms = []
    for atom in AtomsInMolecule:
        # Assuming AtomsInMolecule is a list-like structure
        typ = atom['typ']
        coord = atom['coord']
        Atoms.append({'typ': typ, 'coord': coord})
    return Atoms


# Main function to get Atoms based on user input (manual or file)
def get_atoms(elem):
    # Ask the user for input mode
    print(' ********************** ')
    print(' DATA INPUT FOR RSDFT')
    print(' *********************  ')
    print('------------------------')

    in_data = 0
    Atoms = []
    n_atom = []
    Z_charge = 0.0
    while (in_data != 1) and (in_data != 2):
        try:
            in_data = int(input('Input data mode: 1 for manual, 2 for file: '))
        except ValueError:
            print('Please enter 1 or 2')

    # Case 1: Manual input
    if in_data == 1:
        Atoms, n_atom, Z_charge = manual_input_species(elem)
        if n_atom is None:
            return None
    # Case 2: File input
    # TODO: try to decide what kind of forms of data should be considered
    elif in_data == 2:
        Atoms, n_atom, Z_charge = load_atomic_data_from_file()
        if n_atom is None:
            return None

    return Atoms, n_atom, Z_charge


# Usage example:
# Assuming elem is a pandas DataFrame loaded with element data from 'elements_new.csv'
Atoms, n_atom, Z_charge = get_atoms(elem)
# print(Atoms,n_atom,Z_charge)
if Atoms is not None:
    print(f"Successfully loaded {len(Atoms)} atomic species.")
else:
    print("No atoms loaded, exiting RSDFT")


def calculate_grid_spacing(Atoms, elem, N_elements, n_atom, Z_charge):
    """
    Calculate the smallest grid spacing hmin and the number of eigenvalues (nev).

    Parameters:
    Atoms (list of dictionaries): List of atomic species with type and coordinates.
    elem (pd.DataFrame): DataFrame containing element information (including grid spacing).
    n_atom (list): List containing the number of atoms for each species.
    Z_charge (float): Charge state of the system.

    Returns:
    hmin (float): Smallest grid spacing.
    nev (int): Number of eigenvalues (number of states).
    """
    N_types = len(Atoms)

    zelec = 0.0
    hmin = 100.0  # Large initial value to find the minimum grid spacing

    # Iterate over the types of atoms
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']  # Get the atomic symbol

        # Look for matching element data in the elem DataFrame
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                Z = elem['Z'].iloc[i] * n_atom[at_typ]  # Number of electrons for this species
                h = elem['h'].iloc[i]  # getting grid spacing
                if h < hmin:
                    hmin = h  # Update hmin if a smaller grid spacing is found
                zelec += Z  # Add the electrons from this species

    # Check for valid electron count
    ztest = zelec - Z_charge
    if ztest < 0:
        print('Problem with charge state. Negative number of electrons.')
        return None, None

    # Calculate the smallest grid spacing and number of eigenvalues
    h = hmin
    nev = max(16, round(0.7 * zelec + 0.5))  # At least 16 eigenvalues are required

    return h, nev, zelec, ztest

h, nev, zelec, ztest = calculate_grid_spacing(Atoms, elem, N_elements, n_atom, Z_charge)

# if h is not None and nev is not None:
#     print(f"Smallest grid spacing (h): {h}")
#     print(f"Number of eigenvalues (nev): {nev}")


def estimate_radius_and_grid(Atoms, elem, N_elements, h):
    """
    Estimate the spherical radius and calculate grid sizes based on atom positions.

    Parameters:
    Atoms (list of dictionaries): List of atomic species with type and coordinates.
    elem (pd.DataFrame): DataFrame containing element information (including atomic size).
    h (float): Grid spacing (smallest h from previous calculations).

    Returns:
    Domain (dict): A dictionary containing grid size information and the spherical radius.
    """
    xyz = []
    rmax = 0.0  # Initialize max radius
    natoms = 0.0  # Initialize number of atoms
    rsize = 0.0  # Initialize radius size
    N_types = len(Atoms)  # Number of atomic species

    # Iterate over the types of atoms
    for at_typ in range(N_types):
        typ = Atoms[at_typ]['typ']  # Get the atomic symbol
        xyz = Atoms[at_typ]['coord']
        natoms = xyz.shape[0]  # Number of atoms of this type
        # Find the corresponding element in elem
        index = None
        for i in range(N_elements):
            if typ == elem['Element'].iloc[i]:
                # Retrieve atomic size/ radius
                rsize = elem['r'].iloc[i]

        # Scan all points to find the atom most removed from the domain center
        for at1 in range(natoms):
            xx, yy, zz = xyz[at1, 0], xyz[at1, 1], xyz[at1, 2]  # Coordinates of atom
            rdis = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)  # Distance from the origin
            rs = rdis + rsize  # Radius including the atomic size

            if rs > rmax:
                rmax = rs  # Update the maximum radius

    sph_rad = rmax  # Spherical radius based on the most distant atom

    # Grid size calculations
    nx = int(2 * sph_rad / h) + 1  # Ensure nx is odd to make adjustments later
    nx = 2 * ((nx + 1) // 2)  # Ensure nx is even
    sph_rad = 0.5 * h * (nx - 1)  # Adjust the spherical radius based on the grid
    ny = nx
    nz = nx

    # Create the Domain dictionary
    Domain = {'radius': sph_rad, 'nx': nx, 'ny': ny, 'nz': nz, 'h': h}

    return Domain, N_types

# Call the function
Domain, N_types = estimate_radius_and_grid(Atoms, elem, N_elements, h)
# if Domain:
#     print(f"Domain: {Domain}")


def write_rsdft_parameter_output(filename, nev, Atoms, n_atom, Domain, h, poldeg, fd_order):
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
            fid.write("\tx [a.u.]\t\ty [a.u.]\t\tz [a.u.]\n")

            # Count total number of atoms and write the coordinates
            atom_count += n_atom[at_typ]
            for i in range(n_atom[at_typ]):
                fid.write(f"\t{xyz[i, 0]:.6f}\t\t{xyz[i, 1]:.6f}\t\t{xyz[i, 2]:.6f}\n")
            fid.write('\n')

        # Final summary of atom data
        fid.write(' --------------------------------------------------\n')
        fid.write(f' Total number of atoms :         {atom_count}\n\n')

        # Write grid and domain information
        fid.write(f' Number of states:               {nev:10d} \n')
        fid.write(f' h grid spacing :                {h:10.5f}  \n')
        fid.write(f' Hamiltonian size :              {Domain["nx"] * Domain["ny"] * Domain["nz"]:10d}  \n')
        fid.write(f' Sphere Radius :                 {Domain["radius"]:10.5f}   \n')
        fid.write(f' # grid points in each direction {Domain["nx"]:10d}  \n')
        fid.write(f' Polynomial degree used :        {poldeg:10d}  \n')
        fid.write(f' Finite difference order :       {fd_order:10d}  \n')
        fid.write(' --------------------------------------------------\n')

# Write the output to the file
write_rsdft_parameter_output('./rsdft_parameter.out', nev, Atoms, n_atom, Domain, h, poldeg, fd_order)

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

# Output some of the initialized variables (for demonstration)
# print(f"Laplacian matrix A of size {A.shape}")
# print(A)
# print(f"Initial error: {err}")
# print(f"Initial iterations: {its}")

print(' Working.....setting up ionic potential...')
start_time = time.time()
E_nuc0 = nuclear(Domain, Atoms, elem, N_elements)
Enuc_time = time.time() - start_time
print(Enuc_time)
# print("Nucleus repulsion energy:", enuc)

# Step 1: Calculate initial charge density and potentials
print(' Working.....setting up diagonal part of ionic potential...')
start_time = time.time()
rho0, hpot0, Ppot = pseudoDiag(Domain, Atoms, elem, N_elements)
pseudoDiag_time = time.time() - start_time
print(pseudoDiag_time)
#print("hpot0: ", hpot0.shape)
#print("Ppot: ", Ppot.shape)

# Step 2: Renormalize if the charge state is not neutral
if Z_charge != 0:
    scaling_factor = ztest / zelec
    rho0 *= scaling_factor
    hpot0 *= scaling_factor

# Step 3: Calculate Hartree energy (in eV)
hpsum0 = np.sum(rho0 * hpot0) * Ry

# Step 4: Write the initial Hartree energy to the output file

with open('./rsdft_parameter.out', 'a') as fid:
    fid.write(f" Initial Hartree energy (eV) = {hpsum0:10.5f}  \n")

# count # atoms for stats
n_atoms = 0
for at in Atoms:
    n_atoms += at['coord'].shape[0]

# Compute the non-local part of the pseudopotential
print(' Working.....setting up nonlocal part of ionic potential...')
start_time = time.time()
vnl = pseudoNL_optimized(Domain, Atoms, elem, N_elements)
pseudoNL_time = time.time() - start_time
print(pseudoNL_time)
# print(vnl)

# Set h from the Domain object
h = Domain['h']

# Screening from Gaussian density
# Transposing and dividing by h^3 as per the original MATLAB code
rhoxc = np.transpose(rho0) / (h**3)

# Assuming `exc_nspn` is a function that computes XC potential
print(' Working.....setting up exchange and correlation potentials...')
start_time = time.time()
XCpot, exc = exc_nspn(Domain, rhoxc, fid)
exc_time = time.time() - start_time
print(exc_time)
#print("XCpot:", XCpot.shape)

# Transpose the result back to match the original code's xcpot = XCpot'
xcpot = np.transpose(XCpot)

#Calculate the number of electrons
Nelec = nelectrons(Atoms, elem, N_elements)

# Adjust for any charge in the system (if Z_charge is non-zero)
if Z_charge != 0:
    Nelec -= Z_charge

# Open a binary file to write the wave function data
# with open('wfn.dat', 'wb') as wfnid:

# At this stage, pot is calculated as the sum of Ppot, hpot0, and 0.5 * xcpot
pot = Ppot + hpot0 + 0.5 * xcpot
#print("pot:", pot.shape)

# SCF LOOP
# when 'preconditioning' is used fall ilu0
PRE = []
if CG_prec:
    print('Calling ilu0 ...')
    # Perform an incomplete LU decomposition
    # spilu() from scipy can be used similarly to MATLAB's luinc
    PRE = spla.spilu(A)  # PRE will be the equivalent of MATLAB's LU structure
    print('done.')

# Clear persistent variables in mixer (no equivalent in Python, can be ignored if not using persistent variables)
# This would be used if `mixer` was defined as a function with persistent variables.
# To mimic, just reinitialize the variable or set to None if needed.

# SCF LOOP starts here
with open('./rsdft_parameter.out', 'a') as fid:
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
    with open('./rsdft_parameter.out', 'a') as fid:
        fid.write(f'\n\n SCF iter # {its}  ... \n')
        fid.write(f'Diagonalization time [sec] :\t{diag_time}\n\n')

    # Get occupation factors and Fermi level
    Fermi_level, occup = occupations(lam[:nev], Fermi_temp, Nelec, 1e-6)

    # Print eigenvalues and occupations
    with open('./rsdft_parameter.out', 'a') as fid:
        fid.write('   State  Eigenvalue [Ry]     Eigenvalue [eV]\n\n')
    for i in range(nev):
        eig = lam[i] * 2 * Ry
        ry = eig / Ry
        occ = occup[i]
        with open('./rsdft_parameter.out', 'a') as fid:
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
        Hpot, _ = pcg(A, hrhs, Hpot, 200, 1e-4, PRE, 'precLU')  # Preconditioned CG
    else:
        print("no CG_prec")
        Hpot, _ = pcg(A, hrhs, Hpot, 200, 1e-4)  # Standard CG

    #print("new Hpot:", Hpot.shape)
    hart_time = time.time() - start_time
    with open('./rsdft_parameter.out', 'a') as fid:
        fid.write(f'\nHartree potential time [sec]: \t{hart_time}\n\n')

    # Get exchange-correlation potential
    XCpot, exc = exc_nspn(Domain, rho, fid)
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
    with open('./rsdft_parameter.out', 'a') as fid:
        fid.write(f'   ... SCF error = {err:10.2e}\n')
    print(f'   ... SCF error = {err:10.2e}\n')

    # Call mixer to update potential
    mixer = msecant1.mixer()
    pot, _ = mixer.mixer(pot, potNew - pot)
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
with open('./rsdft_parameter.out', 'a') as fid:
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
del mixer

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

