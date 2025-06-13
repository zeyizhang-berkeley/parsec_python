import scipy.io
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


def splineData(file_path: str = 'splineData.mat') -> Tuple[List[Dict], List[str]]:
    """
    Optimized version to load spline data for atoms.

    Parameters:
    file_path (str): Path to the splineData.mat file

    Returns:
    Tuple[List[Dict], List[str]]: Tuple containing:
        - List of dictionaries with atom-specific data
        - List of data labels

    Raises:
    FileNotFoundError: If splineData.mat is not found
    ValueError: If required atom data is missing
    """
    # Data labels (constant)
    DATA_LABELS = ['radius', 'charge', 'hartree', 'pot_P', 'pot_S', 'wfn_P', 'wfn_S']

    # Define atoms (constant)
    ATOMS = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'
    ]

    # Check if file exists
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Could not find {file_path}")

    try:
        # Load the data from splineData.mat
        mat_data = scipy.io.loadmat(file_path)

        # Pre-allocate list with known size
        AtomFuncData = []

        # Process each atom's data
        for atom in ATOMS:
            data_key = f'data{atom}'

            if data_key not in mat_data:
                raise ValueError(f"Data for atom {atom} not found in '{file_path}'")

            # Convert to numpy array if not already
            atom_data = np.asarray(mat_data[data_key])

            # Store data with type hint for better IDE support
            AtomFuncData.append({
                'atom': atom,
                'data': atom_data
            })

        return AtomFuncData, DATA_LABELS

    except Exception as e:
        raise RuntimeError(f"Error loading spline data: {str(e)}")


def get_atom_data(AtomFuncData: List[Dict], atom_symbol: str) -> np.ndarray:
    """
    Helper function to quickly retrieve data for a specific atom.

    Parameters:
    AtomFuncData (List[Dict]): List of atom data dictionaries
    atom_symbol (str): Atomic symbol (e.g., 'O' for oxygen)

    Returns:
    np.ndarray: Data for the specified atom

    Raises:
    ValueError: If atom_symbol is not found
    """
    for atom_dict in AtomFuncData:
        if atom_dict['atom'] == atom_symbol:
            return atom_dict['data']
    raise ValueError(f"Atom {atom_symbol} not found in data")


# Example usage and testing
if __name__ == "__main__":
    import time


    def test_spline_data():
        try:
            # Time the data loading
            t0 = time.time()
            AtomFuncData, data_list = splineData()
            t1 = time.time()

            print(f"Data loading time: {t1 - t0:.4f} seconds")
            print(f"Number of atoms loaded: {len(AtomFuncData)}")
            print(f"Available data types: {data_list}")

            # Test data access
            O_data = get_atom_data(AtomFuncData, 'O')
            print(f"\nOxygen data shape: {O_data.shape}")

            # Verify data structure
            for atom_dict in AtomFuncData:
                assert 'atom' in atom_dict
                assert 'data' in atom_dict
                assert isinstance(atom_dict['data'], np.ndarray)

            print("\nAll data verified successfully!")

        except Exception as e:
            print(f"Error during testing: {str(e)}")


    # Run tests
    test_spline_data()