import numpy as np
import scipy.sparse as sp
from splineData_new import splineData
from preProcess_new import preProcess_numba
from fspline_new import fspline_optimized
from fsplevalIO_new import fsplevalIO_optimized
from spconvert_new import spconvert_optimized

def pseudoNL_optimized(Domain, Atoms, elem, N_elements):
    """
    Optimized computation of the non-local pseudopotential contribution to potential energy.
    """

    # Extract domain variables
    nx, ny, nz, h, rad = Domain['nx'], Domain['ny'], Domain['nz'], Domain['h'], Domain['radius']

    # Load spline data
    AtomFuncData, data_list = splineData()

    # Dimension of the sparse matrix
    ndim = nx * ny * nz

    # Initialize sparse matrix for nonlocal pseudopotential
    vnl = sp.lil_matrix((ndim, ndim), dtype=np.float64)

    # Identify data columns for pot_P, pot_S, and wfn_P
    pot_P = data_list.index('pot_P') if 'pot_P' in data_list else -1
    pot_S = data_list.index('pot_S') if 'pot_S' in data_list else -1
    wav_P = data_list.index('wfn_P') if 'wfn_P' in data_list else -1

    # Iterate over all atom types
    for at_typ, atom_data in enumerate(Atoms):
        typ = atom_data['typ']

        # Find matching atom data in splineData
        atom_index = next((i for i, data in enumerate(AtomFuncData) if data['atom'] == typ), None)
        if atom_index is None:
            continue

        atom_spline_data = AtomFuncData[atom_index]['data']
        xi = atom_spline_data[:, 0]

        # Extract or initialize potPS and wfn_P
        potPS = atom_spline_data[:, pot_P] if pot_P >= 0 else np.zeros_like(xi)
        if pot_S >= 0:
            potPS -= atom_spline_data[:, pot_S]
        wfn_P = atom_spline_data[:, wav_P] if wav_P >= 0 else np.zeros_like(xi)

        # Pre-process data
        I_wfn = preProcess_numba(wfn_P)
        xi_wfn, wfn_P = xi[I_wfn], wfn_P[I_wfn]
        I_pot = preProcess_numba(potPS)
        xi_pot, potPS = xi[I_pot], potPS[I_pot]

        # Generate spline coefficients
        zWav, cWav, dWav = fspline_optimized(xi_wfn, wfn_P)
        zPotPS, cPotPS, dPotPS = fspline_optimized(xi_pot, potPS)

        # Element data
        elem_index = elem.index[elem['Element'] == typ][0]
        xint = elem['Zvalue'].iloc[elem_index] / (h ** 3)
        xyz = atom_data['coord']
        natoms = xyz.shape[0]
        Rzero = elem['R'].iloc[elem_index]

        # Process each atom
        for atom_pos in xyz:
            xxa, yya, zza = atom_pos
            i0, j0, k0 = [round((coord + rad) / h + 1) for coord in (xxa, yya, zza)]
            span = round(Rzero / h)

            points, distances = [], []

            for k in range(k0 - span, k0 + span + 1):
                zzz = (k - 1) * h - rad - zza
                for j in range(j0 - span, j0 + span + 1):
                    yyy = (j - 1) * h - rad - yya
                    for i in range(i0 - span, i0 + span + 1):
                        xxx = (i - 1) * h - rad - xxa
                        dd = np.sqrt(xxx ** 2 + yyy ** 2 + zzz ** 2)
                        if 0 < dd < Rzero:
                            points.append([i + nx * ((j - 1) + (k - 1) * ny), xxx, yyy, zzz])
                            distances.append(dd)

            if not points:
                continue

            points = np.array(points)
            nn, xx, yy, zz = points[:, 0].astype(int), points[:, 1], points[:, 2], points[:, 3]
            dd = np.array(distances)

            vspp, wavpp = np.zeros_like(dd), np.zeros_like(dd)
            j_p_ps, j_wfn = 1, 1

            for idx, dist in enumerate(dd):
                vspp[idx], j_p_ps = fsplevalIO_optimized(zPotPS, cPotPS, dPotPS, xi_pot, dist, j_p_ps)
                wavpp[idx], j_wfn = fsplevalIO_optimized(zWav, cWav, dWav, xi_wfn, dist, j_wfn)

            ulmspx = xx / dd * wavpp * vspp
            ulmspy = yy / dd * wavpp * vspp
            ulmspz = zz / dd * wavpp * vspp

            xmatrix = np.outer(ulmspx, ulmspx) / xint
            ymatrix = np.outer(ulmspy, ulmspy) / xint
            zmatrix = np.outer(ulmspz, ulmspz) / xint
            total = xmatrix + ymatrix + zmatrix

            vnll = np.zeros((len(nn) ** 2 + 1, 3))
            vnll[:len(nn) ** 2, 0] = np.repeat(nn, len(nn))
            vnll[:len(nn) ** 2, 1] = np.tile(nn, len(nn))
            vnll[:len(nn) ** 2, 2] = total.ravel()
            vnll[-1] = [ndim, ndim, 0]

            vnl += spconvert_optimized(vnll)

    return vnl