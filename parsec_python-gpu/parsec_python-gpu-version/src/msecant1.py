import numpy as np
from qinv import qinv

class mixer:
    def __init__(self):
        # Persistent variables corresponding to MATLAB's persistent variables.
        self.clear_mixer = None
        self.DX = None
        self.DF = None
        self.N = None
        self.group_size = None
        self.mix = None
        self.tol = None
        self.restart_factor = None
        self.EN_stage = None
        self.EN_like = None

    def mixer(self, x1, f1):
        if self.clear_mixer is None:
            clear_mixer = 0
        x1, f1 = self.msecant1(x1, f1)
        return x1, f1

    def includemix(self):
        """
        This function defines parameters for mixing that can be used by the multi-secant method.

        Types of mixing methods:
        - Simple mixing (simplemix)
        - Multi-secant methods with Type-I update to minimize the change of Jacobian (msecant1)
        - Multi-secant methods with Type-II update to minimize the change of inverse Jacobian (msecant2)
        - Hybrid methods (msecant3)
        """

        # Indicates Broyden-like (EN_like=0) or EN-like (EN_like=1) update.
        self.EN_like = 0

        # group_size is the size of the groups.
        # For Broyden's family, set group_size=1 and EN_like=0;
        # The (first) EN-like algorithm proposed by Yang is obtained by setting
        # group_size=1 and EN_like=1 and using msecant1.
        self.group_size = 1

        # Mixing parameter.
        self.mix = 0.5

        # A hybrid method (msecant3) chooses Type-I or Type-II update depending on the secant errors.
        self.preferred_type = 1

        # (Relative) tolerance for ill-conditioned linear systems.
        self.tol = np.finfo(float).eps

        # If |f_new| is too large relative to |f_old|, then perform restart.
        self.restart_factor = 0.1

    def msecant1(self, x1, f1):
        """
        Multi-secant methods for solving nonlinear equations f(x)=0;
        Type-I methods that minimize the change of approximate Jacobian;
        [new_x, m] = msecant1(x1, f1);

        Input:
        x1 = latest iterate;
        f1 = f(x1).

        Output:
        x_new = new estimate of the solution to f(x)=0;
        m = number of secant equations.
        """
        # Initially the columns of DX and DF store x_{j+1}-x_j and f_{j+1}-f_j,
        # respectively. The E and V in the note also share the storage with DX and
        # DF, respectively. N stores the last group N_i in the note.
        if self.DX is None:  # Check if persistent variables are uninitialized
            # Store the current iterate.
            m = 0
            self.includemix()
            self.DX = x1[:, np.newaxis]
            self.DF = f1[:, np.newaxis]
            # Set the new estimate.
            x_new = x1 + self.mix * f1  # Simple mixing.
            if self.EN_like == 0:  # Broyden-like update.
                self.EN_stage = 0
            else:  # EN-like update.
                self.EN_stage = 2  # Next EN stage.
            return x_new, m
        else:
            # Number of previous iterates.
            m = self.DX.shape[1]

        if self.group_size == 0:
            sz = m + 1  # Take all available iterates in one group.
        else:
            sz = self.group_size

        # When the current iterate is bad, perform restart.
        # More precisely, if ||f0|| < restart_factor*||f1||, then restart, where f0
        # is the previous function value. In particular, restart_factor = 0 implies
        # never restart.
        if self.EN_stage != 1 and m >= 2 and np.linalg.norm(self.DF[:, -1], 2) < self.restart_factor * np.linalg.norm(f1, 2):
            x1 = self.DX[:, m-1]
            f1 = self.DF[:, m-1]
            self.DX = x1[:, np.newaxis]
            self.DF = f1[:, np.newaxis]
            x_new = x1 + self.mix * f1  # Simple mixing.
            m = 0
            return x_new, m

        # Compute res and ngroup and update DX and DF.
        res = (m + sz - 1) % sz + 1  # Size of the last group.
        ngroup = (m - res) // sz
        # ngroup does not count the last group; the number of groups is ngroup+1.
        dx = []
        if self.EN_stage != 1:
            self.DX[:, m-1] = x1 - self.DX[:, m-1]  # dx_m = x_{m+1} - x_m.
            self.DF[:, m-1] = f1 - self.DF[:, m-1]  # df_m = f_{m+1} - f_m.
            dx = self.DX[:, m-1]
            self.DX[:, m-1] += self.mix * self.DF[:, m-1]
            for i in range(ngroup):
                self.DX[:, m-1] -= self.DX[:, (i * sz):((i + 1) * sz)] @ (self.DF[:, (i * sz):((i + 1) * sz)].T @ self.DF[:, m-1])
            # DX[:, -1] is now E[:, -1] in the note.

        # Compute new x.
        x_new = x1 + self.mix * f1
        for i in range(ngroup):
            x_new -= self.DX[:, (i * sz):((i + 1) * sz)] @ (self.DF[:, (i * sz):((i + 1) * sz)].T @ f1)
            # DX[:,...] is E[:,...] in the note.
            # DF[:,...] is V[:,...] in the note.

        # Now deal with the last group.
        if self.EN_stage != 1:
            self.N = np.zeros((x1.size, res))
            self.N[:, res - 1] = -self.mix * dx
            for i in range(ngroup):
                self.N[:, res - 1] += self.DF[:, (i * sz):((i + 1) * sz)] @ (self.DX[:, (i * sz):((i + 1) * sz)].T @ dx)

        if self.EN_stage == 1 and res == sz:
            x_new -= self.DX[:, ngroup * sz:m] @ (self.DF[:, ngroup * sz:m].T @ f1)
        else:
            M = self.N.T @ self.DF[:, m - res:m]
            C = qinv(M, self.tol)  # Quasi-inverse of M by QR factorization.
            if res == sz:
                self.DF[:, m - sz:m] = self.N @ C.T
                x_new -= self.DX[:, m - res:m] @ (self.DF[:, m - res:m].T @ f1)
            else:
                x_new -= self.DX[:, m - res:m] @ ((C @ self.N.T) @ f1)

        if self.EN_stage != 2:
            self.DX = np.hstack([self.DX, x1[:, np.newaxis]])
            self.DF = np.hstack([self.DF, f1[:, np.newaxis]])

        if self.EN_stage == 1:
            self.EN_stage = 2  # Next EN stage.
        elif self.EN_stage == 2:
            self.EN_stage = 1  # Next EN stage.

        return x_new, m
