import cupy as cp


def FermiDirac(lam, EF, Temp, Nelec):
    """
    GPU version of the CPU Fermi-Dirac occupation helper.
    """
    kT = Temp * 6.33327186e-06
    spin = 1

    logits = (lam - EF) / kT
    exp_neg = cp.exp(-cp.abs(logits))
    occup = cp.where(
        logits >= 0,
        spin * exp_neg / (1.0 + exp_neg),
        spin / (1.0 + exp_neg),
    )

    fe = cp.sum(occup) - Nelec / 2
    return fe, occup
