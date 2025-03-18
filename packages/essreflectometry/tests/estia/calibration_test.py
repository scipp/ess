import numpy as np
import scipp as sc


def generate_valid_calibration_parameters():
    I0 = np.random.random()
    Pp = np.random.random()
    Pa = -np.random.random()
    Ap = np.random.random()
    Aa = -np.random.random()
    Rspp = np.random.random()
    Rsaa = Rspp * np.random.random()
    return tuple(map(sc.scalar, (I0, Pp, Pa, Ap, Aa, Rspp, Rsaa)))


def intensity_from_parameters(I0, Pp, Pa, Ap, Aa, Rpp, Rpa, Rap, Raa):
    return (
        I0
        * (
            Rpp * (1 + Ap) * (1 + Pp)
            + Rpa * (1 - Ap) * (1 + Pp)
            + Rap * (1 + Ap) * (1 - Pp)
            + Raa * (1 - Ap) * (1 - Pp)
        ),
        I0
        * (
            Rpp * (1 + Aa) * (1 + Pp)
            + Rpa * (1 - Aa) * (1 + Pp)
            + Rap * (1 + Aa) * (1 - Pp)
            + Raa * (1 - Aa) * (1 - Pp)
        ),
        I0
        * (
            Rpp * (1 + Ap) * (1 + Pa)
            + Rpa * (1 - Ap) * (1 + Pa)
            + Rap * (1 + Ap) * (1 - Pa)
            + Raa * (1 - Ap) * (1 - Pa)
        ),
        I0
        * (
            Rpp * (1 + Aa) * (1 + Pa)
            + Rpa * (1 - Aa) * (1 + Pa)
            + Rap * (1 + Aa) * (1 - Pa)
            + Raa * (1 - Aa) * (1 - Pa)
        ),
    )
