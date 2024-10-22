from ..spectroscopy.types import FinalEnergy
from .types import ArcNumber


def arc_number(final_energy: FinalEnergy) -> ArcNumber:
    """Calculate BIFROST arc index number from pixel final energy

    The BIFROST analyzers are each set to diffract an
    energy in the set (2.7, 3.2, 3.8, 4.4. 5.0) meV.
    This energy is only valid for the central point of the center
    tube of the associated detector triplet. All other pixels
    will have a final energy slightly higher or lower.

    This function assigns the closest arc number indexing the
    ordered set above.

    Parameters
    ----------
    final_energy: scipp.Variable
        The per-pixel (or event) final neutron energy

    Returns
    -------
    :
        The arc index of the analyzer from which the neutron scattered
    """
    import scipp as sc

    minimum = sc.scalar(2.7, unit='meV')
    step = sc.scalar(0.575, unit='meV')
    return sc.round((final_energy - minimum) / step).to(dtype='int')


providers = (arc_number,)
