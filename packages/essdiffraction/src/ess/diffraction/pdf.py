# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

r"""Pair distribution functions and related functions.

Here, we use the following definitions and naming convention:

.. list-table::
   :header-rows: 1

   * - Name
     - Definition
   * - Pair Correlation Function
     - :math:`G(r) = \frac{2}{\pi} \int_o^\infty\, Q (S(Q) - 1) \sin(Qr) \text{d}Q`
   * - Pair Distribution Function
     - :math:`g(r) = 1 + \frac{G(r)}{4 \pi \rho r}`
   * - Radial Distribution Function
     - :math:`\text{RDF}(r) = 4\pi r^2 \rho g(r)`
   * - Linearized Radial Distribution Function
     - :math:`T(r) = \frac{\text{RDF}(r)}{r}`
   * - Running Coordination Number
     - :math:`C(r) = \int_0^r\, \text{RDF}(r') \text{d}r'`

Where :math:`\rho` is the atomic density.

"""

import scipp as sc

from ess.reduce.uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties


def pair_correlation_function(
    s: sc.DataArray,
    r: sc.Variable,
    *,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode = UncertaintyBroadcastMode.drop,  # noqa: E501
    return_covariances: bool = False,
) -> sc.DataArray | tuple[sc.DataArray, sc.DataArray]:
    """Compute the pair correlation function from a structure factor.

    Computes the pair correlation function :math:`G(r)` from the overall
    scattering function :math:`S(Q)`.
    See `Review: Pair distribution functions from neutron total scattering for the study of local structure in disordered materials <https://www.sciencedirect.com/science/article/pii/S2773183922000374>`_
    for the definition of :math:`G(r)` or :mod:`ess.diffraction.pdf`.
    (Note, in the reference, the pair correlation function is denoted as :math:`D(r)`,
    but since :math:`G(r)` is the more common name, that is what is used here).

    The inputs to the function are:

    * A histogram representing :math:`S(Q)` with :math:`N` bins on a bin-edge grid with
      :math:`N+1` edges :math:`Q_j` for :math:`j=0\\ldots N`.
    * The bin-edge grid over :math:`r`. The output histogram representing :math:`G(r)`
      will be computed on this grid.

    In each output bin, the output is computed as:

    .. math::
        G_{i+\\frac{1}{2}} &= \\frac{2}{\\pi(r_{i+1}-r_i)} \\int_{r_i}^{r_{i+1}} \\int_{0}^\\infty (S(Q) - 1) Q \\sin(Q r) dQ \\ dr  \\\\
        &\\approx \\frac{2}{\\pi(r_{i+1}-r_i)} \\sum_{j=0}^{N-1} (S(Q)_{j+\\frac{1}{2}} - 1) (\\cos(\\bar{Q}_{j+\\frac{1}{2}} r_{i})-\\cos(\\bar{Q}_{j+\\frac{1}{2}} r_{i+1})) \\Delta Q_{j+\\frac{1}{2}}

    Note that in the above expression the subscript :math:`_{j+\\frac{1}{2}}` is used
    to denote quantities belonging to the :math:`j^\\text{th}` bin of a histogram,
    :math:`\\bar{Q}_{j+\\frac{1}{2}} = \\frac{Q_j + Q_{j+1}}{2}` and
    :math:`\\Delta Q_{j+\\frac{1}{2}} = Q_{j+1} - Q_{j}`.

    Parameters
    ----------
    s:
        1D DataArray representing :math:`S(Q)` with
        a bin-edge coordinate called ``'Q'``.
    r:
        1D array, bin-edges of output grid.
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
        Choose how uncertainties in S(Q) are broadcast to G(r).
        Defaults to ``UncertaintyBroadcastMode.drop``.
    return_covariances:
        If true the second output of the function will be a 2D array representing
        the covariance matrix of the entries in the first output.

    Returns
    -------
    g:
        1D DataArray representing :math:`G(r)` with a bin-edge coordinate called
        ``'r'`` that is the provided output grid.
    cov:
        2D DataArray representing the covariance matrix of the entries in ``g``.
        Only returned if ``return_covariances=True``.
    """  # noqa: E501
    q = s.coords['Q']
    qm = sc.midpoints(q)
    dq = q[1:] - q[:-1]
    dr = r[1:] - r[:-1]

    v = sc.cos(qm * r * sc.scalar(1, unit='rad'))
    v = v[r.dim, :-1] - v[r.dim, 1:]

    ioq = (s - sc.scalar(1.0, unit=s.unit)) * dq
    mat_ioq = broadcast_uncertainties(ioq, prototype=v, mode=uncertainty_broadcast_mode)
    c = 2 / sc.constants.pi / dr
    g = c * (v * mat_ioq).sum(q.dim)
    g = sc.DataArray(g.data, coords={'r': r})
    if return_covariances:
        cov_g = _covariance_of_matrix_vector_product(c * v, ioq)
        cov_g = sc.DataArray(
            cov_g, coords={d: r.rename_dims({r.dim: d}) for d in cov_g.dims}
        )
        return g, cov_g
    return g


def _covariance_of_matrix_vector_product(A, v):
    if A.variances is not None:
        raise ValueError('The expression is not valid if the matrix has variances.')
    v = sc.variances(v)
    if A.dims[1] != v.dim:
        A = A.transpose()
    cov = (sc.sqrt(v) * A).values
    cov = cov @ cov.T
    return sc.array(
        dims=[A.dims[0], A.dims[0] + '_2'], values=cov, unit=v.unit * A.unit**2
    )
