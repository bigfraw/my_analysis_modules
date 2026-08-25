'''
Coupled flux at a satellite receiver for a ground-to-satellite uplink through turbulence.

Implements the scintillation / beam-wander model from:

F. Dios, J. A. Rubio, A. Rodriguez, and A. Comeron, "Scintillation and beam-wander
analysis in an optical ground station-satellite uplink," Appl. Opt., AO, vol. 43,
no. 19, pp. 3866-3873, Jul. 2004, doi: 10.1364/AO.43.003866.
'''

import numpy as np
from scipy.special import gamma, hyp1f1
from fields import gaussz, zR
from general_atmospherics import scintillation_index_1D


# ---------------------------------------------------------------------------
# Beam geometry / coherence
# ---------------------------------------------------------------------------

def spherical_wave_coherence_diameter(k, L, cn2_vals, z):
    '''
    Spherical-wave coherence diameter r0s.

    r0s = [ 0.42*k^2 * integral_0^L( Cn2(z) * ((L-z)/L)^(5/3) dz ) ]^(-3/5)

    This is equation (3) of Dios et al. 2004, DOI 10.1364/AO.43.003866 (printed
    page 3868). The paper states it "for the uplink", so the weight
    ((L-z)/L)^(5/3) is TRANSMITTER-referred. Andrews and Phillips 2nd ed.,
    chapter 6, equations (115) and (116), print the mirror weight (z/L)^(5/3)
    for the receiver-referred (downlink) radius. The two are a plane-of-
    reference difference, not a fault. Do not flip this weight.

    Parameters:
        k (float) : Optical wave number (2*pi/lambda).
        L (float) : Propagation distance [m].
        cn2_vals (numpy.ndarray) : Cn2 values sampled at `z`.
        z (numpy.ndarray) : Path coordinates for `cn2_vals` [m].

    Returns:
        float : Spherical-wave coherence diameter r0s.
    '''
    z = np.asarray(z)
    if L < np.max(z):
        z_grid = z[z <= L]
        cn2_vals = cn2_vals[z <= L]
    else:
        z_grid = z

    weight = ((L - z_grid) / L) ** (5.0 / 3.0)
    path_integral = np.trapz(cn2_vals * weight, z_grid)

    return (0.42 * (k ** 2) * path_integral) ** (-3 / 5)


def short_term_beam_waist(W0, L, Z0, k, r0s, return_squared=False, w_free=None):
    '''
    Short-term beam waist at propagation distance z=L.

    W_ST^2(L) = W0^2 * (1 + L^2/Z0^2) + 2*{ [4.2*L/(k*r0s)] * [1 - 0.26*(r0s/W0)^(1/3)] }^2

    The W0^2*(1 + L^2/Z0^2) term is the free-space width at L for a COLLIMATED
    beam. For a deliberately diverged (or focused) transmitter, pass the actual
    free-space width as `w_free` and it replaces that term; `Z0` is then unused.
    The turbulence term is unaffected -- it is a spreading angle set by r0s, not
    by the transmitter geometry -- but `W0` must stay the physical beam radius
    at the launch aperture, since the 0.26*(r0s/W0)^(1/3) correction (which
    removes the wander contribution) is an aperture-size effect.

    Note that correction turns negative for r0s/W0 > (1/0.26)^3 ~ 57, i.e. a
    small aperture in very weak turbulence. It is squared, so the result stays
    positive, and the 4.2*L/(k*r0s) prefactor shrinks faster than the correction
    grows (the term falls off as r0s^(-4/3)), so the waist still tends to the
    free-space value -- but the factor itself is outside its intended range there.

    Parameters:
        W0 (float) : Beam radius at the transmit aperture [m].
        L (float) : Propagation distance [m].
        Z0 (float) : Rayleigh range [m]. Ignored when `w_free` is given.
        k (float) : Optical wave number (2*pi/lambda).
        r0s (float) : Spherical-wave coherence diameter [m].
        return_squared (bool, optional) : If True, return W_ST^2 instead of W_ST.
        w_free (float, optional) : Free-space (turbulence-free) beam radius at
            L [m], for a non-collimated transmitter.

    Returns:
        float : Short-term beam waist (or its square).
    '''
    if w_free is None:
        geometric_term = W0 ** 2 * (1.0 + (L ** 2) / (Z0 ** 2))
    else:
        geometric_term = w_free ** 2
    turbulence_factor = (4.2 * L / (k * r0s)) * (1.0 - 0.26 * (r0s / W0) ** (1 / 3))
    w_st_squared = geometric_term + 2.0 * turbulence_factor ** 2

    if return_squared:
        return w_st_squared
    return np.sqrt(np.maximum(w_st_squared, 0.0))


def beam_wander_variance(L, cn2, ws, z):
    '''
    Beam-wander variance <beta^2>.

    <beta^2> = 2.07 * integral_0^L( Cn2(z) * (L-z)^2 * [1/Ws(z)]^(1/3) dz )

    This is equation (11) of Dios et al. 2004, DOI 10.1364/AO.43.003866
    (printed page 3868). The constant, the integrand and the path weight agree
    with the paper. Dios does not derive equation (11). He takes it from
    Belmonte, Applied Optics 39, 5426 (2000), DOI 10.1364/AO.39.005426.

    CONVENTION: <beta^2> is the RADIAL (two-axis) displacement variance. Dios
    equation (9) gives beta = sqrt(beta_x^2 + beta_y^2), and equation (10) gives
    <beta_x^2> = <beta_y^2> = 0.5*<beta^2>. So a caller that draws one Cartesian
    axis must use the variance 0.5*<beta^2>.

    KNOWN DIFFERENCE from Andrews and Phillips, 2nd ed., DOI 10.1117/3.626196.
    Chapter 6, equation (93) (printed page 203) has the SAME integrand with the
    constant 7.25, and its infinite-outer-scale form, equation (94) (printed
    page 204), gives <r_c^2> = 2.42 Cn2 L^3 W0^(-1/3) for a collimated beam.
    For a constant Cn2 and Ws = W0, the constant 2.07 above gives
    0.69 Cn2 L^3 W0^(-1/3), which is 3.50 times lower. The Andrews quantity is
    also a radial variance, so the axis convention does NOT explain the gap.

    KEEP 2.07. Two split-step wave-optics simulations validate this form. Dios
    figure 3 (printed page 3871) compares equation (11) with an FFT-BPM
    simulation of the same uplink, and the two agree closely. Belmonte 2000
    (DOI 10.1364/AO.39.005426, the source Dios takes equation (11) from) prints
    the same 2.07 form as his equation (21) (printed page 5435) and compares it
    with his own phase-screen simulation in figures 11 and 12; it matches in
    weak-to-moderate turbulence. A factor of 3.50 would be plain on either plot.

    The two constants are two derivations of the SAME radial quantity. The 2.07
    is the Yura / Mironov-Nosov IMAGE-MOTION level arm (Belmonte references 46 to
    48), which keeps the centroid tilt only. The Andrews 7.25 is the beam-wave
    SPECTRAL-FILTER route (chapter 6, equations (88) and (89)), which keeps all
    large-scale refraction, so it is the larger constant. Belmonte measures the
    true centroid displacement directly (his equation (20)) and gets the 2.07
    value, so the difference is settled by simulation, not on paper. See the C-01
    closure in olb/docs/andrews-crosscheck.md.

    Ws(z) is the beam radius at z. Dios prints the symbol W_s(z) but does not
    define it. This function uses the free-space (diffracted) radius. For an
    uplink that choice changes almost nothing, because all the turbulence is in
    the first 20 km, where Ws(z) stays near W0.

    Parameters:
        L (float) : Propagation distance [m].
        cn2 (numpy.ndarray) : Cn2 profile sampled at `z`.
        ws (numpy.ndarray) : Beam radius profile Ws(z) [m], sampled at `z`.
        z (numpy.ndarray) : Path coordinates [m].

    Returns:
        float : Beam-wander variance <beta^2> [m^2].
    '''
    z_grid = np.asarray(z)
    cn2_vals = np.asarray(cn2, dtype=float)
    ws_vals = np.asarray(ws, dtype=float)
    integrand = cn2_vals * ((L - z_grid) ** 2) * ((1.0 / ws_vals) ** (1 / 3))
    return 2.07 * np.trapz(integrand, z_grid)


def long_term_beam_waist(w_st, beta2):
    '''
    Long-term beam waist, combining short-term spreading with beam wander.

    W_LT^2(L) = W_ST^2(L) + 2*<beta^2>

    This is equation (1) of Dios et al. 2004, DOI 10.1364/AO.43.003866 (printed
    page 3867). The paper repeats it in equation (29), printed page 3870. The
    factor 2 is the paper's own factor on a RADIAL <beta^2> (see
    `beam_wander_variance`). It is NOT a per-axis to radial conversion.

    Andrews and Phillips, 2nd ed., DOI 10.1117/3.626196, chapter 6, equation
    (100) (printed page 205), puts the factor 1 on a radial <r_c^2>. With the
    constant of each source, the wander part of W_LT^2 is
    1.38 Cn2 L^3 W0^(-1/3) by Dios and 2.42 Cn2 L^3 W0^(-1/3) by Andrews. So
    the two combination rules differ by 1.75, not by 3.50. The Dios factor 2 and
    the Dios constant 2.07 partially cancel the gap.

    Parameters:
        w_st (float) : Short-term beam waist [m].
        beta2 (float) : Beam-wander variance <beta^2> [m^2].

    Returns:
        float : Long-term beam waist [m].
    '''
    return np.sqrt(w_st ** 2 + 2 * beta2)


def long_term_beam_waist_collimated(W0, L, Z0, k, r0s, return_squared=False, w_free=None):
    '''
    Long-term beam waist for a collimated beam, equation (2) of Dios et al. 2004.

    W_LT^2(L) = W0^2 * (1 + L^2/Z0^2) + 2*(4*L/(k*r0s))^2

    This is the average irradiance width over a long exposure, so it already
    contains the beam wander; it needs no separate <beta^2> term. Compare with
    `short_term_beam_waist`, which carries the extra 1 - 0.26*(r0s/W0)^(1/3)
    factor that removes the wander contribution.

    Passing `w_free` replaces the collimated free-space term W0^2*(1 + L^2/Z0^2)
    with the free-space width of an arbitrary transmitter (diverged or focused),
    leaving the r0s spreading term untouched. Note that the paper states eq. (2)
    for collimated beams, so this is an extension of it, justified by the
    turbulence term being a transmitter-independent spreading angle; the
    `long_term_beam_waist_rytov` route carries the general beam parameters
    natively and is the better cross-check when the beam is strongly diverged.

    Parameters:
        W0 (float) : Beam radius at the transmit aperture [m]. Unused when
            `w_free` is given.
        L (float) : Propagation distance [m].
        Z0 (float) : Rayleigh range [m]. Ignored when `w_free` is given.
        k (float) : Optical wave number (2*pi/lambda).
        r0s (float) : Spherical-wave coherence diameter [m].
        return_squared (bool, optional) : If True, return W_LT^2 instead of W_LT.
        w_free (float, optional) : Free-space (turbulence-free) beam radius at
            L [m], for a non-collimated transmitter.

    Returns:
        float : Long-term beam waist (or its square).
    '''
    if w_free is None:
        geometric_term = W0 ** 2 * (1.0 + (L ** 2) / (Z0 ** 2))
    else:
        geometric_term = w_free ** 2
    turbulence_term = 2.0 * (4.0 * L / (k * r0s)) ** 2
    w_lt_squared = geometric_term + turbulence_term

    if return_squared:
        return w_lt_squared
    return np.sqrt(np.maximum(w_lt_squared, 0.0))


def long_term_beam_waist_rytov(W0, L, k, cn2_vals, z, return_squared=False, w_free=None):
    '''
    Long-term beam waist from the second-order Rytov approximation (Andrews
    et al.), equations (4)-(6) of Dios et al. 2004.

    W_LT(L) = W(L) * sqrt(1 + G_u)

        G_u = 4*pi^2*k^2 * int_0^L int_0^inf kappa*Phi_n(z,kappa)
              * {1 - exp[-(Lambda*L*kappa^2/k)*(1 - z/L)^2]} dkappa dz

        Lambda = 2*L/(k*W^2(L))

    with W(L) the free-space diffraction-limited width. For a Kolmogorov
    spectrum, Phi_n = 0.033*Cn2(z)*kappa^(-11/3), the kappa integral has the
    closed form

        int_0^inf kappa^(-8/3)*(1 - exp(-a*kappa^2)) dkappa
            = -0.5*gamma(-5/6)*a^(5/6),     a = (Lambda*L/k)*(1 - z/L)^2

    so only the path integral over z is done numerically here:

        G_u = 4*pi^2*k^2 * 0.033 * (-0.5*gamma(-5/6)) * (Lambda*L/k)^(5/6)
              * int_0^L Cn2(z)*(1 - z/L)^(5/3) dz

    For a uniform Cn2 this reduces to the familiar weak-fluctuation result
    G_u = 1.33*sigma_R^2*Lambda^(5/6), with sigma_R^2 = 1.23*Cn2*k^(7/6)*L^(11/6).

    Both W(L) and Lambda are properties of the free-space beam, so a diverged or
    focused transmitter is handled by passing its free-space width at L as
    `w_free`: Lambda = 2*L/(k*w_free^2) then follows and the whole expression
    generalises without further change (`W0` becomes unused). This is the
    natural route for non-collimated beams -- the Andrews formulation is written
    in the general beam parameters to begin with.

    Parameters:
        W0 (float) : Beam radius at the transmit aperture [m], assumed to be a
            collimated waist. Unused when `w_free` is given.
        L (float) : Propagation distance [m].
        k (float) : Optical wave number (2*pi/lambda).
        cn2_vals (numpy.ndarray) : Cn2 values sampled at `z`.
        z (numpy.ndarray) : Path coordinates for `cn2_vals` [m].
        return_squared (bool, optional) : If True, return W_LT^2 instead of W_LT.
        w_free (float, optional) : Free-space (turbulence-free) beam radius at
            L [m], for a non-collimated transmitter.

    Returns:
        float : Long-term beam waist (or its square).
    '''
    z = np.asarray(z)
    cn2_vals = np.asarray(cn2_vals, dtype=float)
    if L < np.max(z):
        z_grid = z[z <= L]
        cn2_vals = cn2_vals[z <= L]
    else:
        z_grid = z

    wL = gaussz(W0, L, 2 * np.pi / k) if w_free is None else w_free
    Lambda = 2 * L / (k * wL ** 2)             # equation (6)

    path_integral = np.trapz(cn2_vals * (1.0 - z_grid / L) ** (5 / 3), z_grid)
    coefficient = 4 * np.pi ** 2 * k ** 2 * 0.033 * (-0.5 * gamma(-5 / 6))
    G_u = coefficient * (Lambda * L / k) ** (5 / 6) * path_integral

    w_lt_squared = wL ** 2 * (1.0 + G_u)

    if return_squared:
        return w_lt_squared
    return np.sqrt(np.maximum(w_lt_squared, 0.0))


# ---------------------------------------------------------------------------
# Scintillation index (Rytov-based, on/off axis)
# ---------------------------------------------------------------------------

def _lambda_function(L, k0, wL):
    return 2 * L / (k0 * wL ** 2)


def _theta_function(L, Z0):
    return (1 + (L / Z0) ** 2) ** (-1)


def _A(z, L, k_0, wL):
    '''Equation (17) of Dios et al. 2004.'''
    Lambda = _lambda_function(L, k_0, wL)
    return (Lambda * L / k_0) * ((L - z) / L) ** 2


def _B(z, L, k_0, Z0):
    '''Equation (18) of Dios et al. 2004.'''
    Theta = _theta_function(L, Z0)
    return (L / k_0) * ((L - z) / L) * (Theta + (1 - Theta) * z / L)


def on_axis_scintillation_index(L, k_0, wL, Z0, cn2s, z_points):
    '''
    On-axis scintillation index sigma_r^2(0, L), equation (16) of Dios et al. 2004.

    Parameters:
        L (float) : Propagation distance [m].
        k_0 (float) : Optical wave number (2*pi/lambda).
        wL (float) : Beam waist at distance L [m].
        Z0 (float) : Rayleigh range [m].
        cn2s (numpy.ndarray) : Cn2 values sampled at `z_points`.
        z_points (numpy.ndarray) : Altitude points [m] for `cn2s`.

    Returns:
        float : On-axis scintillation index.
    '''
    a_z = _A(z_points, L, k_0, wL)
    b_z = _B(z_points, L, k_0, Z0)
    ratio = b_z / a_z

    # The cosine multiplies ONLY the second term. Dios et al. 2004 equation (16)
    # (DOI 10.1364/AO.43.003866) and Andrews and Phillips 2nd ed. chapter 8,
    # equation (17) (printed page 263, DOI 10.1117/3.626196) both give
    # A^(5/6) - (A^2 + B^2)^(5/12) * cos[(5/6) arctan(B/A)]. Factor out A^(5/6)
    # to get the form below. Before 2026-08 a parenthesis closed too early, so
    # the cosine multiplied the full bracket.
    integrand = cn2s * a_z ** (5 / 6) * (1 - (1 + ratio ** 2) ** (5 / 12) * np.cos((5 / 6) * np.arctan(ratio)))
    result = np.trapz(integrand, z_points)

    coefficient = 4 * np.pi ** 2 * k_0 ** 2 * gamma(-5 / 6) * 0.033
    return coefficient * result


def off_axis_scintillation_index(L, k_0, wL, cn2s, z_points, r):
    '''
    Off-axis scintillation index sigma_r,L^2(r, L), equation (20) of Dios et al. 2004.

    Parameters:
        L (float) : Propagation distance [m].
        k_0 (float) : Optical wave number (2*pi/lambda).
        wL (float) : Beam waist at distance L [m].
        cn2s (numpy.ndarray) : Cn2 values sampled at `z_points`.
        z_points (numpy.ndarray) : Altitude points [m] for `cn2s`.
        r (float) : Off-axis distance at which to evaluate [m].

    Returns:
        float : Off-axis scintillation index at radius `r`.
    '''
    a_z = _A(z_points, L, k_0, wL)
    hyp_arg = 2 * r ** 2 / wL ** 2
    integrand = cn2s * a_z ** (5 / 6) * (hyp1f1(-5 / 6, 1, hyp_arg) - 1)
    result = np.trapz(integrand, z_points)

    coefficient = 4 * np.pi ** 2 * k_0 ** 2 * gamma(-5 / 6) * 0.033
    return coefficient * result


# ---------------------------------------------------------------------------
# Irradiance (flux) at the receiver
# ---------------------------------------------------------------------------

def on_axis_irradiance(beta, wst_L, xi_beta):
    '''
    Instantaneous on-axis irradiance, combining beam-wander displacement `beta`
    with a log-amplitude turbulence fluctuation `xi_beta`.

    Parameters:
        beta (float or numpy.ndarray) : Instantaneous beam-wander displacement [m].
        wst_L (float) : Short-term beam waist at the receiver [m].
        xi_beta (float or numpy.ndarray) : Log-amplitude fluctuation sample.

    Returns:
        float or numpy.ndarray : Normalized on-axis irradiance.
    '''
    return np.exp(2 * xi_beta) * np.exp(-2 * beta ** 2 / wst_L ** 2)


def mean_off_axis_irradiance(r, wlt_L):
    '''
    Mean off-axis irradiance profile (long-term beam spread only).

    Parameters:
        r (float or numpy.ndarray) : Off-axis distance [m].
        wlt_L (float) : Long-term beam waist at the receiver [m].

    Returns:
        float or numpy.ndarray : Normalized mean off-axis irradiance.
    '''
    return np.exp(-2 * r ** 2 / wlt_L ** 2)


# ---------------------------------------------------------------------------
# Top level: coupled flux at the satellite receiver
# ---------------------------------------------------------------------------

def coupled_flux_sample(beta, cn2_profile, Z0, hs, L, k_0, wL, wL_lt):
    '''
    Draw a single realization of the turbulence-induced flux fluctuation at
    beam-wander displacement `beta`.

    Parameters:
        beta (float) : Instantaneous beam-wander displacement [m].
        cn2_profile (numpy.ndarray) : Cn2 profile sampled at `hs`.
        Z0 (float) : Rayleigh range [m].
        hs (numpy.ndarray) : Altitude points [m] for `cn2_profile`.
        L (float) : Propagation distance [m].
        k_0 (float) : Optical wave number (2*pi/lambda).
        wL (float) : Free-space (diffraction-limited) beam radius at L [m].
            This is the W(L) of Dios equation (15), which sets Lambda.
        wL_lt (float) : Long-term beam waist at distance L [m]. Equation (24)
            uses it for the mean-irradiance weight of equation (25).

    Returns:
        tuple : (xi, xi_on_axis, sigma2_x, sigma2_x_on_axis, sigma2_gauss, sigma2_gauss_on_axis)
    '''
    sigma2_off = off_axis_scintillation_index(L, k_0, wL, cn2_profile, hs, beta)
    sigma2_on = on_axis_scintillation_index(L, k_0, wL, Z0, cn2_profile, hs)

    # Dios et al. 2004 equation (25) (DOI 10.1364/AO.43.003866, printed page
    # 3870): sigma2_I,Gb = (sigma2_I + sigma2_I,r) * <I>^2, where <I> is the
    # mean irradiance of equation (24) at the wander position beta. Equations
    # (13), (16) and (20) normalize the index to the LOCAL mean irradiance;
    # equation (25) re-normalizes it to the mean irradiance at the BEAM CENTER,
    # which is the normalization that equation (26) needs. Section 5, step
    # (c)(ii) of the paper tells you to use equation (25) at this point.
    #
    # An earlier patch removed this weight, because Andrews and Phillips 2nd ed.
    # chapter 8, equations (9) and (15) (DOI 10.1117/3.626196) keep the local
    # normalization. But this module implements Dios, and the removal made it
    # disagree with the paper it cites. The weight is back (2026-08-25).
    I_off = mean_off_axis_irradiance(beta, wL_lt)
    sigma2_gauss = (sigma2_on + sigma2_off) * I_off ** 2
    sigma2_gauss_on_axis = sigma2_on * I_off ** 2

    sigma2_x = 0.25 * np.log(1 + sigma2_gauss)
    sigma2_x_on_axis = 0.25 * np.log(1 + sigma2_gauss_on_axis)

    xi = np.random.normal(-sigma2_x, np.sqrt(sigma2_x), 1)
    xi_on_axis = np.random.normal(-sigma2_x_on_axis, np.sqrt(sigma2_x_on_axis), 1)

    return xi, xi_on_axis, sigma2_x, sigma2_x_on_axis, sigma2_gauss, sigma2_gauss_on_axis


def coupled_flux_montecarlo(w0, elevation, labda, L, hs, n_samples, n_apertures,
                             cn2_profile=None, hv57_A=1.7e-14):
    '''
    Monte Carlo estimate of the coupled flux at a satellite receiver for a
    ground-to-satellite uplink through turbulence.

    Parameters:
        w0 (float) : Transmit beam waist radius [m].
        elevation (float) : Ground station elevation angle [deg]; scales
            `cn2_profile` by airmass = 1/sin(elevation).
        labda (float) : Wavelength [m].
        L (float) : Ground-to-satellite path length [m].
        hs (numpy.ndarray) : Altitude points [m] the Cn2 profile is sampled at.
        n_samples (int) : Number of Monte Carlo beam-wander/scintillation draws.
        n_apertures (int) : Number of independent on-axis samples averaged per
            receiver aperture (models receive-side aperture averaging).
        cn2_profile (numpy.ndarray, optional) : Continuous Cn2(h) profile
            [m^(-2/3)] at zenith (airmass=1), matching `hs` -- NOT a
            per-layer/pre-integrated Cn2*dh quantity (e.g. from
            `turbulence_models.HV57_Bufton_profile`, which has different
            units and will blow up these formulas). If None, generated from
            `fast.turbulence_models.HV57(hs, A=hv57_A)` (requires the `fast`
            package).
        hv57_A (float, optional) : Hufnagel-Valley ground-level Cn2 scale,
            used only when `cn2_profile` is not supplied.

    Returns:
        dict : Fried parameter (`r0s`, the spherical-wave coherence
        diameter), beam sizes (`w_st`, `w_lt`), mean/empirical scintillation
        indices, and the per-sample flux arrays `Is`, `Is_on`, `Is_summed`.
    '''
    if cn2_profile is None:
        from fast import turbulence_models
        cn2_profile = turbulence_models.HV57(hs, A=hv57_A)

    airmass = 1 / np.sin(np.radians(elevation))
    cn2_profile = np.asarray(cn2_profile) * airmass

    Z0 = zR(w0, labda)
    k_0 = 2 * np.pi / labda
    wL = gaussz(w0, L, labda)

    ws = gaussz(w0, hs, labda)
    beta2 = beam_wander_variance(L, cn2_profile, ws, hs)
    r0s = spherical_wave_coherence_diameter(k_0, L, cn2_profile, hs)
    w_st = short_term_beam_waist(w0, L, Z0, k_0, r0s)
    w_lt = long_term_beam_waist(w_st, beta2)

    xis = np.zeros(n_samples)
    xis_on_axis = np.zeros(n_samples)
    betas = np.zeros(n_samples)
    sigma2_xs = np.zeros(n_samples)
    sigma2_xs_on_axis = np.zeros(n_samples)
    sigma2_gauss = np.zeros(n_samples)

    for i in range(n_samples):
        betax = np.random.normal(0, np.sqrt(0.5 * beta2), 1)
        betay = np.random.normal(0, np.sqrt(0.5 * beta2), 1)
        beta = np.sqrt(betax ** 2 + betay ** 2)

        xi, xi_on, s2x, s2x_on, s2g, _ = coupled_flux_sample(beta, cn2_profile, Z0, hs, L, k_0, wL, w_lt)
        betas[i] = np.squeeze(beta)
        xis[i] = np.squeeze(xi)
        xis_on_axis[i] = np.squeeze(xi_on)
        sigma2_xs[i] = np.squeeze(s2x)
        sigma2_xs_on_axis[i] = np.squeeze(s2x_on)
        sigma2_gauss[i] = np.squeeze(s2g)

    Is = on_axis_irradiance(betas, w_st, xis)
    Is_on = on_axis_irradiance(betas, w_st, xis_on_axis)
    n_blocks = Is.shape[0] // n_apertures
    Is_summed = np.mean(Is[: n_blocks * n_apertures].reshape(n_blocks, n_apertures), axis=1)

    return {
        "w0": w0,
        "elevation": elevation,
        "r0s": r0s,
        "w_st": w_st,
        "w_lt": w_lt,
        "sigma2_x_mean": float(np.mean(sigma2_xs)),
        "sigma2_x_on_mean": float(np.mean(sigma2_xs_on_axis)),
        "sigma2_i_expected_mean": float(np.mean(sigma2_gauss)),
        "sigma2_i_empirical": float(scintillation_index_1D(Is)),
        "Is": Is,
        "Is_on": Is_on,
        "Is_summed": Is_summed,
    }


# ---------------------------------------------------------------------------
# Cross-validation of the three long-term beam waist models
# ---------------------------------------------------------------------------

def uniform_cn2_for_r0s(r0s, k, L, n_z=2001):
    '''
    Uniform Cn2 path that produces a target spherical-wave coherence diameter,
    i.e. equation (3) inverted.

    For a constant Cn2 the path weighting integrates to
    int_0^L ((L-z)/L)^(5/3) dz = (3/8)*L, so

        Cn2 = r0s^(-5/3) / (0.42 * k^2 * (3/8) * L).

    Lets a sweep be parametrised by the Fried parameter directly instead of by
    turbulence profiles. Note `r0s` is the SPHERICAL-wave coherence diameter
    used throughout this module; for a uniform path the plane-wave Fried
    parameter is smaller by (8/3)^(3/5) ~ 1.81.

    Parameters:
        r0s (float) : Target spherical-wave coherence diameter [m].
        k (float) : Optical wave number (2*pi/lambda).
        L (float) : Propagation distance [m].
        n_z (int, optional) : Number of path samples.

    Returns:
        tuple : (cn2_vals, z) with `cn2_vals` constant [m^(-2/3)] and `z` [m].
    '''
    z = np.linspace(0.0, L, n_z)
    cn2 = r0s ** (-5 / 3) / (0.42 * k ** 2 * (3 / 8) * L)
    return np.full_like(z, cn2), z


def compare_long_term_models(w0, r0s, L, labda=1550e-9, n_z=2001):
    '''
    Evaluate the three long-term beam waist models at one (W0, r0s) point.

    The models are:
        'wander'    : short-term spread combined with beam wander,
                      W_LT = sqrt(W_ST^2 + 2*<beta^2>)  (`long_term_beam_waist`)
        'collimated': Dios equation (2)   (`long_term_beam_waist_collimated`)
        'rytov'     : Andrews equations (4)-(6)  (`long_term_beam_waist_rytov`)

    The first two are the same paper by two routes and should track each other;
    the third is a second-order Rytov expansion and is only expected to agree
    while the fluctuations are weak (sigma_R^2 < 1).

    Parameters:
        w0 (float) : Transmit beam waist radius [m].
        r0s (float) : Spherical-wave coherence diameter [m].
        L (float) : Propagation distance [m].
        labda (float, optional) : Wavelength [m].
        n_z (int, optional) : Number of path samples for the integrals.

    Returns:
        dict : the three waists, the free-space and short-term widths, the
        wander variance, and the regime diagnostics `sigma_R2`, `Lambda`,
        `w0_over_r0s`.
    '''
    k = 2 * np.pi / labda
    Z0 = zR(w0, labda)
    cn2_vals, z = uniform_cn2_for_r0s(r0s, k, L, n_z=n_z)

    wL = gaussz(w0, L, labda)
    ws = gaussz(w0, z, labda)

    w_st = short_term_beam_waist(w0, L, Z0, k, r0s)
    beta2 = beam_wander_variance(L, cn2_vals, ws, z)

    return {
        "w0": w0,
        "r0s": r0s,
        "wander": long_term_beam_waist(w_st, beta2),
        "collimated": long_term_beam_waist_collimated(w0, L, Z0, k, r0s),
        "rytov": long_term_beam_waist_rytov(w0, L, k, cn2_vals, z),
        "w_free": wL,
        "w_st": w_st,
        "beta2": beta2,
        "sigma_R2": 1.23 * cn2_vals[0] * k ** (7 / 6) * L ** (11 / 6),
        "Lambda": 2 * L / (k * wL ** 2),
        "w0_over_r0s": w0 / r0s,
    }


def cross_validate_long_term_waist(w0s, r0ss, L=20e3, labda=1550e-9, n_z=2001,
                                   weak_limit=1.0, verbose=True):
    '''
    Sweep the three long-term beam waist models over a grid of transmit waists
    and Fried parameters, and report where they agree.

    Every quantity is returned as a 2D array of shape (len(w0s), len(r0ss)).
    Agreement is summarised over the weak-fluctuation subset sigma_R^2 <
    `weak_limit`, which is the only region where all three models claim to be
    valid; outside it the Rytov result is expected to run away.

    Parameters:
        w0s (array_like) : Transmit beam waist radii to sweep [m].
        r0ss (array_like) : Spherical-wave coherence diameters to sweep [m].
        L (float, optional) : Propagation distance [m].
        labda (float, optional) : Wavelength [m].
        n_z (int, optional) : Number of path samples for the integrals.
        weak_limit (float, optional) : sigma_R^2 below which the models are
            compared for agreement.
        verbose (bool, optional) : Print the comparison tables.

    Returns:
        dict : `w0s`, `r0ss`, the three waist grids (`wander`, `collimated`,
        `rytov`), `w_free`, `w_st`, `sigma_R2`, `Lambda`, the pairwise relative
        deviations (`dev_coll_wander`, `dev_rytov_wander`, `dev_rytov_coll`)
        and the weak-fluctuation mask `weak`.
    '''
    w0s = np.atleast_1d(np.asarray(w0s, dtype=float))
    r0ss = np.atleast_1d(np.asarray(r0ss, dtype=float))

    keys = ("wander", "collimated", "rytov", "w_free", "w_st", "sigma_R2", "Lambda")
    out = {key: np.zeros((w0s.size, r0ss.size)) for key in keys}

    for i, w0 in enumerate(w0s):
        for j, r0s in enumerate(r0ss):
            point = compare_long_term_models(w0, r0s, L, labda=labda, n_z=n_z)
            for key in keys:
                out[key][i, j] = point[key]

    out["dev_coll_wander"] = out["collimated"] / out["wander"] - 1.0
    out["dev_rytov_wander"] = out["rytov"] / out["wander"] - 1.0
    out["dev_rytov_coll"] = out["rytov"] / out["collimated"] - 1.0
    out["weak"] = out["sigma_R2"] < weak_limit
    out["w0s"] = w0s
    out["r0ss"] = r0ss
    out["L"] = L
    out["labda"] = labda

    if verbose:
        _print_cross_validation(out, weak_limit)

    return out


def _print_cross_validation(res, weak_limit):
    '''Print the tables produced by `cross_validate_long_term_waist`.'''
    w0s, r0ss = res["w0s"], res["r0ss"]
    header = "  W0 [cm] |" + "".join(f"{r * 100:>9.1f}" for r in r0ss)

    print(f"\nL = {res['L'] / 1e3:g} km, lambda = {res['labda'] * 1e9:g} nm"
          f"   (columns: r0s [cm])")

    for name, label in (("wander", "W_LT [cm]: short-term + wander"),
                        ("collimated", "W_LT [cm]: Dios eq. (2)"),
                        ("rytov", "W_LT [cm]: Andrews eqs. (4)-(6)")):
        print(f"\n{label}")
        print(header)
        for i, w0 in enumerate(w0s):
            row = "".join(f"{v * 100:>9.3g}" for v in res[name][i])
            print(f"{w0 * 100:>9.1f} |" + row)

    # sigma_R^2 is a property of the path alone, so one row covers every W0
    print("\nsigma_R^2 (uniform path, same for every W0)")
    print(header)
    print(" " * 9 + " |" + "".join(f"{v:>9.3g}" for v in res["sigma_R2"][0]))

    weak = res["weak"]
    print(f"\nAgreement where sigma_R^2 < {weak_limit:g} "
          f"({weak.sum()} of {weak.size} grid points):")
    for key, label in (("dev_coll_wander", "Dios eq.(2) vs short-term+wander"),
                       ("dev_rytov_wander", "Rytov      vs short-term+wander"),
                       ("dev_rytov_coll", "Rytov      vs Dios eq.(2)      ")):
        if weak.any():
            dev = np.abs(res[key][weak])
            print(f"  {label} : median {np.median(dev) * 100:6.2f} %, "
                  f"max {dev.max() * 100:6.2f} %")
        else:
            print(f"  {label} : no weak-fluctuation points in this grid")

    strong = ~weak
    if strong.any():
        dev = np.abs(res["dev_rytov_wander"][strong])
        print(f"  Rytov vs short-term+wander, sigma_R^2 >= {weak_limit:g} "
              f"({strong.sum()} points) : median {np.median(dev) * 100:6.2f} %, "
              f"max {dev.max() * 100:6.2f} %  <- expected to break down")


def cross_validation_demo(plot=False):
    '''
    Cross-validate the three long-term beam waist models over a spread of
    transmit waists and Fried parameters, from weak to strong turbulence.
    '''
    w0s = np.array([0.02, 0.05, 0.10, 0.25])
    r0ss = np.array([0.02, 0.05, 0.10, 0.30, 1.00])

    res = cross_validate_long_term_waist(w0s, r0ss, L=20e3, labda=1550e-9)

    for key in ("wander", "collimated", "rytov"):
        assert np.all(np.isfinite(res[key])) and np.all(res[key] > 0)
        # every model must be at least the free-space diffraction width
        assert np.all(res[key] >= res["w_free"] * (1 - 1e-9))

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, w0s.size, figsize=(4 * w0s.size, 3.5), sharey=True)
        for i, (ax, w0) in enumerate(zip(np.atleast_1d(axes), w0s)):
            for key, style in (("wander", "-"), ("collimated", "--"), ("rytov", ":")):
                ax.loglog(r0ss, res[key][i] * 100, style, marker="o", label=key)
            ax.loglog(r0ss, res["w_free"][i] * 100, color="k", alpha=0.4, label="free space")
            ax.set_title(f"W0 = {w0 * 100:g} cm")
            ax.set_xlabel("r0s [m]")
        np.atleast_1d(axes)[0].set_ylabel("W_LT [cm]")
        np.atleast_1d(axes)[-1].legend()
        fig.tight_layout()
        plt.show()

    print("cross_validation_demo() OK")
    return res


def demo():
    '''Small Monte Carlo run, sanity-checking the coupled flux output.'''
    try:
        from fast import turbulence_models
    except ImportError:
        print("Skipping demo: the 'fast' package (turbulence_models) is not installed.")
        return

    hs = np.logspace(np.log10(1), np.log10(20e3), 20)
    cn2 = turbulence_models.HV57(hs, A=1.7e-14)

    result = coupled_flux_montecarlo(
        w0=0.038, elevation=30, labda=1550e-9, L=600e3,
        hs=hs, n_samples=1000, n_apertures=1, cn2_profile=cn2,
    )

    # plot of the flux distribution
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(result["Is"], bins=30, density=True, alpha=0.5, label="Is")
    plt.hist(result["Is_on"], bins=30, density=True, alpha=0.5, label="Is_on")
    plt.xlabel("Normalized on-axis irradiance")
    plt.ylabel("Probability density")
    plt.title("Coupled flux distribution at satellite receiver")
    plt.legend()
    plt.show()

    assert np.all(np.isfinite(result["Is"])) and np.all(result["Is"] >= 0)
    assert np.all(np.isfinite(result["Is_on"])) and np.all(result["Is_on"] >= 0)
    assert np.isfinite(result["sigma2_i_empirical"]) and result["sigma2_i_empirical"] >= 0

    print(f"w_st={result['w_st']:.4f} m, w_lt={result['w_lt']:.4f} m")
    print(f"sigma2_i (expected, empirical) = {result['sigma2_i_expected_mean']:.4e}, "
          f"{result['sigma2_i_empirical']:.4e}")
    print("demo() OK")


if __name__ == "__main__":
    cross_validation_demo()
    demo()
