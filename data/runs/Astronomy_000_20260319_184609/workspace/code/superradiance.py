"""
Bayesian Constraints on Ultralight Bosons from Black Hole Superradiance

This module implements the physics of black hole superradiance for ultralight bosons (ULBs)
and a Bayesian statistical framework to derive constraints on ULB masses and self-interaction
couplings from black hole mass-spin posterior distributions.

Based on: Arvanitaki & Dubovsky (2011), "Exploring the String Axiverse with Precision
Black Hole Physics", arXiv:1004.3558

Author: Analysis Code
"""

import numpy as np
from scipy import special
import warnings

# Physical constants in CGS / natural units
G_N = 6.674e-8          # cm^3 g^-1 s^-2
c = 2.998e10             # cm/s
hbar = 1.055e-27         # erg s
M_sun = 1.989e33         # g
eV_to_g = 1.783e-33      # g per eV/c^2
eV_to_inv_cm = 1.0 / (hbar * c / eV_to_g)  # 1/cm per eV (using hbar*c)
M_Pl = 1.221e19 * 1e9    # Planck mass in eV (1.221e28 eV)
yr_to_s = 3.156e7         # seconds per year
age_universe = 1.38e10 * yr_to_s  # age of universe in seconds

# Gravitational radius: r_g = G_N * M / c^2
def r_g(M_bh_solar):
    """Gravitational radius in cm."""
    return G_N * M_bh_solar * M_sun / c**2

def r_g_natural(M_bh_solar):
    """Gravitational radius in natural units (1/eV), i.e. r_g * c / hbar mapped to 1/eV.
    Actually we work with the dimensionless product alpha = mu * r_g."""
    return G_N * M_bh_solar * M_sun / (hbar * c)


def alpha(mu_eV, M_bh_solar):
    """
    Dimensionless gravitational coupling: alpha = mu * r_g
    where mu is the boson mass and r_g = G_N * M / (hbar * c) in natural units.

    Parameters
    ----------
    mu_eV : float or array
        Boson mass in eV
    M_bh_solar : float or array
        Black hole mass in solar masses

    Returns
    -------
    alpha : float or array
        Dimensionless gravitational coupling
    """
    # alpha = mu * G_N * M / (hbar * c)
    # = (mu_eV * eV_to_g) * G_N * (M_bh_solar * M_sun) / (hbar * c)
    # Let's compute numerically
    # G_N * M_sun / (hbar * c) = 6.674e-8 * 1.989e33 / (1.055e-27 * 2.998e10)
    # = 1.327e26 / 3.163e-17 = 4.196e42 per solar mass, in units of 1/g
    # Then alpha = mu_eV * eV_to_g * above * M_bh_solar
    # = mu_eV * 1.783e-33 * 4.196e42 * M_bh_solar
    # = mu_eV * 7.482e9 * M_bh_solar
    #
    # More carefully: alpha = mu * r_g where r_g = G M / c^2 and mu = mass/hbar
    # So alpha = (mu_eV / (hbar*c)) * (G M / c^2)  ... wait
    # mu in 1/length = mu_eV * eV / (hbar * c)
    # r_g = G M / c^2 in length
    # alpha = mu_eV * eV_to_erg / (hbar * c) * G * M / c^2
    #       = mu_eV * 1.602e-12 / (1.055e-27 * 2.998e10) * 6.674e-8 * M_solar * M_bh / (2.998e10)^2

    eV_to_erg = 1.602e-12
    factor = eV_to_erg / (hbar * c) * G_N / c**2 * M_sun
    # = 1.602e-12 / (3.163e-17) * 6.674e-8 / (8.988e20) * 1.989e33
    # = 5.065e4 * 7.426e-29 * 1.989e33
    # = 5.065e4 * 1.477e5
    # = 7.48e9
    return mu_eV * factor * M_bh_solar


def omega_plus(a_star):
    """
    Angular velocity of the BH horizon: w+ = a* / (2 r_g (1 + sqrt(1 - a*^2)))
    In units of 1/r_g.

    Parameters
    ----------
    a_star : float or array
        Dimensionless spin parameter a/r_g (0 to 1)
    """
    return a_star / (2.0 * (1.0 + np.sqrt(1.0 - a_star**2)))


def superradiance_condition(alpha_val, l, m, a_star):
    """
    Check if the superradiance condition is satisfied:
    omega < m * w+

    For the hydrogen-like spectrum: omega ≈ mu * (1 - alpha^2 / (2*n_bar^2))
    where n_bar = n + l + 1 is the principal quantum number.

    The condition simplifies to: alpha < m * a* / (1 + sqrt(1 - a*^2))
    at leading order (omega ≈ mu).
    """
    w_plus = omega_plus(a_star)
    alpha_crit = m * w_plus * 2.0  # since w+ is in units where r_g=1, and alpha = mu*r_g
    # More precisely: superradiance when alpha < m * w+ (with w+ = a*/(2*(1+sqrt(1-a*^2))))
    # So alpha_crit = m * a* / (2*(1+sqrt(1-a*^2)))
    # But wait, the condition is omega < m * w+, and omega ≈ mu, and alpha = mu*r_g
    # w+ = a/(2*r_g*(1+sqrt(1-(a/r_g)^2)))
    # So omega < m * w+ => mu < m * a*/(2*r_g*(1+sqrt(1-a*^2)))
    # => mu*r_g < m*a*/(2*(1+sqrt(1-a*^2)))
    # => alpha < m*a*/(2*(1+sqrt(1-a*^2)))
    alpha_crit = m * a_star / (2.0 * (1.0 + np.sqrt(np.clip(1.0 - a_star**2, 0, None))))
    return alpha_val < alpha_crit


def superradiance_rate_nlm(alpha_val, a_star, l, m, n=0):
    """
    Superradiance rate from the non-relativistic approximation, Eq. (18) of Arvanitaki & Dubovsky.

    Gamma_{lmn} = 2 * mu * alpha^{4l+4} * r+ * (m*w+ - mu) * C_{lmn}

    Returns rate in units of 1/r_g (multiply by c/r_g to get 1/s).

    Parameters
    ----------
    alpha_val : float
        Dimensionless gravitational coupling
    a_star : float
        BH spin parameter
    l, m, n : int
        Quantum numbers
    """
    if alpha_val <= 0 or a_star <= 0:
        return 0.0

    # r+ / r_g = 1 + sqrt(1 - a*^2)
    r_plus_over_rg = 1.0 + np.sqrt(max(1.0 - a_star**2, 0.0))

    # w+ in units of 1/r_g
    w_plus = a_star / (2.0 * r_plus_over_rg)

    # mu in units of 1/r_g is alpha/r_g, but since we measure w+ in 1/r_g units:
    # mu*r_g = alpha, so mu = alpha/r_g
    # m*w+ - mu = (m*w_plus - alpha) / r_g   ... all in 1/r_g units
    # Actually: m*w+ has units of angular velocity, mu has units of 1/time
    # In units where r_g = 1: mu = alpha, w+ = a*/(2*r+)
    # Superradiance factor: (m*w+ - omega) where omega ≈ mu = alpha

    sr_factor = m * w_plus - alpha_val  # must be > 0 for superradiance
    if sr_factor <= 0:
        return 0.0

    # n_bar = n + l + 1 (principal quantum number)
    n_bar = n + l + 1

    # C_{lmn} from Eq. (18)
    # C = 2^{4l+2} * (2l+n+1)! / ((l+n+1)^{2l+4} * n!) * (l! / ((2l)! * (2l+1)!))^2
    #     * prod_{j=1}^{l} (j^2 * (1 - a*^2) + 4*r+^2*(m*w+ - mu)^2)

    try:
        # Factorial terms
        from math import factorial, lgamma

        log_C = (4*l + 2) * np.log(2)
        log_C += lgamma(2*l + n + 2)  # (2l+n+1)!
        log_C -= (2*l + 4) * np.log(l + n + 1)  # 1/(l+n+1)^{2l+4}
        log_C -= lgamma(n + 1)  # 1/n!
        log_C += 2 * lgamma(l + 1)  # (l!)^2
        log_C -= 2 * lgamma(2*l + 1)  # 1/((2l)!)^2
        log_C -= 2 * lgamma(2*l + 2)  # 1/((2l+1)!)^2

        # Product term
        log_prod = 0.0
        for j in range(1, l + 1):
            term = j**2 * (1.0 - a_star**2) + 4.0 * r_plus_over_rg**2 * sr_factor**2
            log_prod += np.log(term)
        log_C += log_prod

        C_lmn = np.exp(log_C)
    except (OverflowError, ValueError):
        return 0.0

    # Gamma = 2 * alpha * alpha^{4l+4} * r+/r_g * sr_factor * C
    # In units of 1/r_g: Gamma * r_g
    # The formula: Gamma_{lmn} = 2 * mu * alpha^{4l+4} * r+ * (m*w+ - mu) * C
    # With mu = alpha (in r_g=1 units), r+ = r_plus_over_rg
    rate = 2.0 * alpha_val * alpha_val**(4*l + 4) * r_plus_over_rg * sr_factor * C_lmn

    return rate


def superradiance_timescale(alpha_val, a_star, M_bh_solar, l=1, m=1, n=0):
    """
    Superradiance instability timescale in seconds.

    tau_sr = 1 / Gamma_sr  where Gamma_sr is the rate in physical units (1/s).

    Gamma_physical = Gamma_rate_in_rg_units * c / r_g(M)
    """
    rate = superradiance_rate_nlm(alpha_val, a_star, l, m, n)
    if rate <= 0:
        return np.inf

    rg = r_g(M_bh_solar)  # in cm
    gamma_phys = rate * c / rg  # 1/s
    return 1.0 / gamma_phys


def critical_spin_for_superradiance(alpha_val, m=1):
    """
    Find the critical spin a* at which superradiance shuts off for a given alpha and m.
    This is the Regge trajectory: alpha = m * a* / (2*(1 + sqrt(1-a*^2)))

    Solving for a*: Let x = a*. Then alpha = m*x/(2*(1+sqrt(1-x^2)))
    => 2*alpha*(1+sqrt(1-x^2)) = m*x
    => 2*alpha + 2*alpha*sqrt(1-x^2) = m*x
    => 2*alpha*sqrt(1-x^2) = m*x - 2*alpha
    => 4*alpha^2*(1-x^2) = (m*x - 2*alpha)^2
    => 4*alpha^2 - 4*alpha^2*x^2 = m^2*x^2 - 4*m*alpha*x + 4*alpha^2
    => -4*alpha^2*x^2 = m^2*x^2 - 4*m*alpha*x
    => 0 = (m^2 + 4*alpha^2)*x^2 - 4*m*alpha*x
    => x = 4*m*alpha / (m^2 + 4*alpha^2)
    """
    a_crit = 4.0 * m * alpha_val / (m**2 + 4.0 * alpha_val**2)
    return np.clip(a_crit, 0, 1.0)


def exclusion_probability_single_sample(M_bh, a_star, mu_eV,
                                         max_age_sec=None,
                                         l_max=5):
    """
    For a single (M, a*) sample, compute the probability that the BH should have been
    spun down by a boson of mass mu_eV. Returns 1 if the BH spin is inconsistent
    with the existence of this boson, 0 otherwise.

    Logic: If superradiance would have spun down the BH (timescale < BH age),
    then the observed high spin excludes this boson mass.

    We check: for each l=m mode from 1 to l_max:
    1. Is the superradiance condition satisfied? (alpha < m*w+)
    2. Is the timescale shorter than the BH age?
    If both, the BH should be on or below the Regge trajectory for that mode.

    Parameters
    ----------
    M_bh : float
        BH mass in solar masses
    a_star : float
        BH dimensionless spin
    mu_eV : float
        Boson mass in eV
    max_age_sec : float or None
        Maximum BH age in seconds (default: age of universe)
    l_max : int
        Maximum l=m mode to check
    """
    if max_age_sec is None:
        max_age_sec = age_universe

    alpha_val = alpha(mu_eV, M_bh)

    # For each l=m mode, check if superradiance would have been effective
    for l in range(1, l_max + 1):
        m = l  # fastest mode has l=m

        if not superradiance_condition(alpha_val, l, m, a_star):
            continue

        # Compute the Regge trajectory spin for this mode
        a_regge = critical_spin_for_superradiance(alpha_val, m=m)

        # If observed spin > Regge spin, then BH should have been spun down
        if a_star <= a_regge:
            continue

        # Check if timescale is short enough
        tau = superradiance_timescale(alpha_val, a_star, M_bh, l=l, m=m, n=0)

        if tau < max_age_sec:
            return 1.0  # This boson mass is excluded by this sample

    return 0.0  # This boson mass is NOT excluded by this sample


def bayesian_exclusion(M_samples, a_samples, mu_eV, max_age_sec=None, l_max=5):
    """
    Compute the Bayesian exclusion probability for a boson of mass mu_eV,
    marginalizing over the BH mass-spin posterior.

    P(excluded | mu) = (1/N) * sum_i P(excluded | M_i, a_i, mu)

    This naturally incorporates the full posterior uncertainty.

    Parameters
    ----------
    M_samples : array
        Posterior samples of BH mass (solar masses)
    a_samples : array
        Posterior samples of BH spin
    mu_eV : float
        Boson mass in eV

    Returns
    -------
    p_exclude : float
        Posterior probability of exclusion (0 to 1)
    """
    N = len(M_samples)
    excluded = 0
    for i in range(N):
        excluded += exclusion_probability_single_sample(
            M_samples[i], a_samples[i], mu_eV, max_age_sec, l_max
        )
    return excluded / N


def compute_exclusion_curve(M_samples, a_samples, mu_grid,
                             max_age_sec=None, l_max=5):
    """
    Compute exclusion probability as a function of boson mass.

    Returns array of exclusion probabilities for each mu in mu_grid.
    """
    p_exclude = np.zeros(len(mu_grid))
    for j, mu in enumerate(mu_grid):
        p_exclude[j] = bayesian_exclusion(M_samples, a_samples, mu, max_age_sec, l_max)
    return p_exclude


def bosenova_critical_mass_fraction(alpha_val, l, fa_eV):
    """
    Critical cloud mass fraction for Bosenova, Eq. (48):
    M_a / M_BH > 2 * l^4 / alpha^2 * (f_a / M_Pl)^2

    Parameters
    ----------
    alpha_val : float
        Gravitational coupling
    l : int
        Orbital quantum number
    fa_eV : float
        Axion decay constant in eV

    Returns
    -------
    critical_fraction : float
    """
    return 2.0 * l**4 / alpha_val**2 * (fa_eV / M_Pl)**2


def self_interaction_constraint(M_samples, a_samples, mu_eV, fa_eV,
                                  max_age_sec=None, l_max=5):
    """
    Check if self-interactions (parameterized by fa) can prevent superradiant spin-down.

    When the Bosenova critical mass is very small (small fa), the cloud collapses
    before extracting significant spin. This weakens the exclusion.

    The level mixing condition Eq. (53) gives:
    M_a/M_BH < (Gamma_1/Gamma_2)^{1/2} * 2*l^4/alpha^2 * (fa/MPl)^2

    If the spin extraction (Delta_a*) needed to reach the Regge trajectory
    requires more mass than the Bosenova limit allows, the exclusion is weakened.

    Returns exclusion probability marginalized over posteriors.
    """
    if max_age_sec is None:
        max_age_sec = age_universe

    N = len(M_samples)
    excluded = 0

    for i in range(N):
        M_bh = M_samples[i]
        a_star = a_samples[i]
        alpha_val = alpha(mu_eV, M_bh)

        sample_excluded = False
        for l in range(1, l_max + 1):
            m = l
            if not superradiance_condition(alpha_val, l, m, a_star):
                continue

            a_regge = critical_spin_for_superradiance(alpha_val, m=m)
            if a_star <= a_regge:
                continue

            tau = superradiance_timescale(alpha_val, a_star, M_bh, l=l, m=m, n=0)
            if tau >= max_age_sec:
                continue

            # Check if self-interactions prevent full spin-down
            # The BH needs to lose Delta_J ~ M_BH * r_g * (a* - a_regge)
            # This requires cloud mass ~ Delta_J / (m * r_cloud)
            # ~ M_BH * (a* - a_regge) * alpha / l^2
            needed_fraction = (a_star - a_regge) * alpha_val / (l**2) * 0.1

            bn_limit = bosenova_critical_mass_fraction(alpha_val, l, fa_eV)

            # If bosenova limit allows enough mass extraction, still excluded
            # Even with bosenova, multiple cycles can occur (tens to hundreds)
            # So we allow ~ 100 bosenova cycles
            effective_limit = 100.0 * bn_limit

            if effective_limit > needed_fraction:
                sample_excluded = True
                break

        if sample_excluded:
            excluded += 1

    return excluded / N


# ---- I/O utilities ----

def load_samples(filepath):
    """Load posterior samples from data file."""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]  # M, a*
