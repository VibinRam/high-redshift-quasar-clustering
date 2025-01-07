import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from astropy.io import ascii
from scipy.integrate import quad
from scipy.optimize import brentq
import math
import time
from colossus.lss import bias

# halo_mass_function.py copied from the link: https://github.com/sbird/DLA_script
from halo_mass_function import HaloMassFunction

cosmo_shen = cosmology.setCosmology('WMAP3', params={'flat': True, 'H0': 71.0, 'Om0': 0.26, 'Ob0': 0.0435, 'sigma8': 0.751, 'ns': 0.938})
cosmo_richards = cosmology.setCosmology('WMAP1', params={'H0': 70.0, 'Om0': 0.3})

# Define the plot style
plt.style.use('MNRAS_Style.mplstyle')

DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/"
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots2.0/"

# Find the integrated qlf from the Richards et al. 2006 QLF integrated over the magnitude ranges of Shen et al. 2007.

# Load the K-correction file of Richards et al. 2006 Table 4 from the data2.0 folder. The file is in ASCII format.
Kcorr_data = ascii.read(DATA_DIRECTORY + 'richards2006_table4.txt')

# Parameters for the variable power law model of richards et al. 2006
# A1 = 0.83 Because there are two A1 values for z < 2.40 and z >= 2.40.
A2 = -0.11
B1 = 1.43
B2 = 36.63
B3 = 34.39
M_star = -26
z_ref = 2.45
phi_star = 10 ** (-5.70)

def phi(M, z, M_star, B1, B2, B3):
    """Calculate the phi term in the QLF equation."""
    psi = np.log10((1 + z) / (1 + z_ref))
    return M - (M_star + B1 * psi + B2 * psi**2 + B3 * psi**3)

def quasar_luminosity_function(M, z):
    """Calculate the z=2 i-band QLF for a given absolute magnitude M and redshift z."""
    if z <= 2.45:
        A = 0.84
    else:
        A = 0.83 + A2 * (z - 2.45)
    phi_value = phi(M, z, M_star, B1, B2, B3)
    return phi_star * 10 ** (A * phi_value) / cosmo_richards.h**3 # Mpc^-3 mag^-1 h^3

def radius_mass(M, z):
    """Radius within which mass M is enclosed in Mpc/h."""
    rho_c = cosmo_shen.rho_c(z) * 1e9  # Critical density in M_sun * h^2/ Mpc^3
    rho_m = cosmo_shen.Om(z) * rho_c
    R = (3 * M / (4 * np.pi * rho_m))**(1/3)
    return R

def sigma(M):
    R = radius_mass(M, 0)
    return cosmo_shen.sigma(R, 0)

def S_M(M):
    """Variance S(M) = sigma^2(M)."""
    sig = sigma(M)
    return sig**2

def w(z):
    delta_crit0 = 0.15 * (12 * np.pi) ** (2/3) * cosmo_shen.Om(0) ** 0.0055
    return delta_crit0 / cosmo_shen.growthFactor(z)

def Lac_Cole_cond_prob(z2, S2, S1, omega1):
    """
    Calculate the conditional probability P(S < S2, omega2 | S1, omega1) 
    using the given expression.

    Parameters:
    S1 (float): Variance corresponding to mass M.
    S2 (float): Variance corresponding to mass 2M.
    omega1 (float): Critical density threshold at initial redshift z.
    omega2 (float): Critical density threshold at final redshift z2.

    Returns:
    float: The conditional probability value.
    """

    omega2 = w(z2)

    # Calculate the prefactor
    prefactor = 0.5 * (omega1 - 2 * omega2) / omega1

    # Calculate the exponential term
    exp_term = np.exp((2 * omega2 * (omega1 - omega2)) / S1)

    # Calculate the first complementary error function term
    erfc_term1 = math.erfc((S2 * (omega1 - 2 * omega2) + S1 * omega2) / (np.sqrt(2 * S1 * S2 * (S1 - S2))))

    # Calculate the second complementary error function term
    erfc_term2 = math.erfc((S1 * omega2 - S2 * omega1) / (np.sqrt(2 * S1 * S2 * (S1 - S2))))

    # Combine all the terms to get the conditional probability
    probability = prefactor * exp_term * erfc_term1 + 0.5 * erfc_term2

    return probability

def halo_life_time(M, z1):
    """Calculate the halo life time for a given mass M and redshift z."""
    S = S_M(M)
    omega1 = w(z1)
    S_2M = S_M(2 * M)
    z=4

    # Solve Lac_Cole_cond_prob for probability of 0.5 to get z2
    z2 = brentq(lambda z: Lac_Cole_cond_prob(z, S_2M, S, omega1) - 0.5, 0, z1)

    # Calculate the halo life time
    t_H = cosmo_shen.age(z2) - cosmo_shen.age(z1) # in Gyr

    return t_H

# Create halo mass function object's dictionary for each redshift in the list z_values so that we can seemlessly call the dndm function
# redshift values that we consider are from 2.9 to 5.4 in steps of 0.1
z_values = np.arange(2.9, 5.3 + 0.1, 0.1)
halo_mass_func_dict = {z: HaloMassFunction(redshift=z, omega_m=0.26, omega_l=0.74, omega_b= 0.0435, ns= 0.938, sigma8= 0.751, hubble=0.71) for z in z_values}

def quasar_density_halo_model(z, t_q, M_min):
    halo_mass_func = halo_mass_func_dict[z]
    """Calculate the quasar number density with given duty cycle and minimum mass."""
    integral, _ = quad(lambda M: halo_mass_func.dndm(M)/halo_life_time(M, z), M_min, 1e15)
    return t_q * integral


# To integrate the quasar luminosity function we need the absolute magnitude limit M_i(z=2) corresponding to the limiting i-band magnitude of 20.2.
# There is some more things shen does for the bins z = 2.9 and 3.0. I will ignore that for now.

# M_i(z=2) we call as M_faint and is calculated as follows using cosmology of Richards:
m_i = 20.2
M_faint_20_2 = m_i - 5 * np.log10(1e6 * (cosmo_richards.luminosityDistance(z_values)/cosmo_richards.h) / 10) \
            - Kcorr_data['KCorr'][[np.abs(Kcorr_data['z'] - z).argmin() for z in z_values]]

# Integrating quasar luminosity function using quad function the integral limit is from -np.inf to M_faint for a given redshift
richards_int_QLF_20_2 = [quad(quasar_luminosity_function, -np.inf, M_faint_20_2[i], args=(z,))[0] for i, z in enumerate(z_values)]

# Find the ratio of comoving volume elements for the two cosmologies
# colossus angular diameter distance units Mpc/h, H(z) units Km/s/Mpc : Shen
# astopy cosmology angular diameter distance units Mpc, H(z) units km/s/Mpc : Richards
com_vol_ratio = ((cosmo_shen.angularDiameterDistance(z_values)/cosmo_shen.h)**2 / (cosmo_richards.angularDiameterDistance(z_values)/cosmo_richards.h)**2) * \
                    (cosmo_richards.Hz(z_values) / cosmo_shen.Hz(z_values)) # * (cosmo_richards.h / cosmo_shen.h)

# Multiply the integrated QLF by the comoving volume ratio to convert it to the Shen cosmology
integral_values_20_2 = richards_int_QLF_20_2 * com_vol_ratio * (cosmo_shen.h / cosmo_richards.h)**3
print(np.column_stack((z_values, richards_int_QLF_20_2, integral_values_20_2)))

# take another m_i value and calculate the integral values again for smaller range of z_values
m_i = 19.1
M_faint_19_1 = m_i - 5 * np.log10(1e6 * (cosmo_richards.luminosityDistance(z_values)/cosmo_richards.h) / 10) \
            - Kcorr_data['KCorr'][[np.abs(Kcorr_data['z'] - z).argmin() for z in z_values]]
# M_values = np.linspace(M_bright, M_faint[0], 1000)
richards_int_QLF_19_1 = [quad(quasar_luminosity_function, -np.inf, M_faint_19_1[i], args=(z,))[0] for i, z in enumerate(z_values)]

# Multiply the integrated QLF by the comoving volume ratio
integral_values_19_1 = richards_int_QLF_19_1 * com_vol_ratio

# Save the richards integrated QLF 
np.save(DATA_DIRECTORY + "richards_int_QLF_20.2.npy", integral_values_20_2)

# Run the halo model for 100 tq values to see how fast it is and report the time taken
# t_q_values = np.linspace(0.1, 1, 100)
# M_min = 5e12
# start = time.time()
# for t_q in t_q_values:
#     quasar_density_hm = [quasar_density_halo_model(z, t_q, M_min) for z in z_values]
# end = time.time()
# print("Time taken for running halo model for 100 tq values is :", end-start)

# Run the halo model for 100 M_min values to see how fast it is and report the time taken
# t_q = 0.3
# M_min_values = np.logspace(10, 15, 100)
# start = time.time()
# for M_min in M_min_values:
#     quasar_density_hm = [quasar_density_halo_model(z, t_q, M_min) for z in z_values]
# end = time.time()
# print("Time taken for running halo model for 100 M_min values is :", end-start)


# Find the quasar number density from the halo model as a function of redshift
# t_q = 1
# M_min = 1.32e13
# quasar_density_hm = [quasar_density_halo_model(z, t_q, M_min) for z in z_values]

# # print richards integrated QLF value corresponding to z=3.0
# print(richards_int_QLF_19_1[np.abs(z_values - 3.0).argmin()])
# print(richards_int_QLF_19_1[np.abs(z_values - 3.5).argmin()])
# print(richards_int_QLF_19_1[np.abs(z_values - 4.0).argmin()])

# # Plot the integrated QLF as a function of redshift
# plt.figure(figsize=(8, 6))
# plt.plot(z_values, richards_int_QLF, label="richards_int_QLF_i_lt_20.2")
# plt.plot(z_values[:10], richards_int_QLF_19_1[:10], label="richards_int_QLF_i_lt_19.1")
# plt.plot(z_values, quasar_density_hm, label="quasar_density_halo_model") 
# plt.xlabel('Redshift $z$')
# plt.ylabel(r'$\Phi(z)$ h$^3$Mpc$^{-3}$')
# # plt.title('Integrated Quasar Luminosity Function vs. Redshift')
# plt.grid(False)
# plt.yscale('log')
# plt.xlim(2.6, 5.5)
# plt.ylim(1e-9, 1e-5)
# plt.gca().set_box_aspect(1)
# plt.legend()

# # Save this plot as a pdf
# plt.savefig(PLOT_DIRECTORY + "integrated_QLF_and_Halo_model_QLF_vs_redshift.pdf")

# plt.show()

# start = time.time()

# t_q = 0.3
# M_min = 5e12
# z_values = np.arange(2.9, 5.4, 0.1)
# print([quasar_density_halo_model(z, t_q, M_min) for z in z_values])

# end = time.time()
# print("Time taken is :", end-start)

# # Plot the halo life time as a function of halo mass
# M_values = np.logspace(10, 15, 100) * cosmo_shen.h # Halo mass in M_sun/h
# t_H_values = [halo_life_time(M, 3) for M in M_values]

# # Plot the halo life time as a function of halo mass
# plt.figure(figsize=(8, 6))
# plt.plot(M_values, t_H_values)
# plt.xscale('log')
# plt.xlabel('Halo Mass $M$ [$M_{\odot}/h$]')
# plt.ylabel('Halo Life Time $t_H$ [Gyr]')
# plt.title('Halo Life Time as a function of Halo Mass')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.show()

# # Plot the quasar luminosity function as a function of absolute magnitude
# M_values = np.linspace(-100, -20, 1000)
# z_values_temp = [2.9, 3.0, 3.5, 4.0, 4.5, 5.0]
# plt.figure(figsize=(8, 6))
# for z in z_values_temp:
#     plt.plot(M_values, quasar_luminosity_function(M_values, z), label=f"z = {z}")
# plt.xlabel('Absolute Magnitude $M$')
# plt.ylabel('Quasar Luminosity Function $\Phi(M)$')
# # plt.title('Quasar Luminosity Function as a function of Absolute Magnitude')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.legend()
# plt.show()

# # Plot the quasar luminosity function integrated over the magnitude ranges M_i less than -27.6 as a function of redshift
# M_values = np.linspace(-30, -27.6, 1000)
# z_values_temp = np.linspace(0, 5.4, 1000)
# richards_int_QLF = [np.trapz(quasar_luminosity_function(M_values, z) * cosmo_richards.h**3, M_values) for z in z_values_temp]
# plt.figure(figsize=(8, 6))
# plt.plot(z_values_temp, richards_int_QLF)
# plt.xlabel('Redshift $z$')
# plt.ylabel('Integrated Quasar Luminosity Function $\Phi(z)$')
# plt.title('Integrated Quasar Luminosity Function as a function of Redshift')
# plt.grid(False)
# plt.yscale('log')
# plt.ylim(0.9e-9, 1e-7)
# plt.gca().set_box_aspect(1)
# plt.show()

# # Plot the lac cole conditional probability as a function of z2
# M = 1e12
# z1 = 3
# z2_values = np.linspace(0, 5.4, 100)
# S2 = S_M(2 * M)
# S1 = S_M(1 * M)
# omega1 = w(z1)
# omega2 = w(z2_values)
# # print(S2, S1, omega1)

# prob_values = [Lac_Cole_cond_prob(z2, S2, S1, omega1) - 0.5 for z2 in z2_values]

# plt.figure(figsize=(8, 6))
# plt.plot(z2_values, prob_values)
# plt.axhline(y=0, color='k', linestyle='--')
# plt.axvline(x=z1, color='k', linestyle='--')
# plt.xlabel('Redshift $z_2$')
# plt.ylabel('Conditional Probability $P(S < S_2, \omega_2 | S_1, \omega_1)$')
# plt.title('Conditional Probability as a function of $z_2$')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.show()

# Plot the critical density threshold as a function of redshift
# z_values = np.linspace(2.9, 5.4, 100)
# omega_values = w(z_values)
# plt.figure(figsize=(8, 6))
# plt.plot(z_values, omega_values)
# plt.xlabel('Redshift $z$')
# plt.ylabel('Critical Density Threshold $\omega(z)$')
# plt.title('Critical Density Threshold as a function of Redshift')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.show()
    

# # Plot sigma(M) as a function of M
# M_values = np.logspace(10, 15, 100) * cosmo_shen.h # Halo mass in M_sun/h
# sigma_values = sigma(M_values)
# plt.figure(figsize=(8, 6))
# plt.plot(M_values, sigma_values)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Halo Mass $M$ [$M_{\odot}/h$]')
# plt.ylabel('Mass Variance $\sigma(M)$')
# plt.title('Mass Variance as a function of Halo Mass')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.show()
