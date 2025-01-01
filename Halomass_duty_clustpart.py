import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from colossus.cosmology import cosmology
from astropy.io import ascii
from scipy.integrate import quad
from scipy.optimize import brentq
import math
import time
from colossus.lss import bias
from Halomass_duty import halo_life_time
import pandas as pd

# halo_mass_function.py copied from the link: https://github.com/sbird/DLA_script
from halo_mass_function import HaloMassFunction

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/"

# Loading the number density of quasars from Shen et al. 2007
shen_file_name = DATA_DIRECTORY + "shen_quasar_sample_datafile1.txt"
shen_data = ascii.read(shen_file_name)
sub_flag = shen_data.columns[8]
good_flag = shen_data.columns[9]
pos = np.where(sub_flag == 1)
pos_good = np.where(good_flag == 1)
red_col_shen = shen_data.columns[5][pos].value

red_col_shen_good = shen_data.columns[5][pos_good].value

bins = np.arange(2.85, 5.45, 0.1)
bin_mid = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
hist, edges = np.histogram(red_col_shen, bins=bins, density=True)
hist_good, edges_good = np.histogram(red_col_shen_good, bins=bins)
hist = hist/np.max(hist)
hist_good = hist_good/np.max(hist_good)

# Define the cosmology of Richards et al. 2006
cosmo_richards = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# h = cosmo_richards.h

cosmo_shen = cosmology.setCosmology('WMAP3')

def halo_bias(M, z):
    return bias.haloBias(M, model = 'jing98', z = z, mdef = 'vir')

def effective_bias(M_min, z):
    M_max = 1e16 * cosmo_shen.h
    halo_mass_func = HaloMassFunction(redshift=z, omega_m=0.26, omega_l=0.74, hubble=0.71)

    # Logarithmic integrand
    log_integrand_num = lambda x: halo_bias(np.exp(x), z) * halo_mass_func.dndm(np.exp(x)) / halo_life_time(np.exp(x), z) * np.exp(x)

    log_integrand_den = lambda x: halo_mass_func.dndm(np.exp(x)) / halo_life_time(np.exp(x), z) * np.exp(x)

    # Integration bounds in log-space
    log_M_min, log_M_max = np.log(M_min), np.log(M_max)
    """Calculate the effective bias as a function of M_min and redshift z."""
    # Numerator: Integrate halo_bias * halo_mass_function over M from M_min to infinity
    numerator = quad(log_integrand_num, log_M_min, log_M_max)[0]

    # Denominator: Integrate halo_mass_function over M from M_min to infinity
    denominator = quad(log_integrand_den, log_M_min, log_M_max)[0]

    return numerator / denominator

def xi_mm(r, z):
    return cosmo_shen.correlationFunction(r, z)

def diff_com_vol(z):
    # in units of Mpc^3/h^2
    c = 2.99792458e5 # speed of light in km/s
    # angular diameter distance in Mpc/h
    # Hz in km/s/Mpc
    return c * (1 + z)**2 * cosmo_shen.angularDiameterDistance(z)**2 / cosmo_shen.Hz(z)

def xi_model(r, z, M_min):
    return effective_bias(M_min, z)**2 * xi_mm(r, z)

def n_qso(z):
    
    return hist_good[np.abs(bin_mid - z).argmin()] # This is wrong. Need to fix this.

# Integrate the xi_model over volume between redshift range 2.9 to 3.5
def xi_bar(r, M_min):
    z_values = np.arange(2.9, 3.5, 0.1)
    xi_values = np.zeros(len(z_values))
    for i, z in enumerate(z_values):
        xi_values[i] = xi_model(r, z, M_min)
        diff_com_vol_vals = diff_com_vol(z)
        n_qso_vals = n_qso(z)
    numerator =  np.trapz(xi_values * diff_com_vol_vals * n_qso_vals**2, z_values)
    denominator = np.trapz(diff_com_vol_vals * n_qso_vals**2, z_values)
    return numerator / denominator

# Plot the xi_model as a function of r
plt.figure(figsize=(8, 6))
r_bins = np.logspace(start=np.log10(1.9868), stop=np.log10(314.915), num=23)
r_values = (r_bins[:-1] + r_bins[1:])/2
z = 4
M_min = 1e12
xi_values = np.zeros(len(r_values))
for i, r in enumerate(r_values):
    xi_values[i] = xi_model(r, z, M_min)
plt.plot(r_values, xi_values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Comoving Separation $r$ [Mpc/h]')
plt.ylabel('Model correlation function $\\xi_\mathrm{model}$')
plt.title('Model Correlation Function as a function of Comoving Separation')
plt.grid(False)
plt.gca().set_box_aspect(1)
plt.ylim(0.5e-3, 1e2)
plt.show()

# # Plot the differential comoving volume as a function of redshift
# z_values = np.linspace(1, 5.4, 100)
# diff_com_vols = np.zeros(len(z_values))
# for i, z in enumerate(z_values):
#     diff_com_vols[i] = diff_com_vol(z)

# plt.figure(figsize=(8, 6))
# plt.plot(z_values, diff_com_vols)
# plt.xlabel('Redshift $z$')
# plt.ylabel('Differential Comoving Volume $dV/dz$')
# plt.title('Differential Comoving Volume as a function of Redshift')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.show()

# This histogram is the number density of quasars as a function of redshift
# n_qso = hist_good


# plt.stairs(hist, edges, label="all fields")
# plt.stairs(hist_good, edges, label="good field only")
# plt.title("Redshift distribution of Shen quasars")
# plt.legend()
# plt.show()




# # Load the correlation function data from the data2.0 folder
#    # Load the coeval correlation function data
# redshifts = [4]
# toyn = 'toy1'
# unique_ids = ['iter1', 'iter2', 'iter3', 'iter4', 'iter5', 'iter6', 'iter7', 'iter8', 
#  'iter9', 'iter10', 'iter11', 'iter12', 'iter13', 'iter14', 'iter15', 'iter16']

# corrfunc_data = []
# for i in redshifts:
#     corrfunc_data.append(pd.read_csv(DATA_DIRECTORY + 'MBII_1e91e12_corrfunc_z{}.csv'.format(i)))
# for i in range(len(redshifts)):
#     corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max']) / 2

# lc_corrfunc_data = []
# inc_lc_corrfunc_data = []
# for j, unique_id in enumerate(unique_ids):
#     lc_corrfunc_data.append([])
#     inc_lc_corrfunc_data.append([])
#     # Load the correlation function data
#     for i in redshifts:
#         lc_corrfunc_data[j].append(pd.read_csv(DATA_DIRECTORY + '{}_MBII_lc1e91e12_corrfunc_z{}.csv'.format(unique_id, i)))
#         inc_lc_corrfunc_data[j].append(pd.read_csv(DATA_DIRECTORY + '{}_MBII_{}inc_lc_corrfunc_z{}.csv'.format(unique_id, toyn, i)))

# # Find the midpoint of each bin, as r_mid = (r_min + r_max) / 2
# for i in range(len(redshifts)):
#     for j in range(len(lc_corrfunc_data)):
#         lc_corrfunc_data[j][i]['r mid'] = (lc_corrfunc_data[j][i]['r min'] + lc_corrfunc_data[j][i]['r max']) / 2
#         inc_lc_corrfunc_data[j][i]['r mid'] = (inc_lc_corrfunc_data[j][i]['r min'] + inc_lc_corrfunc_data[j][i]['r max']) / 2

# # Now plot the correlation function for all the iterations
# for i, redshift in enumerate(redshifts):
#     fig, (ax, err_ax) = plt.subplots(2,1, figsize=(6, 8), height_ratios=[3, 1], sharex=True)

#     avg_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
#     avg_inc_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
#     err_avg_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
#     err_avg_inc_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
#     for j in range(len(lc_corrfunc_data)):
#         avg_lc_corrfunc_data += lc_corrfunc_data[j][i]['Landy Szalay'][:-3]
#         avg_inc_lc_corrfunc_data += inc_lc_corrfunc_data[j][i]['Landy Szalay'][:-3]
#         err_avg_lc_corrfunc_data += np.power(lc_corrfunc_data[j][i]['Pois Error'][:-3], 2)
#         err_avg_inc_lc_corrfunc_data += np.power(inc_lc_corrfunc_data[j][i]['Pois Error'][:-3], 2)

#     avg_lc_corrfunc_data = avg_lc_corrfunc_data / len(lc_corrfunc_data)
#     avg_inc_lc_corrfunc_data = avg_inc_lc_corrfunc_data / len(lc_corrfunc_data)
#     err_avg_lc_corrfunc_data = np.sqrt(err_avg_lc_corrfunc_data / len(lc_corrfunc_data))
#     err_avg_inc_lc_corrfunc_data = np.sqrt(err_avg_inc_lc_corrfunc_data / len(lc_corrfunc_data))

#     # Plot the correlation functions of the coeval and lightcone data
#     plt.plot(lc_corrfunc_data[0][i]['r mid'][:-3], avg_lc_corrfunc_data, label='lc')
#     plt.plot(inc_lc_corrfunc_data[0][i]['r mid'][:-3], avg_inc_lc_corrfunc_data, label='inc lc')
#     plt.plot(corrfunc_data[i]['r mid'], corrfunc_data[i]['Landy Szalay'], label='coeval')

# Plot the correlation function as a function of r
# plt.figure(figsize=(8, 6))
# r_values = np.logspace(-1, 2, 100)
# z_list = [2.9, 3.5, 4.0, 4.5, 5.4]
# xi_values = np.zeros(len(r_values))
# for z in z_list:
#     for i, r in enumerate(r_values):
#         xi_values[i] = xi_mm(r, z)
#     plt.plot(r_values, xi_values, label=f"$z = {z}$")

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Comoving Separation $r$ [Mpc/h]')
# plt.ylabel('Correlation Function $\\xi_{mm}$')
# plt.title('Correlation Function as a function of Comoving Separation')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.legend()
# plt.show()


# # Plot the effective bias as a function of redshift
# M_min = [1e12, 5e12, 1e13, 5e13]
# z_values = np.linspace(2.9, 5.4, 100)

# # For each M_min value, calculate the effective bias at each redshift
# b_eff = np.zeros((len(M_min), len(z_values)))
# for i, M in enumerate(M_min):
#     for j, z in enumerate(z_values):
#         b_eff[i, j] = effective_bias(M, z)

# plt.figure(figsize=(8, 6))
# for i, M in enumerate(M_min):
#     plt.plot(z_values, b_eff[i], label=f"$M_{{\\mathrm{{min}}}} = {M:.1e}$")
# plt.xlabel('Redshift $z$')
# plt.ylabel('Effective Bias $b_{\mathrm{eff}}$')
# plt.title('Effective Bias as a function of Redshift')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.legend()
# plt.show()

# # Plot the integrant of the effective bias as a function of M
# z = 3.5
# M_values = np.logspace(10, 16, 100)
# integrant = np.zeros(len(M_values))
# for i, M in enumerate(M_values):
#     integrant[i] = halo_bias(M, z) * HaloMassFunction(redshift=z, omega_m=0.26, omega_l=0.74, hubble=0.71).dndm(M)/halo_life_time(M,z)

# plt.figure(figsize=(8, 6))
# plt.plot(M_values, integrant)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Halo Mass $M$ [$M_{\odot}/h$]')
# plt.ylabel('Integrant')
# plt.title('Integrant of the Effective Bias as a function of Halo Mass')
# plt.grid(False)
# plt.gca().set_box_aspect(1)
# plt.show()
