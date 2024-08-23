## Here I want to find the correlation function of the lightcone at redshifts 4, 5, 6, 7, 8, 9, 10.
## Lightcone used is 'bh_coordinates_lightcone_v3.npy' in "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import numpy.ma as ma
from Corrfunc.theory.DD import DD
from sklearn.neighbors import KernelDensity

# define the Data directory
DATA_DIRECTORY = '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/'
# define the Plot directory
PLOT_DIRECTORY = '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/'

# define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h = 0.7

# Import the lightcone data
Lightcone = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/bh_coordinates_lightcone_v3.npy')

# Extract the black hole coordinates
x_coordinates = Lightcone[:, 0]
y_coordinates = Lightcone[:, 1]
z_coordinates = Lightcone[:, 2]

x_coordinates_array = [0]*7
y_coordinates_array = [0]*7
z_coordinates_array = [0]*7

redshifts = np.array([4, 5, 6, 7, 8, 9, 10])
# convert redshifts to comoving distances
red_com = cosmo.comoving_distance(redshifts).value * h

# For the case of redshift 4.
# comoving distance at redshift 4
com_dist4 = red_com[0]

# define a 100 Mpc/h box extending from com_dist4 to com_dist4 + 100 Mpc/h
com_dist4_end = com_dist4 + 100

# find the indices of the black holes that are within the box
ind4 = np.where((z_coordinates >= com_dist4) & (z_coordinates < com_dist4_end))[0]

# extract the x, y, z coordinates of the black holes within the box
x_coordinates_array[0] = x_coordinates[ind4]
y_coordinates_array[0] = y_coordinates[ind4]
z_coordinates_array[0] = z_coordinates[ind4]

# For the case of 5 to 9 redshifts
# comoving distance at redshift 5 to 9
for i in range(5):
    com_dist_start = red_com[i+1]
    com_dist_end = com_dist_start + 100

    # find the indices of the black holes that are within the box
    ind = np.where((z_coordinates >= com_dist_start) & (z_coordinates < com_dist_end))[0]

    # extract the x, y, z coordinates of the black holes within the box
    x_coordinates_array[i+1] = x_coordinates[ind]
    y_coordinates_array[i+1] = y_coordinates[ind]
    z_coordinates_array[i+1] = z_coordinates[ind]

# for the case of redshift 10
# comoving distance at redshift 10
com_dist10 = red_com[6]

# define a 100 Mpc/h box extending from com_dist10 - 100 Mpc/h to com_dist10
com_dist10_start = com_dist10 - 100

# find the indices of the black holes that are within the box
ind10 = np.where((z_coordinates >= com_dist10_start) & (z_coordinates < com_dist10))[0]

# extract the x, y, z coordinates of the black holes within the box
x_coordinates_array[6] = x_coordinates[ind10]
y_coordinates_array[6] = y_coordinates[ind10]
z_coordinates_array[6] = z_coordinates[ind10]

# Find the two point correlation function of the black holes at redshifts 4, 5, 6, 7, 8, 9, 10.
# Now I want to calculate the correlation function using the selected black holes and compare it with the correlation function using all black holes.
# Full balck holes correlation function.
corrfunc_data = []

for i in range(7):
    x_coordinates_sub = x_coordinates_array[i]
    y_coordinates_sub = y_coordinates_array[i]
    z_coordinates_sub = z_coordinates_array[i]

    min_x = 0
    max_x = 100 # h^-1 Mpc. Here x_coords, y_coords, z_coords are in h^-1 Mpc unlike the black holes coordinates directly from the MBII data.
    min_y = 0
    max_y = 100

    mult = 20 ## Number of random points used as a multiple of number of data points

    n_D = len(x_coordinates_sub)
    n_rand = mult * n_D
    rand_x = np.random.uniform(min_x, max_x, n_rand)
    rand_y = np.random.uniform(min_y, max_y, n_rand)

    #-------------------------------------------------------------------------------------------
    #Drawing random numbers for z from smoothed distribution of the data z vals

    z_bin_size = 0.1
    z_bin = np.arange(np.min(z_coordinates_sub), np.max(z_coordinates_sub), z_bin_size)[:,np.newaxis]
    z_bin_mid = (z_bin + z_bin_size/2)[:-1]
    kde = KernelDensity(kernel="gaussian", bandwidth=50).fit(z_coordinates_sub[:,np.newaxis])
    log_dens = kde.score_samples(z_bin_mid)
    pdf = np.exp(log_dens)
    # ax.fill(pos_z[:, 0], pdf, fc="#AAAAFF")
    cdf = np.cumsum(pdf)
    cdf = cdf / np.max(cdf)
    cdf = np.insert(cdf, 0, 0)

    z_bin = z_bin.flatten()
    z_bin_mid = z_bin_mid.flatten()

    uni_val = np.random.rand(n_rand)
    bin_indices = np.searchsorted(cdf, uni_val)
    bin_edges = z_bin[bin_indices - 1]
    bin_diff = z_bin[bin_indices] - z_bin[bin_indices-1]
    bin_weights = (uni_val - cdf[bin_indices-1]) / (cdf[bin_indices] - cdf[bin_indices-1])
    rand_z = bin_edges + bin_weights * bin_diff

    ## ---------------------------------------------------------------------------------------------------------------------------------------------------

    bins = np.logspace(start=np.log10(1.9868), stop=np.log10(314.915), num=23) #np.arange(0.01, 100, bin_size)
    bin_mids = (bins[0:-1] + bins[1:])/2

    result_DD = DD(autocorr=1, nthreads=1, binfile=bins, X1 = x_coordinates_sub, Y1 = y_coordinates_sub, Z1 = z_coordinates_sub, periodic=False)
    result_RR = DD(autocorr=1, nthreads=1, binfile=bins, X1 = rand_x, Y1 = rand_y, Z1 = rand_z, periodic=False)
    result_DR = DD(autocorr=0, nthreads=1, binfile=bins, X1 = x_coordinates_sub, Y1 = y_coordinates_sub, Z1 = z_coordinates_sub, X2 = rand_x, Y2 = rand_y, Z2 = rand_z, periodic=False)

    DD_norm = (n_D * (n_D - 1))/2
    RR_norm = (n_rand * (n_rand - 1))/2
    DR_norm = n_D * n_rand

    DD_count = result_DD['npairs']/2
    RR_count = result_RR['npairs']/2
    DR_count = result_DR['npairs']

    nil_pos = np.where(DD_count * RR_count * DR_count == 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        LandSzal2pcf = (DD_count/DD_norm - 2 * DR_count/DR_norm + RR_count/RR_norm)/(RR_count/RR_norm)

    LandSzal2pcf = ma.array(LandSzal2pcf)
    LandSzal2pcf[nil_pos] = ma.masked

    with np.errstate(divide='ignore', invalid='ignore'):
        pois_err = (1 + LandSzal2pcf)/np.sqrt(np.minimum(DD_count, n_D))

    pois_err = ma.array(pois_err)
    pois_err[nil_pos] = ma.masked

    from pandas import DataFrame

    df = DataFrame({"r min":bins[0:-1], "r max":bins[1:], "DD count":DD_count, "DR count":DR_count, "RR count": RR_count, "Landy Szalay":LandSzal2pcf, "Pois Error":pois_err})

    #Save the data to a file
    df.to_csv(DATA_DIRECTORY + 'MBII_lc_corrfunc_z{}.csv'.format(redshifts[i]), index=False)

    corrfunc_data.append(df)

# Plot the correlation function
plt.style.use('MNRAS_Style.mplstyle')
for i in range(len(redshifts)):
    corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max'])/2
    plt.errorbar(corrfunc_data[i]['r mid'], corrfunc_data[i]['Landy Szalay'], yerr=corrfunc_data[i]['Pois Error'], fmt='.-', markersize=5, capsize=5, label='z = {}'.format(redshifts[i]))

plt.ylabel(r'$\xi(r)$')
plt.xlabel(r'r ($h^{-1}$ Mpc)')
# plt.ylim(-0.5, 10)
# plt.xlim(0, 150)
plt.legend()
plt.xscale('log')
plt.yscale('log')

# Save this plot as a pdf to Plot directory
plt.savefig(PLOT_DIRECTORY + 'MBII_lc_corrfunc.pdf')