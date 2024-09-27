## Here I want to find the correlation function of the lightcone at redshifts 4, 5, 6, 7, 8, 9, 10.
## Lightcone used is 'bh_coordinates_lightcone_v3.npy' in "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import numpy.ma as ma
from Corrfunc.theory.DD import DD
from sklearn.neighbors import KernelDensity
import sys

# Unique id for the input and output files
unique_id = sys.argv[1]

def lightcone_rounder(x):
    return int(np.round(x + np.random.rand() - 0.5))
# vectorize the function
lightcone_rounder = np.vectorize(lightcone_rounder)

# define the Data directory
DATA_DIRECTORY = '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/'
# define the Plot directory
PLOT_DIRECTORY = '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots2.0/'

# define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h = 0.7

#----------------------------------------------------------------------------------------------------------------
# This whole section is done in a similar way to when I was creating the lightcone.

# Define the number of pixels in each dimension
num_pixels_x = 50
num_pixels_y = 50
num_pixels_z = 50

# Calculate the pixel size in each dimension in Mpch^-1
x_range = (0, 100) #(np.min(x_coordinates), np.max(x_coordinates))
y_range = (0, 100) #(np.min(y_coordinates), np.max(y_coordinates))
z_range = (0, 100) #(np.min(z_coordinates), np.max(z_coordinates))

pixel_size_x = (x_range[1] - x_range[0]) / num_pixels_x
pixel_size_y = (y_range[1] - y_range[0]) / num_pixels_y
pixel_size_z = pixel_size_x

# Define the redshifts.
redshifts = np.array([4, 5, 6, 7, 8, 9, 10])

# Calculate the comoving distances
new_z_axis = cosmo.comoving_distance(redshifts).value * h  # Convert to h^-1 Mpc

# Define the redshifts for each slice of the lightcone.
new_z_axis = np.arange(new_z_axis[0], new_z_axis[-1], pixel_size_z)

# Calculate the number of slices
new_num_slices = len(new_z_axis)

# Define the pixel centers
pixel_centers_x = np.linspace(x_range[0] + pixel_size_x / 2, x_range[1] - pixel_size_x / 2, num_pixels_x)
pixel_centers_y = np.linspace(y_range[0] + pixel_size_y / 2, y_range[1] - pixel_size_y / 2, num_pixels_y)
pixel_centers_z = np.copy(new_z_axis)

#--------------------------------------------------------------------------------------------------------------

# Import the lightcone data
Lightcone = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/{}_Lightcone_lumcut1e91e12.npy'.format(unique_id))

# Making the catalog of the lightcone
# Define the number of black holes in each pixel
n_bh = np.copy(lightcone_rounder(Lightcone))

# Initialize an empty list to store the black hole coordinates
bh_coordinates = []

# Iterate over the pixels and distribute the black holes
for k in range(new_num_slices):
    for i in range(num_pixels_x):
        for j in range(num_pixels_y):
            # Get the number of black holes in the current pixel
            n = n_bh[i, j, k]
            
            # Generate random x, y, z coordinates for the black holes in the pixel
            x_coords = np.random.uniform(low=pixel_centers_x[i] - pixel_size_x / 2, high=pixel_centers_x[i] + pixel_size_x / 2, size=n)
            y_coords = np.random.uniform(low=pixel_centers_y[j] - pixel_size_y / 2, high=pixel_centers_y[j] + pixel_size_y / 2, size=n)
            z_coords = np.random.uniform(low=pixel_centers_z[k] - pixel_size_z / 2, high=pixel_centers_z[k] + pixel_size_z / 2, size=n)
            
            # Append the coordinates to the list
            bh_coordinates.extend(list(zip(x_coords, y_coords, z_coords)))

# Convert the list of coordinates to a numpy array
bh_coordinates = np.array(bh_coordinates)

# Extract the black hole coordinates
x_coordinates = bh_coordinates[:, 0]
y_coordinates = bh_coordinates[:, 1]
z_coordinates = bh_coordinates[:, 2]

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

    mult = 50 ## Number of random points used as a multiple of number of data points

    n_D = len(x_coordinates_sub)
    n_rand = mult * n_D
    rand_x = np.random.uniform(min_x, max_x, n_rand)
    rand_y = np.random.uniform(min_y, max_y, n_rand)

    #-------------------------------------------------------------------------------------------
    #Drawing random numbers for z from smoothed distribution of the data z vals

    z_bin_size = 0.1
    z_bin = np.arange(np.min(z_coordinates_sub), np.max(z_coordinates_sub), z_bin_size)[:,np.newaxis]
    z_bin_mid = (z_bin + z_bin_size/2)[:-1]
    kde = KernelDensity(kernel="gaussian", bandwidth=2).fit(z_coordinates_sub[:,np.newaxis])
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
    df.to_csv(DATA_DIRECTORY + '{}_MBII_lc1e91e12_corrfunc_z{}.csv'.format(unique_id, redshifts[i]), index=False)

    corrfunc_data.append(df)
    print("Correlation function at redshift {} done.".format(redshifts[i]))
    print("Number of data points: {}".format(n_D))

    # Plot the z_coordinates_sub histogram along with the rand_z histogram both normalized.
    plt.figure()
    plt.hist(z_coordinates_sub, bins=50, density=True, alpha=0.5, label='Data')
    plt.hist(rand_z, bins=50, density=True, alpha=0.5, label='Random')
    plt.xlabel('z')
    plt.ylabel('Normalized counts')
    plt.legend(title= 'z = {}'.format(redshifts[i]))
    plt.grid(visible=False)
    plt.gca().set_box_aspect(1)
    plt.savefig(PLOT_DIRECTORY + '{}_MBII_lc_1e91e12_z{}_hist.pdf'.format(unique_id, redshifts[i]))

# Plot the correlation function
plt.style.use('MNRAS_Style.mplstyle')
for i in range(len(redshifts)):
    corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max'])/2
    plt.errorbar(corrfunc_data[i]['r mid'], corrfunc_data[i]['Landy Szalay'], yerr=corrfunc_data[i]['Pois Error'], fmt='.-', markersize=5, capsize=5, label='z = {}'.format(redshifts[i]))

plt.ylabel(r'$\xi(r)$')
plt.xlabel(r'r ($h^{-1}$ Mpc)')
# plt.ylim(-0.5, 10)
# plt.xlim(0, 150)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xscale('log')
plt.yscale('log')
plt.grid(visible=False)
plt.gca().set_box_aspect(1)

# Save this plot as a pdf to Plot directory
plt.savefig(PLOT_DIRECTORY + '{}_MBII_lc1e91e12_corrfunc.pdf'.format(unique_id))