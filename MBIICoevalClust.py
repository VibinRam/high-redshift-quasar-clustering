# I want to find the two point correlation function in each of the simulation box. 

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
# Import the cosmology
from astropy.cosmology import FlatLambdaCDM
# Import the DD function from Corrfunc
from Corrfunc.theory.DD import DD
from matplotlib.patches import Rectangle
from pandas import DataFrame

plt.style.use('MNRAS_Style.mplstyle')

# Define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Define the reduced Hubble constant
h = 0.7

# Define the path to data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Save the plot as a pdf file
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/"


file_paths = ['/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_034.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_029.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_026.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_024.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_020.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_019.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_018.txt']

redshifts = np.array([4, 5, 6, 7, 8, 9, 10])

corrfunc_data = []

for i,file_path in enumerate(file_paths):
    # Load the data from the file
    data = np.loadtxt(file_path)
    
    # Extract the x, y, and z coordinates
    x_coordinates = data[:, 1]
    y_coordinates = data[:, 2]
    z_coordinates = data[:, 3]

    # Define the number of black holes to choose for the subset
    num_black_holes = 100000

    # randomly draw num_black_holes from the bh_coordinates
    if num_black_holes > x_coordinates.shape[0]:
        num_black_holes = x_coordinates.shape[0]
    random_indices = np.random.choice(x_coordinates.shape[0], num_black_holes, replace=False)

    bh_pos_x = x_coordinates[random_indices]/1000 # Convert to h^-1 Mpc
    bh_pos_y = y_coordinates[random_indices]/1000 # Convert to h^-1 Mpc
    bh_pos_z = z_coordinates[random_indices]/1000 # Convert to h^-1 Mpc
    print(len(bh_pos_x), flush=True)

    # Now I want to calculate the correlation function using the selected black holes and compare it with the correlation function using all black holes.
    # Full balck holes correlation function.

    min_x = 0
    max_x = 100 # h^-1 Mpc. Here x_coords, y_coords, z_coords are in h^-1 Mpc unlike the black holes coordinates directly from the MBII data.
    min_y = 0
    max_y = 100
    min_z = 0
    max_z = 100

    mult = 50 ## Number of random points used as a multiple of number of data points

    n_D = len(bh_pos_x)
    n_rand = mult * n_D
    rand_x = np.random.uniform(min_x, max_x, n_rand)
    rand_y = np.random.uniform(min_y, max_y, n_rand)
    rand_z = np.random.uniform(min_z, max_z, n_rand)

    bins = np.logspace(start=np.log10(1.9868), stop=np.log10(314.915), num=23) #np.arange(0.01, 100, bin_size)
    bin_mids = (bins[0:-1] + bins[1:])/2

    result_DD = DD(autocorr=1, nthreads=1, binfile=bins, X1 = bh_pos_x, Y1 = bh_pos_y, Z1 = bh_pos_z, periodic=False)
    result_RR = DD(autocorr=1, nthreads=1, binfile=bins, X1 = rand_x, Y1 = rand_y, Z1 = rand_z, periodic=False)
    result_DR = DD(autocorr=0, nthreads=1, binfile=bins, X1 = bh_pos_x, Y1 = bh_pos_y, Z1 = bh_pos_z, X2 = rand_x, Y2 = rand_y, Z2 = rand_z, periodic=False)

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

    df = DataFrame({"r min":bins[0:-1], "r max":bins[1:], "DD count":DD_count, "DR count":DR_count, "RR count": RR_count, "Landy Szalay":LandSzal2pcf, "Pois Error":pois_err})

    #Save the data to a file
    df.to_csv(DATA_DIRECTORY + 'MBII_corrfunc_z{}.csv'.format(redshifts[i]), index=False)

    corrfunc_data.append(df)

# Plot the correlation functions    

plt.style.use('MNRAS_Style.mplstyle')

# Find the midpoint of each bin, as r_mid = (r_min + r_max) / 2
for i in range(len(redshifts)):
    corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max']) / 2

# Make an array to store the plot objects
plot_objects = np.zeros(len(redshifts), dtype=object)   

# Plot all the correlation functions
for i in range(len(redshifts)):
    plot_objects[i] = plt.errorbar(corrfunc_data[i]['r mid'][:-3], corrfunc_data[i]['Landy Szalay'][:-3], yerr=corrfunc_data[i]['Pois Error'][:-3], fmt='.-', markersize=5, capsize=5)

plt.ylabel(r'$\xi(r)$')
plt.xlabel(r'r ($h^{-1}$ Mpc)')
# plt.ylim(-0.5, 10)
# plt.xlim(0, 150)
plt.xscale('log')
plt.yscale('log')

Khandai_DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/"
temp_rmid = corrfunc_data[0]['r mid']

# Now also plot the correlation function from khandai et al. 2015 as power laws
MBII_fig25_filename = "Khandai_fig25_data.csv"
MBII_fig25 = np.loadtxt(Khandai_DATA_DIRECTORY + MBII_fig25_filename, skiprows=1, delimiter=',')

MBII_fig25_z = MBII_fig25[:, 0]
MBII_fig25_r0 = MBII_fig25[:, 1]

MBII_gamma = 2.0

# Make another array to store the plot objects
plot_objects_khandai = np.zeros(len(MBII_fig25_z[5:]), dtype=object)

for i in range(len(MBII_fig25_z[5:])):
    plot_objects_khandai[i], = plt.plot(temp_rmid, (temp_rmid/MBII_fig25_r0[i+5])**(-MBII_gamma))

# Make title proxy
title_proxy = Rectangle((0,0), 0, 0, color='w')

# Make a list of the following form: [title_proxy, plot_objects_khandai[0], plot_objects_khandai[1], ..., plot_objects_khandai[n], title_proxy, plot_objects[0], plot_objects[1], ..., plot_objects[n]]
legend_list = [title_proxy]
legend_list.extend(plot_objects_khandai)
legend_list.append(title_proxy)
legend_list.extend(plot_objects)

# Make a list to hold the labels for the legend like: ['Khandai et al. 2015', 'z = 4', ... , 'z = 6', 'This work', 'z = 4', ... , 'z = 10']
legend_labels = ['Khandai et al. 2015']
legend_labels.extend(['z = {}'.format(int(np.round(MBII_fig25_z[i+5]))) for i in range(len(MBII_fig25_z[5:]))])
legend_labels.append('This work')
legend_labels.extend(['z = {}'.format(redshifts[i]) for i in range(len(redshifts))])

# give legends outside the plots
plt.legend(legend_list, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(visible=False)
plt.gca().set_box_aspect(1)

plt.savefig(PLOT_DIRECTORY + 'MBII_corrfunc.pdf')

plt.show()