# plot the correlation function from MBII_corrfunc_z{}.csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('MNRAS_Style.mplstyle')

# Define the path to data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Define the path to Plot directory
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/"

# Define the redshifts
redshifts = np.array([4, 5, 6, 7, 8, 9, 10])

# Load the correlation function data
corrfunc_data = []
for i in redshifts:
    corrfunc_data.append(pd.read_csv(DATA_DIRECTORY + 'MBII_corrfunc_z{}.csv'.format(i)))

# Find the midpoint of each bin, as r_mid = (r_min + r_max) / 2
for i in range(len(redshifts)):
    corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max']) / 2

# Load the Khadai et al. 2015 Fig 25 Lum>10^9 Lsun data
temp_DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/"
temp_rmid = corrfunc_data[0]['r mid']

# Now also plot the correlation function from khandai et al. 2015 as power laws
MBII_fig25_filename = "Khandai_fig25_data.csv"
MBII_fig25 = np.loadtxt(temp_DATA_DIRECTORY + MBII_fig25_filename, skiprows=1, delimiter=',')

MBII_fig25_z = MBII_fig25[:, 0]
MBII_fig25_r0 = MBII_fig25[:, 1]

MBII_gamma = 2.0

for i, redshift in enumerate(redshifts):

    plt.figure(figsize=(6, 6))

    # Plot all the correlation functions
    plt.errorbar(corrfunc_data[i]['r mid'][:-3], corrfunc_data[i]['Landy Szalay'][:-3], yerr=corrfunc_data[i]['Pois Error'][:-3], label='MassiveBlack-II (This work)', fmt='.', markersize=5, capsize=5)

    plt.ylabel(r'$\xi(r)$')
    plt.xlabel(r'r ($h^{-1}$ cMpc)')
    plt.legend()
    # plt.ylim(-0.5, 10)
    # plt.xlim(0, 150)
    plt.xscale('log')
    plt.yscale('log')

    # Now also plot the correlation function from khandai et al. 2015 as power laws. redshift 4, 5, 6 are at index 5, 6, 7
    print(i)
    if redshift in [4, 5, 6]:
        plt.plot(temp_rmid, (temp_rmid/MBII_fig25_r0[i+5])**(-MBII_gamma), label='MassiveBlack-II (Khandai et al. 2015)')

    plt.legend(title='z = {}'.format(redshift))
    plt.grid(visible=False)
    # set the box to be square
    plt.gca().set_box_aspect(1)

    # Adjust the padding for x-axis tick labels
    plt.tick_params(axis='x', pad=10)

    # Save the figure as a pdf
    plt.savefig(PLOT_DIRECTORY + 'MBII_corrfunc_z{}.pdf'.format(redshift), bbox_inches='tight')

    plt.show()