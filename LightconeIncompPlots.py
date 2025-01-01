# plot the correlation function from MBII_corrfunc_z{}.csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('MNRAS_Style.mplstyle')

# Define the path to data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/"

# Define the path to Plot directory
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots2.0/"

# Define the ploting style
plt.style.use('MNRAS_Style.mplstyle')

# Define the redshifts
redshifts = np.array([4, 5, 6, 7, 8, 9, 10])

# Load the Khadai et al. 2015 Fig 25 Lum>10^9 Lsun data
temp_DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/"

# Now also plot the correlation function from khandai et al. 2015 as power laws
MBII_fig25_filename = "Khandai_fig25_data.csv"
MBII_fig25 = np.loadtxt(temp_DATA_DIRECTORY + MBII_fig25_filename, skiprows=1, delimiter=',')

MBII_fig25_z = MBII_fig25[:, 0]
MBII_fig25_r0 = MBII_fig25[:, 1]

MBII_gamma = 2.0

def coeval_plot():
    # Load the correlation function data
    corrfunc_data1 = []
    corrfunc_data2 = []
    corrfunc_data3 = []
    for i in redshifts:
        corrfunc_data1.append(pd.read_csv(DATA_DIRECTORY + 'MBII_1e91e10_corrfunc_z{}.csv'.format(i)))
        corrfunc_data2.append(pd.read_csv(DATA_DIRECTORY + 'MBII_1e101e11_corrfunc_z{}.csv'.format(i)))
        corrfunc_data3.append(pd.read_csv(DATA_DIRECTORY + 'MBII_1e111e12_corrfunc_z{}.csv'.format(i)))

    # Find the midpoint of each bin, as r_mid = (r_min + r_max) / 2
    for i in range(len(redshifts)):
        corrfunc_data1[i]['r mid'] = (corrfunc_data1[i]['r min'] + corrfunc_data1[i]['r max']) / 2
        corrfunc_data2[i]['r mid'] = (corrfunc_data2[i]['r min'] + corrfunc_data2[i]['r max']) / 2
        corrfunc_data3[i]['r mid'] = (corrfunc_data3[i]['r min'] + corrfunc_data3[i]['r max']) / 2

    # Plot all the correlation functions
    for i, redshift in enumerate(redshifts):
        fig, ax = plt.subplots(1,1, figsize=(6, 4))

        # Plot all the correlation functions
        ax.errorbar(corrfunc_data1[i]['r mid'][:-3], corrfunc_data1[i]['Landy Szalay'][:-3], yerr=corrfunc_data1[i]['Pois Error'][:-3], label=r'$10^9-10^{10}$', fmt='.', markersize=5, capsize=5)
        ax.errorbar(corrfunc_data2[i]['r mid'][:-3], corrfunc_data2[i]['Landy Szalay'][:-3], yerr=corrfunc_data2[i]['Pois Error'][:-3], label=r'$10^{10}-10^{11}$', fmt='.', markersize=5, capsize=5)
        ax.errorbar(corrfunc_data3[i]['r mid'][:-3], corrfunc_data3[i]['Landy Szalay'][:-3], yerr=corrfunc_data3[i]['Pois Error'][:-3], label=r'$10^{11}-10^{12}$', fmt='.', markersize=5, capsize=5)
        ax.set_ylabel(r'$\xi(r)$')
        ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        ax.legend()
        # plt.ylim(-0.5, 10)
        # plt.xlim(0, 150)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Now also plot the correlation function from khandai et al. 2015 as power laws. redshift 4, 5, 6 are at index 5, 6, 7
        print(redshift)
        temp_rmid = corrfunc_data1[i]['r mid'][:-3]
        if redshift in [4, 5, 6]:
            ax.plot(temp_rmid, (temp_rmid/MBII_fig25_r0[i+5])**(-MBII_gamma), label='Khandai et al. 2015 (coeval?)')

        # put legends outside the plot
        ax.legend(title='MassiveBlack-II (z = {})'.format(redshift), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(visible=False)
        # set the box to be square
        ax.set_box_aspect(1)

        # Adjust the padding for x-axis tick labels
        ax.tick_params(axis='x', pad=10)

        # Save the figure as a pdf
        plt.savefig(PLOT_DIRECTORY + 'MBII_coeval_3lumbin_corrfunc_z{}.pdf'.format(redshift), bbox_inches='tight')
        plt.show()


def single_plot(unique_id, toyn): 
    # Load the correlation function data
    corrfunc_data = []
    lc_corrfunc_data = []
    inc_lc_corrfunc_data = []
    for i in redshifts:
        corrfunc_data.append(pd.read_csv(DATA_DIRECTORY + 'MBII_1e91e12_corrfunc_z{}.csv'.format(i)))
        lc_corrfunc_data.append(pd.read_csv(DATA_DIRECTORY + '{}_MBII_lc1e91e12_corrfunc_z{}.csv'.format(unique_id, i)))
        inc_lc_corrfunc_data.append(pd.read_csv(DATA_DIRECTORY + '{}_MBII_{}inc_lc_corrfunc_z{}.csv'.format(unique_id, toyn, i)))

    # Find the midpoint of each bin, as r_mid = (r_min + r_max) / 2
    for i in range(len(redshifts)):
        corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max']) / 2
        lc_corrfunc_data[i]['r mid'] = (lc_corrfunc_data[i]['r min'] + lc_corrfunc_data[i]['r max']) / 2
        inc_lc_corrfunc_data[i]['r mid'] = (inc_lc_corrfunc_data[i]['r min'] + inc_lc_corrfunc_data[i]['r max']) / 2

    for i, redshift in enumerate(redshifts):

        fig, (ax, err_ax) = plt.subplots(2,1, figsize=(6, 8), height_ratios=[3, 1], sharex=True)

        # Plot all the correlation functions
        ax.errorbar(corrfunc_data[i]['r mid'][:-3], corrfunc_data[i]['Landy Szalay'][:-3], yerr=corrfunc_data[i]['Pois Error'][:-3], label='This work (coeval)', fmt='.', markersize=5, capsize=5)
        ax.errorbar(lc_corrfunc_data[i]['r mid'][:-3], lc_corrfunc_data[i]['Landy Szalay'][:-3], yerr=lc_corrfunc_data[i]['Pois Error'][:-3], label='This work (lightcone)', fmt='.', markersize=5, capsize=5)
        ax.errorbar(inc_lc_corrfunc_data[i]['r mid'][:-3], inc_lc_corrfunc_data[i]['Landy Szalay'][:-3], yerr=inc_lc_corrfunc_data[i]['Pois Error'][:-3], label='{} incomp (lightcone)'.format(toyn), fmt='.', markersize=5, capsize=5)
        
        ax.set_ylabel(r'$\xi(r)$')
        # ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        ax.legend()
        # plt.ylim(-0.5, 10)
        # plt.xlim(0, 150)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Now also plot the correlation function from khandai et al. 2015 as power laws. redshift 4, 5, 6 are at index 5, 6, 7
        print(redshift)
        temp_rmid = corrfunc_data[i]['r mid'][:-3]
        if redshift in [4, 5, 6]:
            ax.plot(temp_rmid, (temp_rmid/MBII_fig25_r0[i+5])**(-MBII_gamma), label='Khandai et al. 2015 (coeval?)')

        ax.legend(title='MassiveBlack-II (z = {})'.format(redshift))
        ax.grid(visible=False)
        # set the box to be square
        ax.set_box_aspect(1)

        # Adjust the padding for x-axis tick labels
        ax.tick_params(axis='x', pad=10)

        # plot the residual of the correlation function between complete and incomplete lightcone.
        residual = lc_corrfunc_data[i]['Landy Szalay'][:-3] - inc_lc_corrfunc_data[i]['Landy Szalay'][:-3]
        err_residual = np.sqrt(lc_corrfunc_data[i]['Pois Error'][:-3]**2 + inc_lc_corrfunc_data[i]['Pois Error'][:-3]**2)
        err_ax.errorbar(lc_corrfunc_data[i]['r mid'][:-3], residual, yerr=err_residual, fmt='.', markersize=5, capsize=5)
        err_ax.set_ylabel(r'$\Delta \xi(r)$')
        err_ax.set_xlabel(r'r ($h^{-1}$ cMpc)')

        # err_ax.set_box_aspect(1)
        err_ax.grid(visible=False)
        err_ax.axhline(0, color='black', linestyle='--', alpha=0.2)

        # Save the figure as a pdf
        # plt.savefig(PLOT_DIRECTORY + 'MBII_toy2inc_lc_corrfunc_z{}.pdf'.format(redshift), bbox_inches='tight')
        plt.savefig(PLOT_DIRECTORY + '{}_MBII_{}inc_lc_corrfunc_z{}.pdf'.format(unique_id, toyn, redshift), bbox_inches='tight')

        plt.show()

# single_plot('iter1')

def avg_plot(unique_ids, toyn):

    # Load the coeval correlation function data
    corrfunc_data = []
    for i in redshifts:
        corrfunc_data.append(pd.read_csv(DATA_DIRECTORY + 'MBII_1e91e12_corrfunc_z{}.csv'.format(i)))
    for i in range(len(redshifts)):
        corrfunc_data[i]['r mid'] = (corrfunc_data[i]['r min'] + corrfunc_data[i]['r max']) / 2

    lc_corrfunc_data = []
    inc_lc_corrfunc_data = []
    for j, unique_id in enumerate(unique_ids):
        lc_corrfunc_data.append([])
        inc_lc_corrfunc_data.append([])
        # Load the correlation function data
        for i in redshifts:
            lc_corrfunc_data[j].append(pd.read_csv(DATA_DIRECTORY + '{}_MBII_lc1e91e12_corrfunc_z{}.csv'.format(unique_id, i)))
            inc_lc_corrfunc_data[j].append(pd.read_csv(DATA_DIRECTORY + '{}_MBII_{}inc_lc_corrfunc_z{}.csv'.format(unique_id, toyn, i)))

    # Find the midpoint of each bin, as r_mid = (r_min + r_max) / 2
    for i in range(len(redshifts)):
        for j in range(len(lc_corrfunc_data)):
            lc_corrfunc_data[j][i]['r mid'] = (lc_corrfunc_data[j][i]['r min'] + lc_corrfunc_data[j][i]['r max']) / 2
            inc_lc_corrfunc_data[j][i]['r mid'] = (inc_lc_corrfunc_data[j][i]['r min'] + inc_lc_corrfunc_data[j][i]['r max']) / 2

    # Now plot the correlation function for all the iterations
    for i, redshift in enumerate(redshifts):
        fig, (ax, err_ax) = plt.subplots(2,1, figsize=(6, 8), height_ratios=[3, 1], sharex=True)
        print(redshift)

        avg_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        avg_inc_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        err_avg_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        err_avg_inc_lc_corrfunc_data = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        for j in range(len(lc_corrfunc_data)):
            avg_lc_corrfunc_data += lc_corrfunc_data[j][i]['Landy Szalay'][:-3]
            avg_inc_lc_corrfunc_data += inc_lc_corrfunc_data[j][i]['Landy Szalay'][:-3]
            err_avg_lc_corrfunc_data += np.power(lc_corrfunc_data[j][i]['Pois Error'][:-3], 2)
            err_avg_inc_lc_corrfunc_data += np.power(inc_lc_corrfunc_data[j][i]['Pois Error'][:-3], 2)

        avg_lc_corrfunc_data = avg_lc_corrfunc_data / len(lc_corrfunc_data)
        avg_inc_lc_corrfunc_data = avg_inc_lc_corrfunc_data / len(lc_corrfunc_data)
        err_avg_lc_corrfunc_data = np.sqrt(err_avg_lc_corrfunc_data / len(lc_corrfunc_data))
        err_avg_inc_lc_corrfunc_data = np.sqrt(err_avg_inc_lc_corrfunc_data / len(lc_corrfunc_data))

        ax.errorbar(lc_corrfunc_data[0][i]['r mid'][:-3], avg_lc_corrfunc_data, yerr=err_avg_lc_corrfunc_data, label='LC', fmt='.', markersize=5, capsize=5)
        ax.errorbar(inc_lc_corrfunc_data[0][i]['r mid'][:-3], avg_inc_lc_corrfunc_data, yerr=err_avg_inc_lc_corrfunc_data, label='{}INC LC'.format(toyn), fmt='.', markersize=5, capsize=5)

        # Plot the coeval correlation function
        ax.errorbar(corrfunc_data[i]['r mid'][:-3], corrfunc_data[i]['Landy Szalay'][:-3], yerr=corrfunc_data[i]['Pois Error'][:-3], label='This work (coeval)', fmt='.', markersize=5, capsize=5)

        # Now also plot the correlation function from khandai et al. 2015 as power laws. redshift 4, 5, 6 are at index 5, 6, 7
        temp_rmid = lc_corrfunc_data[0][i]['r mid'][:-3]
        if redshift in [4, 5, 6]:
            ax.plot(temp_rmid, (temp_rmid/MBII_fig25_r0[i+5])**(-MBII_gamma), label='Khandai et al. 2015 (coeval?)')

        ax.set_ylabel(r'$\xi(r)$')
        # ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        ax.legend()
        # plt.ylim(-0.5, 10)
        # plt.xlim(0, 150)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(title='MassiveBlack-II (z = {})'.format(redshift))
        ax.grid(visible=False)
        # set the box to be square
        ax.set_box_aspect(1)

        # Adjust the padding for x-axis tick labels
        ax.tick_params(axis='x', pad=10)

        ax.set_xlim(1.9, 314)

        # plot the residual of the correlation function between complete and incomplete lightcone.
        residual = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        err_residual = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        for j in range(len(lc_corrfunc_data)):
            residual += lc_corrfunc_data[j][i]['Landy Szalay'][:-3] - inc_lc_corrfunc_data[j][i]['Landy Szalay'][:-3]
            err_residual += lc_corrfunc_data[j][i]['Pois Error'][:-3]**2 + inc_lc_corrfunc_data[j][i]['Pois Error'][:-3]**2

        residual = residual / len(lc_corrfunc_data)
        err_residual = np.sqrt(err_residual / len(lc_corrfunc_data)) 

        err_ax.errorbar(lc_corrfunc_data[0][i]['r mid'][:-3], residual, yerr=err_residual, fmt='.', markersize=5, capsize=5)
        err_ax.set_ylabel(r'$\Delta \xi(r)$')
        err_ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        err_ax.grid(visible=False)
        err_ax.axhline(0, color='black', linestyle='--', alpha=0.2)

        # Save the figure as a pdf
        plt.savefig(PLOT_DIRECTORY + 'MBII_{}inc_lc_corrfunc_z{}.pdf'.format(toyn, redshift), bbox_inches='tight')

        plt.show()

# avg_plot(['iter1', 'iter2', 'iter3', 'iter4', 'iter5', 'iter6', 'iter7', 'iter8'], 'toy2.2')
coeval_plot()