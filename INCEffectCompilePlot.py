# plot the correlation function from MBII_corrfunc_z{}.csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('MNRAS_Style.mplstyle')

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/INCeffectCompilData/"
# Define the data2.0 directory
DATA_DIRECTORY_main = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/"
# define the Plot directory
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots2.0/"

# Define the ploting style
plt.style.use('MNRAS_Style.mplstyle')

# Define the redshifts
redshifts = np.array([4, 5, 6, 7])

# Load the Khadai et al. 2015 Fig 25 Lum>10^9 Lsun data
temp_DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/"

# Now also plot the correlation function from khandai et al. 2015 as power laws
MBII_fig25_filename = "Khandai_fig25_data.csv"
MBII_fig25 = np.loadtxt(temp_DATA_DIRECTORY + MBII_fig25_filename, skiprows=1, delimiter=',')

MBII_fig25_z = MBII_fig25[:, 0]
MBII_fig25_r0 = MBII_fig25[:, 1]

MBII_gamma = 2.0

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
        # residual = lc_corrfunc_data[i]['Landy Szalay'][:-3] - inc_lc_corrfunc_data[i]['Landy Szalay'][:-3]
        # err_residual = np.sqrt(lc_corrfunc_data[i]['Pois Error'][:-3]**2 + inc_lc_corrfunc_data[i]['Pois Error'][:-3]**2)
        # err_ax.errorbar(lc_corrfunc_data[i]['r mid'][:-3], residual, yerr=err_residual, fmt='.', markersize=5, capsize=5)
        # err_ax.set_ylabel(r'$\Delta \xi(r)$')
        # err_ax.set_xlabel(r'r ($h^{-1}$ cMpc)')

        # Plot the relative error between complete and incomplete correlation functions
        # Define the values using the new notation
        xi_incomplete = inc_lc_corrfunc_data[i]['Landy Szalay'][:-3]
        xi_complete = lc_corrfunc_data[i]['Landy Szalay'][:-3]

        sigma_incomplete = inc_lc_corrfunc_data[i]['Pois Error'][:-3] # absolute errors
        sigma_complete = lc_corrfunc_data[i]['Pois Error'][:-3]

        # Compute the error in the relative difference
        rel_diff = 1 - xi_incomplete/xi_complete
        err_rel_diff = np.sqrt(((xi_incomplete / xi_complete**2) * sigma_complete)**2 + 
                            (sigma_incomplete / xi_complete)**2)

        err_ax.errorbar(lc_corrfunc_data[i]['r mid'][:-3], rel_diff, yerr=err_rel_diff, fmt='.', markersize=5, capsize=5)
        err_ax.set_ylabel(r'$\frac{\Delta \xi(r)}{\xi(r)}')
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

        # # plot the residual of the correlation function between complete and incomplete lightcone.
        # residual = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        # err_residual = np.zeros(len(lc_corrfunc_data[0][i]['r mid'][:-3]))
        # for j in range(len(lc_corrfunc_data)):
        #     residual += lc_corrfunc_data[j][i]['Landy Szalay'][:-3] - inc_lc_corrfunc_data[j][i]['Landy Szalay'][:-3]
        #     err_residual += lc_corrfunc_data[j][i]['Pois Error'][:-3]**2 + inc_lc_corrfunc_data[j][i]['Pois Error'][:-3]**2

        # residual = residual / len(lc_corrfunc_data)
        # err_residual = np.sqrt(err_residual / len(lc_corrfunc_data)) 

        # Save data to CSV
        data = {
            "r mid": lc_corrfunc_data[0][i]['r mid'][:-3],
            "avg_lc_corrfunc": avg_lc_corrfunc_data,
            "avg_inc_lc_corrfunc": avg_inc_lc_corrfunc_data,
            "err_avg_lc_corrfunc": err_avg_lc_corrfunc_data,
            "err_avg_inc_lc_corrfunc": err_avg_inc_lc_corrfunc_data,
        }
        residual = avg_lc_corrfunc_data - avg_inc_lc_corrfunc_data
        err_residual = np.sqrt(err_avg_lc_corrfunc_data**2 + err_avg_inc_lc_corrfunc_data**2)
        data["residual"] = residual
        data["err_residual"] = err_residual

        output_filename = DATA_DIRECTORY + 'MBII_{}inc_lc_corrfunc_avg_z{}.csv'.format(toyn, redshift)
        pd.DataFrame(data).to_csv(output_filename, index=False)
        print(f"Saved averaged data to {output_filename}")

        err_ax.errorbar(lc_corrfunc_data[0][i]['r mid'][:-3], residual, yerr=err_residual, fmt='.', markersize=5, capsize=5)
        err_ax.set_ylabel(r'$\Delta \xi(r)$')
        err_ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        err_ax.grid(visible=False)
        err_ax.axhline(0, color='black', linestyle='--', alpha=0.2)

        # Save the figure as a pdf
        plt.savefig(PLOT_DIRECTORY + 'MBII_{}inc_lc_corrfunc_z{}.pdf'.format(toyn, redshift), bbox_inches='tight')

        plt.show()

# Define the power-law function
def power_law(r, r0, gamma):
    return (r/r0)**(-gamma)

def avg_plot_with_fit(unique_ids, toyn):

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

        # Fit avg_lc_corrfunc_data and avg_inc_lc_corrfunc_data with power-law fits
        popt_lc, pcov_lc = curve_fit(power_law, lc_corrfunc_data[0][i]['r mid'][:-3], avg_lc_corrfunc_data, sigma=err_avg_lc_corrfunc_data)
        popt_inc_lc, pcov_inc_lc = curve_fit(power_law, inc_lc_corrfunc_data[0][i]['r mid'][:-3], avg_inc_lc_corrfunc_data, sigma=err_avg_inc_lc_corrfunc_data)

        # Extract fitted parameters
        r0_lc, gamma_lc = popt_lc
        r0_inc_lc, gamma_inc_lc = popt_inc_lc

        # Generate the fitted curves
        lc_fit = power_law(lc_corrfunc_data[0][i]['r mid'], r0_lc, gamma_lc)
        inc_lc_fit = power_law(lc_corrfunc_data[0][i]['r mid'], r0_inc_lc, gamma_inc_lc)

        # Plot the power law curves
        ax.plot(lc_corrfunc_data[0][i]['r mid'], lc_fit, label='LC Fit: $r_0={:.2f}, \\gamma={:.2f}$'.format(r0_lc, gamma_lc), linestyle='--')
        ax.plot(lc_corrfunc_data[0][i]['r mid'], inc_lc_fit, label='{}INC LC Fit: $r_0={:.2f}, \\gamma={:.2f}$'.format(toyn, r0_inc_lc, gamma_inc_lc), linestyle='--')

        # Plot the coeval correlation function
        ax.errorbar(corrfunc_data[i]['r mid'][:-3], corrfunc_data[i]['Landy Szalay'][:-3], yerr=corrfunc_data[i]['Pois Error'][:-3], label='This work (coeval)', fmt='.', markersize=5, capsize=5)

        # Now also plot the correlation function from khandai et al. 2015 as power laws. redshift 4, 5, 6 are at index 5, 6, 7
        temp_rmid = lc_corrfunc_data[0][i]['r mid'][:-3]
        if redshift in [4, 5, 6]:
            ax.plot(temp_rmid, (temp_rmid/MBII_fig25_r0[i+5])**(-MBII_gamma), label='Khandai et al. 2015 (coeval?)')

        ax.set_ylabel(r'$\xi(r)$')
        # ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        # plt.ylim(-0.5, 10)
        # plt.xlim(0, 150)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(title='MassiveBlack-II (z = {})'.format(redshift), loc='center left', bbox_to_anchor=(1, 0.5))
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
        plt.savefig(PLOT_DIRECTORY + 'MBII_{}inc_lc_corrfunc_z{}_fitted.pdf'.format(toyn, redshift), bbox_inches='tight')

        plt.show()

def replot_from_csv(toyn, redshifts):
    for redshift in redshifts:
        # Construct the file paths
        csv_file = DATA_DIRECTORY + 'MBII_{}inc_lc_corrfunc_avg_z{}.csv'.format(toyn, redshift)
        coeval_file = DATA_DIRECTORY + 'MBII_1e91e12_corrfunc_z{}.csv'.format(redshift)

        # Load the averaged lightcone data
        try:
            data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"File not found: {csv_file}")
            continue

        # Load the coeval correlation function data
        try:
            coeval_data = pd.read_csv(coeval_file)
            coeval_data['r mid'] = (coeval_data['r min'] + coeval_data['r max']) / 2
            # trim the last 3 data points
            coeval_data = coeval_data[:-3]
        except FileNotFoundError:
            print(f"Coeval file not found: {coeval_file}")
            continue

        # Plot the data
        fig, (ax, err_ax) = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[3, 1], sharex=True)

        # Correlation function plot
        ax.errorbar(
            data['r mid'], data['avg_lc_corrfunc'],
            yerr=data['err_avg_lc_corrfunc'], label='LC', fmt='.', markersize=5, capsize=5
        )
        ax.errorbar(
            data['r mid'], data['avg_inc_lc_corrfunc'],
            yerr=data['err_avg_inc_lc_corrfunc'], label='{}INC LC'.format(toyn), fmt='.', markersize=5, capsize=5
        )

        # Add coeval data
        ax.errorbar(
            coeval_data['r mid'], coeval_data['Landy Szalay'],
            yerr=coeval_data['Pois Error'], label='Coeval', fmt='.', markersize=5, capsize=5
        )

        # Now also plot the correlation function from khandai et al. 2015 as power laws. redshift 4, 5, 6 are at index 5, 6, 7
        if redshift in [4, 5, 6]:
            ax.plot(coeval_data['r mid'], (coeval_data['r mid']/MBII_fig25_r0[redshift+1])**(-MBII_gamma), label='Khandai et al. 2015 (coeval?)')

        # Customize the plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$\xi(r)$')
        ax.legend(title=f'MassiveBlack-II (z = {redshift})')
        ax.grid(visible=False)
        ax.set_xlim(1.9, 314)

        # Residual plot
        err_ax.errorbar(
            data['r mid'], data['residual'],
            yerr=data['err_residual'], fmt='.', markersize=5, capsize=5
        )
        err_ax.set_ylabel(r'$\Delta \xi(r)$')
        err_ax.set_xlabel(r'r ($h^{-1}$ cMpc)')
        err_ax.axhline(0, color='black', linestyle='--', alpha=0.2)
        err_ax.grid(visible=False)

        plt.show()


from astropy.cosmology import FlatLambdaCDM

# Define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h = 0.7

redshifts = np.array([4, 5, 6, 7])
red_com = cosmo.comoving_distance(redshifts).value * h #h^-1 Mpc

def compilation_error_axes_plot_with_toy_model(toys, redshifts):
    # Set up the grid for subplots
    fig, axes = plt.subplots(
        len(toys), len(redshifts) + 1,  # Extra column for the toy model
        figsize=(4 * (len(redshifts) + 1), 3 * len(toys)), 
        sharex=False, sharey=False  # Allow independent scaling
    )
    fig.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

    for toy_idx, toyn in enumerate(toys):
        # Plot the toy model as the first column
        ax_toy = axes[toy_idx, 0] if len(toys) > 1 else axes[0]

        # Construct the file path for the toy model selection function
        toy_csv_file = DATA_DIRECTORY_main + f"{toyn}_SelectionFunction.csv"
        
        # Load the toy model data
        try:
            selection_function_df = pd.read_csv(toy_csv_file)
            new_z_axis = selection_function_df['z_axis']
            func = selection_function_df['func']
        except FileNotFoundError:
            print(f"Toy model file not found: {toy_csv_file}")
            continue

        # Plot the toy model data
        ax_toy.plot(new_z_axis, func, label=f'{toyn} model')
        ax_toy.set_xlabel('Comoving distance (Mpc/h)')
        ax_toy.set_ylabel('Completeness')
        ax_toy.set_ylim(0, 1.1)
        ax_toy.legend()
        # ax_toy.set_title(f"{toyn} Model")
        
        # Add twin axis for redshift
        ax_toy2 = ax_toy.twiny()
        ax_toy2.set_xlim(ax_toy.get_xlim())
        ax_toy2.set_xticks(red_com)
        ax_toy2.set_xticklabels(redshifts)
        # ax_toy2.set_xlabel("Redshift")

        # Loop through redshifts for the remaining columns
        for redshift_idx, redshift in enumerate(redshifts):
            ax = axes[toy_idx, redshift_idx + 1] if len(toys) > 1 else axes[redshift_idx + 1]

            # Construct the file paths for residuals
            csv_file = DATA_DIRECTORY + f'MBII_{toyn}inc_lc_corrfunc_avg_z{redshift}.csv'

            # Load the averaged lightcone data
            try:
                data = pd.read_csv(csv_file)
            except FileNotFoundError:
                print(f"File not found: {csv_file}")
                continue

            # Residual plot (error axes)
            ax.errorbar(
                data['r mid'], data['residual'],
                yerr=data['err_residual'], fmt='.', markersize=5, capsize=5
            )
            ax.axhline(0, color='black', linestyle='--', alpha=0.2)

            # Customize the plot
            ax.set_xscale('log')
            ax.grid(visible=False)
            ax.set_xlim(1.9, 314)
            ax.set_ylim(-0.3, 0.3)

            # Set titles and labels
            if toy_idx == 0:
                ax.set_title(f"z = {redshift}")
            if toy_idx == len(toys) - 1:
                ax.set_xlabel(r'r ($h^{-1}$ cMpc)')

    # Save the compilation plot
    output_plot = PLOT_DIRECTORY + 'Compilation_MBII_residuals_with_toy_model.pdf'
    plt.savefig(output_plot, bbox_inches='tight')
    print(f"Saved compilation plot: {output_plot}")

    plt.show()



# Set the number of iterations
n_iter = 16  # You can change this to any number of iterations

# Create the array dynamically
iter_array = [f"iter{i}" for i in range(1, n_iter + 1)]

# Call the avg_plot function with the dynamically created array
avg_plot_with_fit(iter_array, 'toy1.3')
# avg_plot(iter_array, 'toy1')

# replot_from_csv('toy1', redshifts)

# compilation_error_axes_plot_with_toy_model(['toy1', 'toy1.1', 'toy1.2', 'toy1.3'], redshifts)