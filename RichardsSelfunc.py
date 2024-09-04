from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"
# define the Plot directory
PLOT_DIRECTORY = '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/'

# Define the plot style
plt.style.use('MNRAS_Style.mplstyle')

# Read Richards et al. 2006 selection function data in file Richards2006SelectionFunc.txt in data directory.
Richards2006SelectionFunc = ascii.read(DATA_DIRECTORY + 'Richards2006SelectionFunc.dat')

# Add 4.0 to the redshifts in the selection function data, so that the redshifts are now from 4.0 to 10.0.
Richards2006SelectionFunc['z'] += 4.0

# Define the minimum and the maximum imag values.
imag_min = np.min(Richards2006SelectionFunc['imag'])
imag_max = np.max(Richards2006SelectionFunc['imag'])

# Make 3 bins out of the range of imag values.
imag_bins = np.linspace(imag_min, imag_max, 4)

# Make a 2d array with lenth of imag_bins - 1 in the first dimension and length of unique z's in Richards2006SelectionFunc['z'] in the second dimension.
# This array will store the selection function values for each imag bin and redshift.
selection_func_values = np.zeros((len(imag_bins) - 1, len(np.unique(Richards2006SelectionFunc['z']))))

for i in range(len(imag_bins) - 1):
    # choose only the data with imag that are within the imag_min and imag_max values.
    Richards2006SelectionFunc_lumbin = Richards2006SelectionFunc[(Richards2006SelectionFunc['imag'] >= imag_bins[i]) & (Richards2006SelectionFunc['imag'] <= imag_bins[i + 1])]

    # Group the data by the redshifts.
    Richards2006SelectionFunc_lumbin_grouped = Richards2006SelectionFunc_lumbin.group_by('z')

    # Take the average of the selection function values for each redshift.
    Richards2006SelectionFunc_lumbin_grouped_avg = Richards2006SelectionFunc_lumbin_grouped.groups.aggregate(np.mean)

    # Plot the point values as a function of redshift.
    plt.plot(Richards2006SelectionFunc_lumbin_grouped_avg['z'], Richards2006SelectionFunc_lumbin_grouped_avg['point'], '.-', linewidth=0.6, markersize=0.7, label=str(round(imag_bins[i],2)) + ' to ' + str(round(imag_bins[i + 1],2)))

    selection_func_values[i] = Richards2006SelectionFunc_lumbin_grouped_avg['point']

# Make a two dimensional array that stores the selection function values for each redshift and imags.
selection_func_values_all = np.zeros((len(np.unique(Richards2006SelectionFunc['z'])), len(np.unique(Richards2006SelectionFunc['imag']))))

for i, z in enumerate(np.unique(Richards2006SelectionFunc['z'])):
    Richards2006SelectionFunc_z_grouped = Richards2006SelectionFunc[Richards2006SelectionFunc['z'] == z]

    for j, imag in enumerate(np.unique(Richards2006SelectionFunc['imag'])):
        Richards2006SelectionFunc_z_grouped_imag = Richards2006SelectionFunc_z_grouped[Richards2006SelectionFunc_z_grouped['imag'] == imag]

        selection_func_values_all[i, j] = Richards2006SelectionFunc_z_grouped_imag['point']


plt.xlim(4, 10.1)
plt.ylim(0, 1.05)
plt.xlabel('Redshift')
plt.ylabel('Completeness')
plt.legend(title=r'$i_{\mathrm{mag}}$', loc='lower right')
plt.grid(visible=False)
plt.gca().set_box_aspect(1)

# Save the plot as a pdf file.
plt.savefig(PLOT_DIRECTORY + 'CompletenessVsRedshift.pdf', bbox_inches='tight')
plt.show()

# Make a 2d plot of the selection function values.
plt.figure()
plt.imshow(selection_func_values, extent=[4, 10, imag_min, imag_max], aspect='auto')
plt.colorbar(label='Completeness')
plt.xlabel('Redshift')
plt.ylabel(r'$i_{\mathrm{mag}}$')
plt.grid(visible=False)
plt.gca().set_box_aspect(1)

# Save the plot as a pdf file.
plt.savefig(PLOT_DIRECTORY + 'CompletenessVsRedshift_imag.pdf', bbox_inches='tight')
plt.show()

z_min = np.min(Richards2006SelectionFunc['z'])
z_max = np.max(Richards2006SelectionFunc['z'])
imag_min = np.min(Richards2006SelectionFunc['imag'])
imag_max = np.max(Richards2006SelectionFunc['imag'])

# Make a 2d plot of the selection function values.
plt.figure()
plt.imshow(selection_func_values_all.T, extent=[z_min, z_max, imag_min, imag_max], aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Completeness')
plt.xlabel('Redshift')
plt.ylabel(r'$i_{\mathrm{mag}}$')
plt.grid(visible=False)
plt.gca().set_box_aspect(1)

# Save the plot as a pdf file.
plt.savefig(PLOT_DIRECTORY + 'CompletenessVsRedshift_imag_all.pdf', bbox_inches='tight')
plt.show()

