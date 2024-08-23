import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Define the plot directory
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/"

# Define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Define the reduced Hubble constant
h = 0.7

# Redshifts
redshifts = np.array([4, 5, 6, 7, 8, 9, 10])

# Calculate the comoving distances
new_z_axis = cosmo.comoving_distance(redshifts).value * h  # Convert to h^-1 Mpc

# Define the x range
x_range = [0, 100]

# Define the minimum luminosities
min_lums = [1e9, 1e10, 1e11]

# Plot the 2d projection of the lightcone
plt.style.use('MNRAS_Style.mplstyle')

# Increase the font size
plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True)

# Store the imshow objects in a list
im_list = []

for i, min_lum in enumerate(min_lums):
	formatted_min_lum = "{:.0e}".format(min_lum).replace('e+0', 'e').replace('e+', 'e')
	
	# Read the light cone file
	Lightcone = np.load(DATA_DIRECTORY + 'Lightcone_lumcut{}.npy'.format(formatted_min_lum))
	
	# Sum over the x axis to get the 2d projection of the lightcone
	Lightcone_2d = np.average(Lightcone[25:30, :, :], axis=0)
	
	ax = axes[i]
	# ax2 = ax.twiny()
	ax.grid(visible=False)
	# ax2.grid(visible=False)
	
	# Get the minimum and maximum values of the data
	vmin = 1
	vmax = 3
	
	# Plot the data with imshow
	im = ax.imshow(1 + Lightcone_2d, origin='lower', cmap='viridis', extent=[new_z_axis[0], new_z_axis[-1], x_range[0], x_range[1]], norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='none', rasterized=True)
	im_list.append(im)

	# # Create the colorbar
	# ticks = [1, 2, 3]
	# cbar = fig.colorbar(cax, ax=ax, ticks=ticks, pad=0.01)
	# cbar.ax.set_yticklabels([str(int(tick - 1)) for tick in ticks])
	# cbar.set_label('Number of black holes')
	
	# Convert redshifts to comoving distances
	red_com = cosmo.comoving_distance(redshifts).value * h  # Convert to h^-1 Mpc
	
	# Remove the xticks and xticklabels
	ax.set_xticks([])
	
	# Now add the redcom values as xticks
	ax.set_xticks(red_com)
	ax.set_xticklabels(redshifts + 1)
	ax.set_xlabel('z + 1')
	
	# Getting the box aspect ratio of the plot, so that the we can determine the x limit in order to make the pixels square.
	ax.set_xlim(red_com[1] - 150, red_com[1] + 150)
	
	# Add another x axis at the top of the plot with comoving distances
	# ax2.set_xticks(np.linspace(red_com[1] - 150, red_com[1] + 150, 5, dtype=int)[1:-1])
	# ax2.set_xticklabels(np.linspace(red_com[1] - 150, red_com[1] + 150, 5, dtype=int)[1:-1])
	# ax2.set_xlim(ax.get_xlim())
	# ax2.set_xlabel(r'comoving distance ($h^{-1}$cMpc)')

axes[1].set_ylabel(r'y ($h^{-1}$cMpc)')

# Create a single colorbar for all subplots
ticks = [1, 2, 3]
cbar = fig.colorbar(im_list[0], ax=axes, orientation='vertical', pad=0.01, fraction=0.02)
cbar.ax.set_yticklabels([str(int(tick - 1)) for tick in ticks])
cbar.set_label('Number of black holes')

# Adjust layout
# plt.tight_layout()

# Save the figure to a pdf file
# plt.savefig(PLOT_DIRECTORY + 'Lightcone2d_combined.pdf')

plt.show()


