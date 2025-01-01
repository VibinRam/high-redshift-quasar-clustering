import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM, z_at_value
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u

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

# Import the lightcone data
Lightcone1e91e10 = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/Lightcone_lumcut1e91e10.npy')
# Lightcone1e91e10_sheninc = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/Lightcone_lumcut1e91e10_sheninc.npy')
# Lightcone1e91e10_toy1inc = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/Lightcone_lumcut1e91e10_toy1inc.npy')
Lightcone1e101e11 = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/Lightcone_lumcut1e101e11.npy')
Lightcone1e111e12 = np.load('/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/Lightcone_lumcut1e111e12.npy')
Lightcones = [Lightcone1e91e10, Lightcone1e101e11, Lightcone1e111e12]

# Plot the 2d projection of the lightcone
plt.style.use('MNRAS_Style.mplstyle')

# Increase the font size
plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True)

# Store the imshow objects in a list
im_list = []

# Get the minimum and maximum values of the data
vmin = 1
vmax = 3
	
# Convert redshifts to comoving distances
red_com = cosmo.comoving_distance(redshifts).value * h  # Convert to h^-1 Mpc
xticks_com = np.linspace(red_com[1] - 150, red_com[1] + 150, 7, dtype=int)[1:-1]

for i in range(len(Lightcones)):
	# Sum over the x axis to get the 2d projection of the lightcone
	Lightcone_2d = np.average(Lightcones[i][25:30, :, :], axis=0)
	
	ax = axes[i]
	ax.grid(visible=False)
	
	# Plot the data with imshow
	im = ax.imshow(1 + Lightcone_2d, origin='lower', cmap='viridis', extent=[new_z_axis[0], new_z_axis[-1], x_range[0], x_range[1]], \
				norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='none', rasterized=True)
	im_list.append(im)

	# im.set_label("sample label") #(r'$10^{} \leq L_{{\rm bol}} < 10^{}$'.format(9 + i, 10 + i))
	# ax.legend()

	# # Create the colorbar
	# ticks = [1, 2, 3]
	# cbar = fig.colorbar(cax, ax=ax, ticks=ticks, pad=0.01)
	# cbar.ax.set_yticklabels([str(int(tick - 1)) for tick in ticks])
	# cbar.set_label('Number of black holes')

# Add the legends
legends = []
legends.append(axes[0].legend(title=r'$10^9 \leq L_{\rm bol} < 10^{10}$', loc='upper right', frameon=False))
legends.append(axes[1].legend(title=r'$10^{10} \leq L_{\rm bol} < 10^{11}$', loc='upper right', frameon=False))
legends.append(axes[2].legend(title=r'$10^{11} \leq L_{\rm bol} < 10^{12}$', loc='upper right', frameon=False))

for legend in legends:
	plt.setp(legend.get_title(), color='white')

# Getting the box aspect ratio of the plot, so that the we can determine the x limit in order to make the pixels square.
axes[0].set_xlim(red_com[1] - 150, red_com[1] + 150)

# Remove the xticks and xticklabels
axes[0].set_xticks([])

# Now add the redcom values as xticks
axes[0].set_xticks(xticks_com)
axes[0].set_xticklabels(np.round(z_at_value(cosmo.comoving_distance, xticks_com/h *u.Mpc).value + 1,2))
axes[2].set_xlabel('z + 1')

axes[1].set_ylabel(r'y ($h^{-1}$cMpc)')

yticks_temp = axes[0].get_yticks()
ytick_labels_temp = axes[0].get_yticklabels()
axes[0].set_yticks(yticks_temp[1:])
axes[0].set_yticklabels(ytick_labels_temp[1:])
axes[2].set_yticks(yticks_temp[:-1])
axes[2].set_yticklabels(np.asarray(yticks_temp[:-1], dtype=int))

# Create a single colorbar for all subplots
ticks = [1, 2, 3]
cbar = fig.colorbar(im_list[0], ax=axes, orientation='vertical', pad=0.01, fraction=0.02)
cbar.ax.set_yticks(ticks)
cbar.ax.set_yticklabels([str(int(tick - 1)) for tick in ticks])
cbar.set_label('Number of black holes')
# # Add another axis on top of im[0] to show the comoing distance in h^-1Mpc.
# ax2 = axes[0].twiny()
# # ax2.set_aspect(aspect=1, adjustable='box')
# print(axes[0].get_xlim())
# ax2.set_xlim(axes[0].get_xlim())
# ax2.set_xticks(xticks_com)
# ax2.set_xticklabels(xticks_com)
# ax2.set_xlabel('Comoving distance ($h^{-1}$ Mpc)')
# ax2.grid(visible=False)

# Adjust layout
# plt.tight_layout()

# Save the figure to a pdf file
plt.savefig(PLOT_DIRECTORY + 'Lightcone2d_1e9_1e12.pdf')

plt.show()

plt.figure()

for i in range(len(Lightcones)):
	# Calculate the number density on lightcone
	number_density = np.sum(Lightcones[i], axis=(0, 1)) / volume

	# # Plot the number density as a function of redshift
	# plt.plot(new_z_axis, number_density, label='normal_vector = {}'.format(np.round(normal_vector, 2)))

	# number density is in units of particles per Mpc/h^3. Convert it to particles per Mpc^3
	number_density = number_density * h**3

# # Plot the BH density as a function of redshift with points
# plt.plot(red_com, BH_density, 'o--', label='Simulation Box')

plt.ylabel(r"Number Density $(/cMpc^3)$")
# plt.title('Number Density as a Function of Redshift')
plt.legend()
# Convert redshifts to comoving distances
red_com = cosmo.comoving_distance(redshifts).value * h # Convert to h^-1 Mpc

# remove the xticks and xticklabels
plt.xticks([])

# Now add the redcom values as xticks
plt.xticks(red_com, redshifts + 1)
plt.xlabel('z + 1')

# Save the figure to the plots directory
plt.savefig(PLOT_DIRECTORY + 'NumberDensityLightcone_lumcut{}{}.pdf'.format(formatted_min_lum, formatted_max_lum))

plt.show()


