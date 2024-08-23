# 1) Load the 7 files which contains MBII black holes details in slices centered at 4, 5, 6, 7, 8, 9, 10 redshifts.
# 2) Extract the x, y, and z coordinates of the black holes.
# 3) Define the number of pixels in each dimension (50, 50, 50). and pixelated the data.
# 4) store each array which is n(x,y) for each redshift in a list called nz_arrays.


import numpy as np
import matplotlib.pyplot as plt
# Import interp1d
from scipy.interpolate import interp1d
# Import the cosmology
from astropy.cosmology import FlatLambdaCDM
# Import the LogNorm object
from matplotlib.colors import LogNorm

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Define the plot directory
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/"

# Define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Define the reduced Hubble constant
h = 0.7

# Define the redshifts
redshifts = np.array([4, 5, 6, 7, 8, 9, 10])

file_paths = ['/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_034.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_029.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_026.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_024.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_020.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_019.txt',
              '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/bhprops_018.txt']

nz_arrays = []

# defining the luminosity bin
min_lum = 1e11
max_lum = np.infty

# format the min_lum in order to use in plot and file names.
# Convert the number to scientific notation and replace 'e+' with 'e'
formatted_min_lum = "{:.0e}".format(min_lum).replace('e+0', 'e').replace('e+','e')

BH_density = []

# Define the number of pixels in each dimension
num_pixels_x = 50
num_pixels_y = 50
num_pixels_z = 50

# Calculate the pixel size in each dimension
x_range = (0, 100000) #(np.min(x_coordinates), np.max(x_coordinates))
y_range = (0, 100000) #(np.min(y_coordinates), np.max(y_coordinates))
z_range = (0, 100000) #(np.min(z_coordinates), np.max(z_coordinates))

pixel_size_x = (x_range[1] - x_range[0]) / num_pixels_x
pixel_size_y = (y_range[1] - y_range[0]) / num_pixels_y
pixel_size_z = (z_range[1] - z_range[0]) / num_pixels_z

for file_path in file_paths:
    # Load the data from the file
    data = np.loadtxt(file_path)
    
    # Extract the x, y, and z coordinates
    x_coordinates = data[:, 1]
    y_coordinates = data[:, 2]
    z_coordinates = data[:, 3]
    bh_lum = data[:, 8]

    # Find the black hole number density
    BH_density.append(len(data[:, 0]))

    # I want to introduce luminosity cuts at this point
    bh_lum_sol = bh_lum * 1.472 * 10 ** 12 # 1 M0/yr * 0.1 * c^2 equvialent to 1.472 x 10^12 L0

    # now get the indices of the black holes that are within the luminosity range
    ind_lum = np.where((bh_lum_sol >= min_lum) & (bh_lum_sol < max_lum))[0]

    x_coordinates = x_coordinates[ind_lum]
    y_coordinates = y_coordinates[ind_lum]
    z_coordinates = z_coordinates[ind_lum]   
    
    # Calculate the pixel indices for each coordinate
    pixel_indices_x = ((x_coordinates - x_range[0]) / pixel_size_x).astype(int)
    pixel_indices_y = ((y_coordinates - y_range[0]) / pixel_size_y).astype(int)
    pixel_indices_z = ((z_coordinates - z_range[0]) / pixel_size_z).astype(int)
    
    # Create an empty n(x, y, z) array
    n_array = np.zeros((num_pixels_x, num_pixels_y, num_pixels_z))
    
    # Count the number of particles in each pixel
    for i in range(len(x_coordinates)):
        n_array[pixel_indices_x[i], pixel_indices_y[i], pixel_indices_z[i]] += 1
    
    nz_arrays.append(n_array)

# Divide by the volume of the simulation box to get the number density
BH_density = np.array(BH_density) / (100**3)

# Remove the h^3 factor from the number density
BH_density = BH_density * h**3

# Convert redshifts to comoving distances
red_com = cosmo.comoving_distance(redshifts).value * h    

# Now each array in nz_arrays contains the n(x, y, z) for each redshift slice centered at 4, 5, 6, 7, 8, 9, 10.

# First convert the redshifts to comoving distances

# Calculate the comoving distances
new_z_axis = cosmo.comoving_distance(redshifts).value * h  # Convert to h^-1 Mpc

# These are the slice centers in the z direction for each redshift slice.

# Now I want to interpolate the data in nz_arrays to get the n(x, y, z) for any comoving distance in the range of new_z_axis.
sim_evolv_linear = interp1d(new_z_axis, nz_arrays, axis=0)

# Calculate the pixel size in each dimension in Mpch^-1
x_range = (0, 100) #(np.min(x_coordinates), np.max(x_coordinates))
y_range = (0, 100) #(np.min(y_coordinates), np.max(y_coordinates))
z_range = (0, 100) #(np.min(z_coordinates), np.max(z_coordinates))

pixel_size_x = (x_range[1] - x_range[0]) / num_pixels_x
pixel_size_y = (y_range[1] - y_range[0]) / num_pixels_y
pixel_size_z = (z_range[1] - z_range[0]) / num_pixels_z

# Now I want to make another 3d array which contains the n(x, y, z) for each comoving distance in the range of new_z_axis.
# Define the new z axis which has the resolution of the simulation box.

new_z_axis = np.arange(new_z_axis[0], new_z_axis[-1], pixel_size_z)

# Create an empty 3d array with shape same as the simulation along x and y
# and the length of the new z axis.

Lightcone = np.zeros((num_pixels_x, num_pixels_y, len(new_z_axis)))

# Now we need to fill the Lightcone array with the data from sim_evolv_linear, where when we move 1 comoving distance in the 
# z direction of the Lightcone array, we move 1 comoving distance in direction chosen at the start in the sim_evolv_linear array.

# We need a function of the following nature:
# Inputs: a

# This funcition finds the started plane of the simulation box given a normal vector and a center point.

from scipy.ndimage import map_coordinates

def create_rotated_plane(normal_vector, plane_center, num_pixels_plane):
    """
    Extract a 2D plane from a 3D array that makes an arbitrary angle with the axes.
    
    Parameters:
    array_3d (ndarray): The input 3D array.
    normal_vector (ndarray): A 3D vector normal to the plane.
    plane_center (ndarray): A 3D point representing the center of the plane.
    plane_size (tuple): The size of the plane in the form (width, height).
    plane_resolution (tuple): The resolution of the plane grid in the form (num_points_x, num_points_y).
    
    Returns:
    ndarray: A 2D array representing the extracted plane.
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Create a grid of points in the plane
    x_size, y_size = num_pixels_plane
    x = np.arange(0, x_size)
    y = np.arange(0, y_size)
    xv, yv = np.meshgrid(x, y)
    
    # Create the plane basis vectors
    if np.all(normal_vector == [0, 0, 1]):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    else:
        u = np.cross(normal_vector, [0, 0, 1])
        u = u / np.linalg.norm(u)
        v = np.cross(normal_vector, u)
    
    # Transform plane points to 3D coordinates
    plane_points = plane_center[:, np.newaxis, np.newaxis] + u[:, np.newaxis, np.newaxis] * xv + v[:, np.newaxis, np.newaxis] * yv

    return plane_points

def trans_plane_points(normal_vector, plane_points):
    # This function transforms the plane points to the new plane points, translated alonged the normal vector by delattr.
    # Define the new plane points
    new_plane_points = plane_points + normal_vector[:, np.newaxis, np.newaxis]

    return new_plane_points

# calculate the volume of one slice z plane of the lightcone
volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * pixel_size_z

# Define the normal vector and center point of the plane
normal_vector = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5)]) #np.array([np.cos(np.radians(15)), np.sin(np.radians(15)), 0.1])

# Define the center point of the plane
plane_center = np.array([20, 10, 5])

plane_size = (x_range[1], y_range[1])
plane_resolution = (pixel_size_x, pixel_size_y)

# Define the plot style
plt.style.use('MNRAS_Style.mplstyle')

# Define the figure size
plt.figure(figsize=(15, 5))

# Create the plane points
plane_points = create_rotated_plane(normal_vector, plane_center, num_pixels_plane=(num_pixels_x, num_pixels_y))

# Find the values from the simulation box corresponding to the plane.
# grid wrap is used to apply periodic boundary conditions
for i, r in enumerate(new_z_axis):
        # print(np.shape(plane_points))
        Lightcone[:,:,i] = map_coordinates(sim_evolv_linear(r), plane_points, order=1, mode='grid-wrap')

        # move the plane points along the normal vector by pixel_size_z.
        plane_points = trans_plane_points(normal_vector, plane_points)

# Save the lightcone to a file
np.save(DATA_DIRECTORY + 'Lightcone_lumcut{}.npy'.format(formatted_min_lum), Lightcone)    

# Calculate the number density on lightcone
number_density = np.sum(Lightcone, axis=(0, 1)) / volume

# number density is in units of particles per Mpc/h^3. Convert it to particles per Mpc^3
number_density = number_density * h**3

# Convert redshifts to comoving distances
red_com = cosmo.comoving_distance(redshifts).value * h

# Plot the number density as a function of redshift
plt.plot(new_z_axis, number_density, label='normal_vector = {}'.format(np.round(normal_vector, 2)))

# Calculate the number density on lightcone
number_density = np.sum(Lightcone, axis=(0, 1)) / volume

# number density is in units of particles per Mpc/h^3. Convert it to particles per Mpc^3
number_density = number_density * h**3

# Plot the BH density as a function of redshift with points
plt.plot(red_com, BH_density, 'o--', label='Simulation Box')

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
plt.savefig(PLOT_DIRECTORY + 'NumberDensityLightcone_lumcut{}.pdf'.format(formatted_min_lum))

plt.show()

# sum over the x axis to get the 2d projection of the lightcone.
Lightcone_2d = np.average(Lightcone[25:30,:,:], axis=0) #np.sum(Lightcone, axis=0)

# Plot the 2d projection of the lightcone
# figure size

#-----

plt.style.use('MNRAS_Style.mplstyle')

# increase the font size
plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(15, 5))
ax2 = ax.twiny()
ax.grid(visible=False)
ax2.grid(visible=False)

# Get the minimum and maximum values of the data
vmin = 1 #np.min(1 + Lightcone_2d)
vmax = 3 #np.max(1 + Lightcone_2d)

# Plot the data with imshow
cax = ax.imshow(1 + Lightcone_2d, origin='lower', cmap='viridis', extent=[new_z_axis[0], new_z_axis[-1], x_range[0], x_range[1]], norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='none', rasterized=True)

ax.set_ylabel(r'y ($h^{-1}$cMpc)')

# modify the color bar ticks by adding -1 to each tick

# Create the colorbar
ticks = [1, 2, 3]
cbar = fig.colorbar(cax, ticks=ticks, pad=0.01)
cbar.ax.set_yticklabels(int(tick - 1) for tick in ticks)
cbar.set_label('Number of black holes')

# Reducing the distance between the colorbar and the plot

# make the plot square
# ax.set_aspect('auto')
# Convert redshifts to comoving distances
red_com = cosmo.comoving_distance(redshifts).value * h # Convert to h^-1 Mpc

# remove the xticks and xticklabels
ax.set_xticks([])

# Now add the redcom values as xticks
ax.set_xticks(red_com, redshifts + 1)
ax.set_xlabel('z + 1')

# Getting the box aspect ratio of the plot, so that the we can determine the x limit inorder to make the pixels square.

ax.set_xlim(red_com[1] - 150, red_com[1] + 150)


# Add another x axis at the top of the plot with comoving distances
ax2.set_xticks(np.linspace(red_com[1] - 150, red_com[1] + 150, 5, dtype = int)[1:-1])
ax2.set_xticklabels(np.linspace(red_com[1] - 150, red_com[1] + 150, 5, dtype = int)[1:-1])
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel(r'comoving distance ($h^{-1}$cMpc)')

# Save the figure to a pdf file
plt.savefig(PLOT_DIRECTORY + 'Lightcone2d_lumcut{}.pdf'.format(formatted_min_lum))

plt.show()

#---

# Making the catalog of the lightcone
# Define the number of black holes in each pixel
n_bh = np.copy(Lightcone)

# Initialize an empty list to store the black hole coordinates
bh_coordinates = []

# Calculate the number of slices
new_num_slices = len(new_z_axis)

# Define the pixel centers
pixel_centers_x = np.linspace(x_range[0] + pixel_size_x / 2, x_range[1] - pixel_size_x / 2, num_pixels_x)
pixel_centers_y = np.linspace(y_range[0] + pixel_size_y / 2, y_range[1] - pixel_size_y / 2, num_pixels_y)
pixel_centers_z = np.copy(new_z_axis)

# Iterate over the pixels and distribute the black holes
for k in range(new_num_slices):
    for i in range(num_pixels_x):
        for j in range(num_pixels_y):
            # Get the number of black holes in the current pixel
            n = round(n_bh[i, j, k])
            
            # Generate random x, y, z coordinates for the black holes in the pixel
            x_coords = np.random.uniform(low=pixel_centers_x[i] - pixel_size_x / 2, high=pixel_centers_x[i] + pixel_size_x / 2, size=n)
            y_coords = np.random.uniform(low=pixel_centers_y[j] - pixel_size_y / 2, high=pixel_centers_y[j] + pixel_size_y / 2, size=n)
            z_coords = np.random.uniform(low=pixel_centers_z[k] - pixel_size_z / 2, high=pixel_centers_z[k] + pixel_size_z / 2, size=n)
            
            # Append the coordinates to the list
            bh_coordinates.extend(list(zip(x_coords, y_coords, z_coords)))

# Convert the list of coordinates to a numpy array
bh_coordinates = np.array(bh_coordinates)

# This concludes the production of the light cone. bh_coordinates now consists of the x, y, z coordinates of the black holes in the light cone, extrapolated
# from the MBII data.

# Now I want to save the bh_coordinates to a file in the data directory so that I can use it in the future.
# I want to save the numpy array to a file using np.save() function.

# Save the bh_coordinates to a file
np.save(DATA_DIRECTORY + 'bh_coordinates_lightcone_lumcut{}_v3.npy'.format(formatted_min_lum), bh_coordinates)