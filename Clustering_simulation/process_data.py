# process_data.py

import sys
import numpy as np

def process_data(npy_file):
    # Replace this with your actual data processing code
    import matplotlib.pyplot as plt
    from astropy.cosmology import z_at_value
    from astropy import units as u
    import numpy as np
    from astropy.cosmology import FlatLambdaCDM

    # Define the path to data directory
    DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/"

    # Define the cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # Define h
    h = 0.7

    # I want to make a completeness map of the light cone constructed. Basically it is a function of z, which gives the fraction of the black holes in the light cone that are detected at a given redshift.
    # First we have to bin the z axis into slices of 0.1 h^-1 Mpc, inorder to assign incompleteness to each slice.

    # Define the slice size
    slice_size = 0.1  # h^-1 Mpc

    # Define the start and end redshifts
    start_z = 4
    end_z = 10

    # Convert the start and end redshifts to comoving distances
    start_comoving = cosmo.comoving_distance(start_z).value / h # h^-1 Mpc
    end_comoving = cosmo.comoving_distance(end_z).value / h

    # Calculate the number of slices
    num_slices = int((end_comoving - start_comoving) / slice_size)

    # Calculate the z coordinates of the slices
    slice_z_coords = np.linspace(start_comoving, end_comoving, num_slices + 1)

    # Calculate the center of the slices
    slice_centers = (slice_z_coords[:-1] + slice_z_coords[1:]) / 2

    #----------------------------------------------------------------------------------------------------------------------------------------------------
    # import argparse

    # # Define the argument parser
    # parser = argparse.ArgumentParser(description='Process completeness map.')
    # parser.add_argument('completeness_map_file', type=str, help='Path to the .npy file containing the completeness map')
    # parser.add_argument('output_csv_file_comp', type=str, help='Path to the output csv file for full distribution')
    # parser.add_argument('output_csv_file_incomp', type=str, help='Path to the output csv file for selected distribution')

    # # Parse the arguments
    # args = parser.parse_args()

    # Load the completeness map from the .npy file
    completeness_map = np.load(npy_file)

    # remove the .npy extension from the file name and "completeness_map"
    npy_file = npy_file.replace("completeness_map", "")
    npy_file = npy_file.replace(".npy", "")
    # remove the path from the file name
    npy_file = npy_file.split("/")[-1]

    print(npy_file)
    # # Let's plot the completeness map
    # plt.figure(figsize=(12, 8))
    # plt.plot(slice_centers, completeness_map, marker='o')
    # plt.xlabel('Comoving distance (h^-1 Mpc)')
    # plt.ylabel('Completeness')
    # plt.title('Completeness map')
    # plt.grid(True)
    # plt.show()

    #----------------------------------------------------------------------------------------------------------------------------------------------------

    # Completenss map is basically the number of black holes that are selected from the given slice out of the total black holes in the slice.
    # x_coords, y_coords, z_coords are the coordinates of the black holes in the light cone constructed.
    # I want to find the slice to which each black hole belongs and choose randomly a fraction of black holes from each slice according to the completeness map.

    # Import the kernel density estimation function from scikit-learn
    from sklearn.neighbors import KernelDensity
    # Import the DD function from Corrfunc
    from Corrfunc.theory.DD import DD

    # Import the black hole coordinates from the file
    bh_coordinates = np.load(DATA_DIRECTORY + 'bh_coordinates_lightcone.npy')

    print("Number of black holes in the light cone:", bh_coordinates.shape[0])

    # Define the number of black holes to choose for the subset
    num_black_holes = 40000

    # randomly draw num_black_holes from the bh_coordinates
    random_indices = np.random.choice(bh_coordinates.shape[0], num_black_holes, replace=False)
    random_bh_coordinates = bh_coordinates[random_indices]

    # Get the z coordinates of the black holes
    z_coords = random_bh_coordinates[:, 2]

    # Get the x and y coordinates of the black holes
    x_coords = random_bh_coordinates[:, 0]
    y_coords = random_bh_coordinates[:, 1]

    # Define the subset of black holes to use for the correlation function
    bh_pos_x = x_coords
    bh_pos_y = y_coords
    bh_pos_z = z_coords

    # Define the slice size
    slice_size = 0.1  # h^-1 Mpc

    # Define the start and end redshifts
    start_z = 4
    end_z = 10

    # Convert the start and end redshifts to comoving distances
    start_comoving = cosmo.comoving_distance(start_z).value / h # h^-1 Mpc
    end_comoving = cosmo.comoving_distance(end_z).value / h

    # Calculate the number of slices
    num_slices = int((end_comoving - start_comoving) / slice_size)

    # Calculate the z coordinates of the slices
    slice_z_coords = np.linspace(start_comoving, end_comoving, num_slices + 1)

    # Calculate the center of the slices
    slice_centers = (slice_z_coords[:-1] + slice_z_coords[1:]) / 2

    # Assign each black hole to a slice based on its z coordinate
    slice_indices = np.digitize(z_coords, slice_z_coords[1:], right=True)

    # Choose a fraction of black holes from each slice based on the completeness map
    selected_bh_indices = []
    for i in range(1, num_slices + 1):
        slice_bh_indices = np.where(slice_indices == i)[0]
        num_bh_in_slice = len(slice_bh_indices)
        num_selected_bh = int(completeness_map[i - 1] * num_bh_in_slice)
        selected_slice_bh_indices = np.random.choice(slice_bh_indices, num_selected_bh, replace=False)
        selected_bh_indices.extend(selected_slice_bh_indices)

    # Get the selected black hole coordinates
    selected_bh_coordinates = random_bh_coordinates[selected_bh_indices]

    # Get the z coordinates of the selected black holes
    selected_z_coords = selected_bh_coordinates[:, 2]

    # Get the x and y coordinates of the selected black holes
    selected_x_coords = selected_bh_coordinates[:, 0]
    selected_y_coords = selected_bh_coordinates[:, 1]

    # Print the number of selected black holes
    print(f'Number of selected black holes: {len(selected_bh_coordinates)}')

    # Now I want to plot the distribution of the selected black holes in z direction and compare it with the full distribution.

    # # Create a histogram of the z coordinates of the selected black holes
    # plt.figure(figsize=(12, 8))
    # plt.hist(z_coords, bins=100, alpha=0.5, label='Full distribution')
    # plt.hist(selected_z_coords, bins=100, alpha=0.5, label='Selected distribution')
    # plt.xlabel('Z coordinate (h^-1 Mpc)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of black holes in the z direction')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Now I want to calculate the correlation function using the selected black holes and compare it with the correlation function using all black holes.
    # Full balck holes correlation function.

    min_x = 0
    max_x = 100 # h^-1 Mpc. Here x_coords, y_coords, z_coords are in h^-1 Mpc unlike the black holes coordinates directly from the MBII data.
    min_y = 0
    max_y = 100

    mult = 10 ## Number of random points used as a multiple of number of data points
    z_bin_size = 0.005   ## Bin size used to produce a new z distribution of the random numbers
    bin_size_ar = [0.5]  ## Bin size used for correlation function measurement

    for bin_size in bin_size_ar:

        n_D = len(bh_pos_x)
        n_rand = mult * n_D
        rand_x = np.random.uniform(min_x, max_x, n_rand)
        rand_y = np.random.uniform(min_y, max_y, n_rand)

        #-------------------------------------------------------------------------------------------
        #Drawing random numbers for z from smoothed distribution of the data z vals

        z_bin = np.arange(np.min(bh_pos_z), np.max(bh_pos_z), z_bin_size)[:,np.newaxis]
        z_bin_mid = (z_bin + z_bin_size/2)[:-1]
        kde = KernelDensity(kernel="gaussian", bandwidth=50).fit(bh_pos_z[:,np.newaxis])
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

        bins = np.arange(0.01, 100, bin_size)
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

        LandSzal2pcf = (DD_count/DD_norm - 2 * DR_count/DR_norm + RR_count/RR_norm)/(RR_count/RR_norm)

        pois_err = (1 + LandSzal2pcf)/np.sqrt(np.minimum(DD_count, n_D))

    from pandas import DataFrame

    df = DataFrame({"r min":bins[0:-1], "r max":bins[1:], "DD count":DD_count, "DR count":DR_count, "RR count": RR_count, "Landy Szalay":LandSzal2pcf, "Pois Error":pois_err})

    #Save the data to a file whose name is given by MBIILC_2pcf + .npy file name + .csv
    df.to_csv(DATA_DIRECTORY + "MBIILC_2pcf_" + npy_file + "_comp.csv", index=False)


    # Selected black holes correlation function.

    bh_pos_x = selected_x_coords
    bh_pos_y = selected_y_coords
    bh_pos_z = selected_z_coords

    min_x = 0
    max_x = 100 # h^-1 Mpc. Here x_coords, y_coords, z_coords are in h^-1 Mpc unlike the black holes coordinates directly from the MBII data.
    min_y = 0
    max_y = 100

    mult = 10 ## Number of random points used as a multiple of number of data points
    z_bin_size = 0.005   ## Bin size used to produce a new z distribution of the random numbers
    bin_size_ar = [0.5]  ## Bin size used for correlation function measurement

    for bin_size in bin_size_ar:

        n_D = len(bh_pos_x)
        n_rand = mult * n_D
        rand_x = np.random.uniform(min_x, max_x, n_rand)
        rand_y = np.random.uniform(min_y, max_y, n_rand)

        #-------------------------------------------------------------------------------------------
        #Drawing random numbers for z from smoothed distribution of the data z vals

        z_bin = np.arange(np.min(bh_pos_z), np.max(bh_pos_z), z_bin_size)[:,np.newaxis]
        z_bin_mid = (z_bin + z_bin_size/2)[:-1]
        kde = KernelDensity(kernel="gaussian", bandwidth=50).fit(bh_pos_z[:,np.newaxis])
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

        bins = np.arange(0.01, 100, bin_size)
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

        LandSzal2pcf = (DD_count/DD_norm - 2 * DR_count/DR_norm + RR_count/RR_norm)/(RR_count/RR_norm)

        pois_err = (1 + LandSzal2pcf)/np.sqrt(np.minimum(DD_count, n_D))

    from pandas import DataFrame

    df = DataFrame({"r min":bins[0:-1], "r max":bins[1:], "DD count":DD_count, "DR count":DR_count, "RR count": RR_count, "Landy Szalay":LandSzal2pcf, "Pois Error":pois_err})

    #Save the data to a file
    df.to_csv(DATA_DIRECTORY + "MBIILC_2pcf_" + npy_file + "_incomp.csv", index=False)

if __name__ == "__main__":
    npy_file = sys.argv[1]  # Get the .npy file from the command line arguments
    process_data(npy_file)
