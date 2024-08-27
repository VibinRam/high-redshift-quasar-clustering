# I want to make a plot of clustering length of quasars as a function of redshift that I have 
# collected from the literature. Each research paper will have measurement of clustering length
# at a particular redshift. We begin with collecting the data from the literature.

import numpy as np
import matplotlib.pyplot as plt
import re

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"

# Save the plot as a pdf file
PLOT_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/"

# Define the plot style
plt.style.use('MNRAS_Style.mplstyle')
# plt.grid(visible=False)

# Read the Clustering_Literature.csv file form the data directory
data = np.genfromtxt(DATA_DIRECTORY + 'Clustering_Literature.csv', delimiter=',', skip_header=1, dtype=None, encoding=None)

paper_name = data[:,0]
red_range = data[:,5]
clust_length = data[:,7]
clust_slope = data[:,8]

# Put an '\' before the '&' character in the paper_name
paper_name = [paper_name[i].replace('&', '\&') for i in range(len(paper_name))]

# convert the red_range which is in the form z_min - z_max to an array of z_min and z_max
red_range = np.array([red_range[i].split('-') for i in range(len(red_range))], dtype=float)

# convert clust_range which is in the form of r + delr_u/ -delr_d.
clust_length = [re.split("\+|/", clust_length[i]) for i in range(len(clust_length))]

# convert clust_slope which is in the form of game + delgam_u/ -delgam_d.
clust_slope = [re.split("\+|/", clust_slope[i]) for i in range(len(clust_slope))]

# Group the data points by the paper name
paper_name_unique = np.unique(paper_name)
paper_name_grouped = [np.where(np.array(paper_name) == paper_name_unique[i]) for i in range(len(paper_name_unique))]
clust_length_grouped = [np.array([clust_length[paper_name_grouped[i][0][j]] for j in range(len(paper_name_grouped[i][0]))], dtype=float) for i in range(len(paper_name_unique))]
clust_slope_grouped = [np.array([clust_slope[paper_name_grouped[i][0][j]] for j in range(len(paper_name_grouped[i][0]))], dtype=float) for i in range(len(paper_name_unique))]
red_range_grouped = [np.array([red_range[paper_name_grouped[i][0][j]] for j in range(len(paper_name_grouped[i][0]))], dtype=float) for i in range(len(paper_name_unique))]

# Plot the data
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for i in range(len(paper_name_unique)):
    ax[0].errorbar((red_range_grouped[i][:,0] + red_range_grouped[i][:,1])/2, clust_length_grouped[i][:,0], yerr=[clust_length_grouped[i][:,1], np.absolute(clust_length_grouped[i][:,2])], fmt='o', capsize=5, label=paper_name_unique[i])
    ax[1].errorbar((red_range_grouped[i][:,0] + red_range_grouped[i][:,1])/2, clust_slope_grouped[i][:,0], yerr=[clust_slope_grouped[i][:,1], np.absolute(clust_slope_grouped[i][:,2])], fmt='o', capsize=5, label=paper_name_unique[i])

ax[0].set_xlabel('Redshift')
ax[0].set_ylabel(r'Clustering Length $r_0$($h^{-1}$Mpc)')
ax[0].legend()
ax[0].set_xlim(0, 6)
ax[0].set_ylim(0, 30)
ax[0].grid(visible=False)

ax[1].set_xlabel('Redshift')
ax[1].set_ylabel(r'Clustering Slope $\gamma$')
ax[1].legend()
ax[1].set_xlim(0, 6)
ax[1].set_ylim(0.7, 2.6)
ax[1].grid(visible=False)

# save the plot as a pdf file
plt.tight_layout()
plt.savefig(PLOT_DIRECTORY + 'ClusteringLiterature.pdf')

plt.show()