# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM

# Define the data directory
DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data/MBIIbhIncompOverlf/"
# define the Plot directory
PLOT_DIRECTORY = '/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Plots/'

# Define the plot style
plt.style.use('MNRAS_Style.mplstyle')

# define the cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
h = 0.7

# Define the continuum slope of the quasar spectrum
alpha = -0.61

# Read Richards et al. 2006 selection function data in file Richards2006SelectionFunc.txt in data directory.
Richards2006SelectionFunc = ascii.read(DATA_DIRECTORY + 'Richards2006SelectionFunc.dat')

def M_1450(m_i, z):
    return m_i - 5 * np.log10(cosmo.luminosity_distance(z).value) - 25 - K_i_1450(z)

def K_i_1450(z):
    return (alpha - 1) * 2.5 * np.log10(1 + z) + (alpha - 2) * 2.5 * np.log10(7700/1450)

# Add the M_1450 column to the data table
Richards2006SelectionFunc['M_1450'] = M_1450(Richards2006SelectionFunc['imag'], Richards2006SelectionFunc['z'])

red_array = np.linspace(0, 10, 1000)

# Plot K_i_1450 vs z
plt.figure(figsize=(6, 4))
plt.plot(red_array, K_i_1450(red_array), 'k-')
plt.xlabel('Redshift')
plt.ylabel('$K_{m_i, 1450}$')
plt.grid(visible=False)
plt.gca().set_box_aspect(1)
plt.show()
