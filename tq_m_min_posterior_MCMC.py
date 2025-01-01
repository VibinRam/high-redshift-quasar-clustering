# Use MCMC to get the best fit parameters for the integrated QLF = quasar_density_hm 
import numpy as np
import emcee
import matplotlib.pyplot as plt
from Halomass_duty import quasar_density_halo_model
import sys
from astropy.cosmology import LambdaCDM
from colossus.cosmology import cosmology

# Define the cosmology of Richards et al. 2006
cosmo_richards = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
# h = cosmo_richards.h

cosmo_shen = cosmology.setCosmology('WMAP3')

DATA_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/PhdProjects/Complicor/Data2.0/"

richards_int_QLF =  np.load(DATA_DIRECTORY + "richards_int_QLF_20.2.npy")

z_values = np.arange(2.9, 5.3, 0.1)

# Define the log-likelihood function
def log_likelihood(params, z_values, observed_Phi, sigma_Phi):
    t_q, M_min = params
    # Calculate the model prediction using the quasar_density_halo_model
    model_Phi = np.array([quasar_density_halo_model(z, t_q, M_min) for z in z_values])

    com_vol_ratio = (cosmo_shen.angularDiameterDistance(z_values)**2 / cosmo_richards.angular_diameter_distance(z_values).value**2) * \
                    (cosmo_richards.H(z_values).value / cosmo_shen.Hz(z_values))
    
    model_Phi = model_Phi * com_vol_ratio
    
    # Calculate the log-likelihood based on Gaussian errors
    return -0.5 * np.sum(((observed_Phi - model_Phi) / sigma_Phi) ** 2)

# Define the log-prior function
def log_prior(params):
    t_q, M_min = params
    # Example priors: log-uniform prior for t_q and uniform for M_min
    if 0.1 < t_q < 10 and 1e10 < M_min < 1e14:
        return 0.0
    return -np.inf

# Define the log-posterior function
def log_posterior(params, z_values, observed_Phi, sigma_Phi):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, z_values, observed_Phi, sigma_Phi)

# Prepare MCMC function
def run_single_chain(walker_index, z_values, observed_Phi, sigma_Phi, n_steps=1000):
    # Initial guesses for [t_q, M_min]
    initial_guess = [1, 1e12]  # starting values

    # Initialize walkers with small random noise around initial guess
    pos = initial_guess + 1e-1 * np.random.randn(len(initial_guess))

    # Set up the sampler
    sampler = emcee.EnsembleSampler(1, len(initial_guess), log_posterior, args=(z_values, observed_Phi, sigma_Phi))

    # Run the MCMC chains
    sampler.run_mcmc(pos, n_steps, progress=False)

    # Save the mcmc chain
    np.save(f"chain_{walker_index}.npy", sampler.get_chain(discard=100, thin=15, flat=True))

observed_Phi = richards_int_QLF
sigma_Phi = 1e-9

# Extract command line arguments
walker_index = int(sys.argv[1])
n_steps = int(sys.argv[2])

# Run the MCMC procedure
run_single_chain(walker_index, z_values, observed_Phi, sigma_Phi, n_steps)

