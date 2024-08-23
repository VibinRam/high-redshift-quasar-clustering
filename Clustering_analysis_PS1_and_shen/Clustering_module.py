import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import LambdaCDM as LCDM
import numpy.ma as ma
from Corrfunc.theory.DD import DD
import scipy.optimize as spopt
from astropy.visualization import astropy_mpl_style
import healpy as hp

DP2_DIRECTORY = "/home/vibin/MyFolder/WorkDesk/DP2/"

cosmo = LCDM(H0=71, Om0=0.26, Ode0=0.74, Ob0=0.0435)

def range_wrapper(arr):

    '''Converts the range of (0 deg to 360 deg) to 
        (-180 deg to 180 deg) with 0 deg matching'''
        
    arr[np.where(arr > 180)] = arr[np.where(arr > 180)] - 360
    arr = -arr
    return arr

def plot_sky_map_ps1(ra, dec, labels = None, title = "Default title", mask = None, mark_size = 30):

    '''Give right ascension in array of decimal degrees (0 deg to 360 deg)
        and declination in array of decimal degrees (-90 deg to 90 deg)'''

    if (mask == None):
        mask = np.zeros_like(ra, dtype='int')
        mask = np.insert(mask, 0, 1)
        labels = ['Default label']

    ### Plots the lines marking the excluded area of the galactic plane
    gal_l = np.linspace(0, 360, 1000)
    gal_b = np.ones(1000) * 20.0
    gal_line_1 = SkyCoord(gal_l, gal_b, frame = 'galactic', unit = 'deg')
    gal_line_2 = SkyCoord(gal_l, -gal_b, frame = 'galactic', unit = 'deg')
    ### gal_line_equat is the galactic line in equatorial coordinates
    gal_line_equat_1 = gal_line_1.transform_to('icrs')   
    gal_line_equat_2 = gal_line_2.transform_to('icrs')
    gal_ra_1 = np.radians(range_wrapper(gal_line_equat_1.ra.degree))
    ###Sorting the array in order to do line plot
    i_sorted = np.argsort(gal_ra_1) 
    gal_ra_1 = gal_ra_1[i_sorted]
    gal_dec_1 = np.radians(gal_line_equat_1.dec.degree)[i_sorted]
    gal_ra_2 = np.radians(range_wrapper(gal_line_equat_2.ra.degree))
    i_sorted = np.argsort(gal_ra_2)
    gal_ra_2 = gal_ra_2[i_sorted]
    gal_dec_2 = np.radians(gal_line_equat_2.dec.degree)[i_sorted]
    ### dec -30 array is obtained
    dec_30_ra = np.radians(range_wrapper(np.linspace(0, 360, 1000)))
    dec_30_dec = np.radians(np.ones(1000) * -30.0)


    fig, ax = plt.subplots(figsize = (12, 7), subplot_kw={'projection': 'mollweide'})
    #ax = plt.subplot(111, projection = "mollweide")

    ### This plots the footprint
    ax.plot(gal_ra_1, gal_dec_1, color = 'black', linewidth = 1) 
    ax.plot(gal_ra_2, gal_dec_2, color = 'black', linewidth = 1)
    ax.plot(dec_30_ra, dec_30_dec, color = 'black', linewidth = 1)
    #-------------------------------------------------------------------------------------------
    ### Plotting the sources
    ra = np.radians(range_wrapper(ra)) 
    dec = np.radians(dec)
    marker = ['o','x']

    for n_source in range(mask[0]):
        source_pos = np.where(mask[1:] == n_source)
        ra_n, dec_n = ra[source_pos], dec[source_pos]

        ax.scatter(ra_n, dec_n, marker = marker[n_source], label = labels[n_source], s = mark_size)
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    ax.set_xticklabels(["10h", "8h", "6h", "4h", "2h", "0h", "22h", "20h", "18h", "16h", "14h"]);
    #plt.style.use(astropy_mpl_style)
    plt.show()

    return ax


def healpix_sky_map_ps1_cover_area(ra, dec, level, title = "Mollweide view"):
    
    '''Give the ra and dec arrays of the source in any numpy readable
    format. level- is the level of healpix pixelation required'''

    """Draws the healpix sky map with the pixels containing the data points
    highlighted. Returns the effective coverage area of the data.
    """

    nside = 2 ** level

    ra_values = np.radians(ra)
    dec_values = np.absolute(np.radians(dec) - np.pi/2)

    pixel_indices = hp.ang2pix(nside, dec_values, ra_values)

    npixels = hp.nside2npix(nside)
    all_pixels = np.zeros(npixels)
    all_pixels[pixel_indices] = 1

    pixels_with_data = np.shape(np.where(all_pixels != 0))[1]
    area_data_pixels = pixels_with_data * hp.nside2pixarea(nside, degrees=True)

    hp.mollview(all_pixels, title=title);
    return area_data_pixels, all_pixels



def find_pi_rp(ra_col, dec_col, red_col, rand_ra_col = None, rand_dec_col = None, rand_red_col = None):
    '''Finds r_p and pi from two catalogs given as three arrays of ra (dec deg), dec (dec deg) and redshift. If only 
    1 catalog is given, finds the pairs between that catalog and itself
    '''
    if((rand_ra_col is None) and (rand_dec_col is None) and (rand_red_col is None)):
        rand_ra_col = ra_col
        rand_dec_col = dec_col
        rand_red_col = red_col

    dist_sq1 = np.square(np.array(cosmo.comoving_distance(red_col)))
    dist_sq2 = np.square(np.array(cosmo.comoving_distance(rand_red_col)))
    v1_sq, v2_sq = np.meshgrid(dist_sq1, dist_sq2)

    ps1_quasars1 = SkyCoord(ra_col*u.deg, dec_col*u.deg, cosmo.comoving_distance(red_col))
    ps1_quasars2 = SkyCoord(rand_ra_col*u.deg, rand_dec_col*u.deg, cosmo.comoving_distance(rand_red_col))
    ps1_quasars1.representation_type = 'cartesian'
    ps1_quasars2.representation_type = 'cartesian'
    
    x1 = np.array(ps1_quasars1.x)
    y1 = np.array(ps1_quasars1.y)
    z1 = np.array(ps1_quasars1.z)
    x2 = np.array(ps1_quasars2.x)
    y2 = np.array(ps1_quasars2.y)
    z2 = np.array(ps1_quasars2.z)    
    x1, x2 = np.meshgrid(x1, x2)
    y1, y2 = np.meshgrid(y1, y2)
    z1, z2 = np.meshgrid(z1, z2)

    v1_v2 = x1*x2 + y1*y2 + z1*z2

    pi_array = np.absolute((v2_sq - v1_sq)/np.sqrt(v1_sq + v2_sq + 2 * v1_v2))
    rp_array = np.sqrt(np.absolute(v1_sq + v2_sq - 2 * v1_v2 - np.square(pi_array)))

    pi_array = pi_array[np.triu_indices(len(pi_array),k = 1)]
    rp_array = rp_array[np.triu_indices(len(rp_array),k = 1)]

    return pi_array, rp_array
    #return v1_sq, v2_sq, v1_v2


def find_s_bined(ra_col, dec_col, red_col, rand_ra_col = None, rand_dec_col = None, rand_red_col = None):

    self_pair = False
    if((rand_ra_col is None) and (rand_dec_col is None) and (rand_red_col is None)):
        rand_ra_col = ra_col
        rand_dec_col = dec_col
        rand_red_col = red_col

        self_pair = True

    ps1_quasars1 = SkyCoord(ra_col*u.deg, dec_col*u.deg, cosmo.comoving_distance(red_col) * 0.71)
    ps1_quasars2 = SkyCoord(rand_ra_col*u.deg, rand_dec_col*u.deg, cosmo.comoving_distance(rand_red_col) * 0.71)
    ps1_quasars1.representation_type = 'cartesian'
    ps1_quasars2.representation_type = 'cartesian'
    
    x1 = np.array(ps1_quasars1.x)
    y1 = np.array(ps1_quasars1.y)
    z1 = np.array(ps1_quasars1.z)
    x2 = np.array(ps1_quasars2.x)
    y2 = np.array(ps1_quasars2.y)
    z2 = np.array(ps1_quasars2.z)
    x1, x2 = np.meshgrid(x1, x2)
    y1, y2 = np.meshgrid(y1, y2)
    z1, z2 = np.meshgrid(z1, z2)

    s_array = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2) + np.power(z2 - z1, 2))

    if(self_pair == True):
        s_array = s_array[np.triu_indices(len(s_array),k = 1)]
    else:
        s_array = s_array.flatten()

    return s_array
    
def find_s_binned_shadab(ra_col, dec_col, red_col, rand_ra_col = None, rand_dec_col = None, rand_red_col = None):

    if((rand_ra_col is None) and (rand_dec_col is None) and (rand_red_col is None)):
        rand_ra_col = ra_col
        rand_dec_col = dec_col
        rand_red_col = red_col

    theta = np.pi*(90 - dec_col)/180
    phi = np.pi * ra_col / 180

    rco = cosmo.comoving_distance(red_col).value * 0.71

    x = rco * np.sin(theta) * np.cos(phi)
    y = rco * np.sin(theta) * np.sin(phi)
    z = rco * np.cos(theta)

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    z1, z2 = np.meshgrid(z, z)

    s_array = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2) + np.power(z2 - z1, 2))
    s_array = s_array[np.triu_indices(len(s_array),k = 1)]

    return s_array

# def make_rand_from_dist(red_col, n_samples):
#     bins = [2.901 + i*0.05 for i in range(50)]
#     hist, edges = np.histogram(red_col, bins=bins)
#     bin_widths = np.diff(edges)
#     cdf = np.cumsum(hist * bin_widths) / np.sum(hist * bin_widths)
#     #plt.stairs(cdf, bins)

#     # Generate new z values that follow the histogram distribution
#     uniform_values = np.random.rand(n_samples)
#     bin_indices = np.searchsorted(cdf, uniform_values)
#     bin_edges = edges[bin_indices-1]
#     bin_diff = edges[bin_indices] - edges[bin_indices-1]
#     bin_weights = (uniform_values - cdf[bin_indices-1]) / (cdf[bin_indices] - cdf[bin_indices-1])

#     new_z_values = bin_edges + bin_weights * bin_diff

#     return new_z_values

def find_wp_rp_single_bin(ra, dec, red, file_name):
    rand_ra, rand_dec, rand_red = make_rand_cat_ps1(len(ra))

    pi_array, rp_array = find_pi_rp(ra, dec, red)
    rand_pi_array, rand_rp_array = find_pi_rp(rand_ra, rand_dec, rand_red)
    cross_pi_array, cross_rp_array = find_pi_rp(ra, dec, red, rand_ra_col=rand_ra,\
                                                rand_dec_col=rand_dec, rand_red_col=rand_red)

    rp_bins = np.float_power(10, np.arange(31) * 0.15)
    rp_mid = (rp_bins[:-1] + rp_bins[1:])/2

    rp_hist = (plt.hist(rp_array, rp_bins)[0]).astype(int)
    rand_rp_hist = (plt.hist(rand_rp_array, rp_bins)[0]).astype(int)
    cross_rp_hist = (plt.hist(cross_rp_array, rp_bins)[0]).astype(int)
    plt.close()

    unfin_pos = np.where(rp_hist * rand_rp_hist * cross_rp_hist == 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        xi_rp = (rp_hist - 2 * cross_rp_hist + rand_rp_hist)/rand_rp_hist
    xi_rp_masked = ma.array(xi_rp)
    xi_rp_masked[unfin_pos] = ma.masked

    with np.errstate(divide='ignore', invalid='ignore'):
        xi_rp_error = (1 + np.absolute(xi_rp))/np.sqrt(np.minimum(rp_hist, len(ra)))
    xi_rp_error_masked = ma.array(xi_rp_error)
    xi_rp_error_masked[unfin_pos] = ma.masked

    file = open(DP2_DIRECTORY + "Data/" + file_name + ".txt", 'w')
    file.write('   rp_mid       DD      RR      DR      xi_rp \n')
    for i in range(len(rp_mid)):
        file.write(f'{rp_mid[i]:9.3f}   {rp_hist[i]:5d}    {rand_rp_hist[i]:5d}   {cross_rp_hist[i]:5d}   {xi_rp[i]:8.5f}   {xi_rp_error[i]:8.5f}\n')

    #xi_rp = np.abs(xi_rp)
    #(rp_hist + rand_rp_hist - 2 * cross_rp_hist)/rand_rp_hist
    fig, ax = plt.subplots()
    ax.errorbar(rp_mid, xi_rp_masked/rp_mid, xi_rp_error_masked/rp_mid, fmt = '*--')
    ax.set_ylabel(r'$w_p(r_p)/r_p$')
    ax.set_xlabel(r'$r_p (Mpc)$')
    ax.axhline(0, ls = '--', lw = 0.5, c = 'black')

    return ax

###------------------------------------------------------------------------------------------------------------------------------------------------------------
### All the routines which are needed to find the redshift space clustering of quasars in PS1 footprint


def make_rand_from_dist_any(red_col, bins, n_samples):            #Make a random z distribution from the data z distribution
    hist, edges = np.histogram(red_col, bins=bins)
    bin_widths = np.diff(edges)
    cdf = np.cumsum(hist * bin_widths) / np.sum(hist * bin_widths)
    cdf = np.insert(cdf, 0, 0)
    #plt.stairs(cdf, bins)

    # Generate new z values that follow the histogram distribution
    uniform_values = np.random.rand(n_samples)
    bin_indices = np.searchsorted(cdf, uniform_values)
    bin_edges = edges[bin_indices-1]
    bin_diff = edges[bin_indices] - edges[bin_indices-1]
    bin_weights = (uniform_values - cdf[bin_indices-1]) / (cdf[bin_indices] - cdf[bin_indices-1])

    new_z_values = bin_edges + bin_weights * bin_diff

    ##---------------------------------------------
    ## Making changes to see how clustering changes
    #new_z_values = np.random.uniform(2.9, 3.5, n_samples)
    #new_z_values[300:] = new_z_values[300:] * 1.01
    ##

    return new_z_values

def make_rand_cat_ps1(mult, red_col):   #Make random catalog in the PS1 footprint
    num = mult * len(red_col)

    ind = np.where(np.arange(num) > -1)
    rand_ra = np.zeros(num)
    rand_dec = np.zeros(num)
    while True:
        ### using Archimede's theorem
        rand_ra[ind] = np.random.uniform(0, 360, len(ind[0]))
        rand_dec[ind] = np.degrees(np.arcsin(np.random.uniform(-0.5, 1, len(ind[0]))))
        ind = np.where((np.absolute(np.arcsin(np.cos(np.radians(rand_dec)) * np.cos(np.radians(27.4)) * np.cos(np.radians(rand_ra - 192.25)) \
                                            + np.sin(np.radians(rand_dec)) * np.sin(np.radians(27.4)))) < np.radians(20)) | ((rand_ra > 7) & (rand_ra < 14)\
                                                & (rand_dec > 37) & (rand_dec < 43)))

        if(len(ind[0]) == 0):
            break

    step = 0.05
    bins = np.arange(np.min(red_col), np.max(red_col) + step, step)
    rand_red = make_rand_from_dist_any(red_col, bins, num)

    return rand_ra, rand_dec, rand_red

def pair_count_corrfunc(ra_col, dec_col, red_col, bins, rand_ra_col = None, rand_dec_col = None, rand_red_col = None):
    #Pair counting routine

    np.set_printoptions(suppress=True)

    autocorr = 0

    quasar_cord = SkyCoord(ra_col*u.deg, dec_col*u.deg, cosmo.comoving_distance(red_col) * 0.71)
    quasar_cord.representation_type = 'cartesian'
    x1 = np.array(quasar_cord.x)
    y1 = np.array(quasar_cord.y)
    z1 = np.array(quasar_cord.z)

    if((rand_ra_col is None) and (rand_dec_col is None) and (rand_red_col is None)):
        autocorr = 1

        results_DD = DD(autocorr, nthreads=1, binfile=bins, X1=x1, Y1=y1, Z1=z1, periodic=False)
        results_DD = np.array(list(map(list, results_DD)))
        results_DD[:, 3] = results_DD[:, 3]/2

        return results_DD

    quasar_cord2 = SkyCoord(rand_ra_col*u.deg, rand_dec_col*u.deg, cosmo.comoving_distance(rand_red_col) * 0.71)
    quasar_cord2.representation_type = 'cartesian'
    x2 = np.array(quasar_cord2.x)
    y2 = np.array(quasar_cord2.y)
    z2 = np.array(quasar_cord2.z)

    results_DD = DD(autocorr, nthreads=1, binfile=bins, X1=x1, Y1=y1, Z1=z1, X2=x2, Y2=y2, Z2=z2, periodic=False)
    results_DD = np.array(list(map(list, results_DD)))
    
    return results_DD

def find_xi_s(ra, dec, red, s_bins, file_name=None, rand_ra = None, rand_dec = None, rand_red = None, draw_ax = None, extra=False, fit=False, ret_result=False):
    #Find the redshift space clustering

    if((rand_ra is None) and (rand_dec is None) and (rand_red is None)):
        raise ValueError

    # s_array = find_s_bined(ra, dec, red)
    # rand_s_array = find_s_bined(rand_ra, rand_dec, rand_red)
    # cross_s_array = find_s_bined(ra, dec ,red, rand_ra_col=rand_ra, rand_dec_col=rand_dec, rand_red_col=rand_red)

    s_mid = (s_bins[:-1] + s_bins[1:])/2

    # s_hist = (np.histogram(s_array, s_bins)[0]).astype(int)
    # rand_s_hist = (np.histogram(rand_s_array, s_bins)[0]).astype(int)
    # cross_s_hist = (np.histogram(cross_s_array, s_bins)[0]).astype(int)

    s_hist = (pair_count_corrfunc(ra, dec, red, s_bins)[:,3]).astype(int)
    rand_s_hist = (pair_count_corrfunc(rand_ra, rand_dec, rand_red, s_bins)[:,3]).astype(int)
    cross_s_hist = (pair_count_corrfunc(ra, dec, red, s_bins, rand_ra_col=rand_ra, rand_dec_col=rand_dec, rand_red_col=rand_red)[:,3]).astype(int)
    s_hist_norm = (len(ra) * (len(ra) - 1))/2
    rand_hist_norm = (len(rand_ra) * (len(rand_ra) - 1))/2
    cross_hist_norm = len(ra) * len(rand_ra)

    unfin_pos = np.where(s_hist * rand_s_hist * cross_s_hist == 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        xi_s = (s_hist/s_hist_norm - 2 * cross_s_hist/cross_hist_norm + rand_s_hist/rand_hist_norm)/(rand_s_hist/rand_hist_norm)
    xi_s_masked = ma.array(xi_s)
    xi_s_masked[unfin_pos] = ma.masked

    with np.errstate(divide='ignore', invalid='ignore'):
        xi_s_error = (1 + xi_s)/np.sqrt(np.minimum(s_hist, len(ra)))
    xi_s_masked_error = ma.array(xi_s_error)
    xi_s_masked_error[unfin_pos] = ma.masked

    if file_name!=None:
        file = open(DP2_DIRECTORY + "Data/" + file_name + ".txt", 'w')
        file.write('  s_mid       DD      RR      DR      xi_s     Delta xi_s\n')
        for i in range(len(s_mid)):
            file.write(f'{s_mid[i]:9.3f}   {s_hist[i]:5d}    {rand_s_hist[i]:5d}   {cross_s_hist[i]:5d}   {xi_s_masked[i]:8.5f}   {xi_s_masked_error[i]:8.5f}\n')
        file.close()

    if draw_ax is not None:
        fig = None
        ax = draw_ax
    elif extra:
        fig, (ax, ax2) = plt.subplots(1,2, figsize= (15,5))
        ax2.errorbar(s_mid, xi_s_masked, yerr = xi_s_masked_error, fmt='sb', ms=5, label="This Work")
        ax2.set_ylabel(r'$\xi(s)$')
        ax2.set_xlabel(r'$s (h^{-1}Mpc)$')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(0.001, 100)
    else:
        fig, ax = plt.subplots()
    ax.errorbar(s_mid, xi_s_masked, yerr = xi_s_masked_error, fmt='sb', capsize=5, ms=5, label="This Work")
    ax.set_ylabel(r'$\xi(s)$')
    ax.set_xlabel(r'$s (h^{-1}Mpc)$')
    ax.axhline(0, ls = '--', lw = 0.5, c = 'black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.001, 100)

    if fit:
        def pow_fun(s, s0, delta):
            return np.power(s/s0, -delta)
        pow_fun_vec = np.vectorize(pow_fun)
        s_val_to_fit = s_mid[~xi_s_masked.mask]
        xi_val_to_fit = np.array(xi_s_masked[~xi_s_masked.mask])
        loca = np.where(xi_val_to_fit > 0)
        pars, cov = spopt.curve_fit(pow_fun, s_val_to_fit[loca], xi_val_to_fit[loca], p0=[1,1])
        print(xi_val_to_fit[loca])
        
        fit_values = pow_fun(s_mid, pars[0], pars[1])
        ax2.plot(s_mid, fit_values, '--', label="Power law fit")
        # ax.plot(s_mid, fit_values, '--')

        ax2.text(1000, 10, r"$s_0 = $"+str(np.round(pars[0],2)), c='red', fontsize=14)
        ax2.text(1000, 4, r"$\delta = $"+str(np.round(pars[1],2)), c='red', fontsize=14)

    if extra:
        return fig, (ax, ax2)
    
    if ret_result:
        return s_mid, xi_s_masked, xi_s_masked_error, s_hist, rand_s_hist, cross_s_hist

    return fig, ax


