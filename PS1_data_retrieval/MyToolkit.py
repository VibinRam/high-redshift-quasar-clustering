
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord



def range_wrapper(arr):

    '''Converts the range of (0 deg to 360 deg) to 
        (-180 deg to 180 deg) with 0 deg matching'''
        
    arr[np.where(arr > 180)] = arr[np.where(arr > 180)] - 360
    arr = -arr
    return arr

def plot_sky_map_ps1(ra, dec, mask, labels, title = "Default title"):

    '''Give right ascension in array of decimal degrees (0 deg to 360 deg)
        and declination in array of decimal degrees (-90 deg to 90 deg)'''

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


    plt.figure(figsize = (12, 7))
    ax = plt.subplot(111, projection = "mollweide")

    ### This plots the footprint
    plt.plot(gal_ra_1, gal_dec_1, color = 'black', linewidth = 1) 
    plt.plot(gal_ra_2, gal_dec_2, color = 'black', linewidth = 1)
    plt.plot(dec_30_ra, dec_30_dec, color = 'black', linewidth = 1)
    #-------------------------------------------------------------------------------------------
    ### Plotting the sources
    ra = np.radians(range_wrapper(ra)) 
    dec = np.radians(dec)
    marker = ['o','x']

    for n_source in range(mask[0]):
        source_pos = np.where(mask[1:] == n_source)
        ra_n, dec_n = ra[source_pos], dec[source_pos]

        plt.scatter(ra_n, dec_n, marker = marker[n_source], label = labels[n_source], s = 30)
    
    plt.title(title)
    plt.legend()
    plt.grid(True)

    ax.set_xticklabels(["10h", "8h", "6h", "4h", "2h", "0h", "22h", "20h", "18h", "16h", "14h"]);
    plt.show()


def healpix_sky_map_ps1_cover_area(ra, dec, level):
    
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

    hp.mollview(all_pixels)
    return [area_data_pixels, all_pixels]