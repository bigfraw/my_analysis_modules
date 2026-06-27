import numpy as np
from LightPipes import *
from my_analysis_modules.general_atmospherics import overlap, power

'''
Atmospheric analysis functions using LightPipes.
'''

def smf(grid_length, wavelength, npix, d):
    '''
    Calculate the back propagated fibre mode for single mode fibre

    Inputs:
    grid_length : float
        The length of the grid in metres
    wavelength : float
        The wavelength of the light in metres
    npix : int
        The number of pixels in the grid
    d : float
        The aperture diameter in metres

    Returns:
    field : numpy.ndarray
        The field of the back propagated fibre mode
    '''
    
    w0 = 2.24/d
    
    w = Begin(grid_length, wavelength, npix)
    w = GaussBeam(w, (1/w0)*m, LG=True, n=0, m=0)
    
    w = SubIntensity(w, Intensity(w)/np.sum(Intensity(w)))
    
    return w.field

def coupling_efficiency(fields, D, grid_length, wavelength=1550e-9, mask = None):

    """
    Calculate the coupling efficiency of a field or series of fields into a fibre.

    Inputs:
        fields : list
            The list of fields to calculate the coupling efficiency of
        D : float
            The aperture diameter in metres
        n_pix : int
            The number of pixels in the grid
        grid_length : float
            The length of the grid in metres
        wavelength : float
            The wavelength of the light in metres
        mask : numpy.ndarray
            The mask of the aperture
    
    Returns:
        coupling_efficiency : float
            The coupling efficiency
        coupled_powers : numpy.ndarray
            The coupled powers of each field
    """

    N = fields[0].shape[0]
    fibre_field = smf(grid_length, wavelength, N, D)
    numerator = []
    denominator = []

    for field in fields:
        if mask is not None:
            field = np.multiply(field, mask)
        numerator.append(np.abs(overlap(field, fibre_field))**2)
        # denominator.append(overlap(field, field))
        denominator.append(power(field))

    return np.abs(sum(numerator) / sum(denominator)), np.asarray(numerator)