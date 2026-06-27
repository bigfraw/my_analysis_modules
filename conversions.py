import numpy as np

'''
Functions for working with units
'''

def todB(x):
    """
    Convert a value to decibels.
    
    Parameters:
    x : float
        The value to convert.
    
    Returns:
    float
        The value in decibels.
    """
    return 10*np.log10(x)

def todBm(x):
    """
    Convert a value to decibels relative to 1 mW.
    
    Parameters:
    x : float
        The value to convert.
    
    Returns:
    float
        The value in decibels relative to 1 mW.
    """
    return 10*np.log10(x/1e-3)

def fromdB(x):
    """
    Convert a value from decibels.
    
    Parameters:
    x : float
        The value to convert.
    
    Returns:
    float
        The value in linear scale.
    """
    return 10**(x/10)

def fromdBm(x):
    """
    Convert a value from decibels relative to 1 mW.
    
    Parameters:
    x : float
        The value to convert.
    
    Returns:
    float
        The value in linear scale.
    """
    return 10**(x/10)*1e-3

def w0_to_div(w0,wavelength=1550e-9,):
    """
    Convert from gaussian waist radius to half-angle divergence

        Parameters:
        ang : float
            The waist radius of the Gaussian beam.
        
        Returns:
        float
            The half-angle divergence of the Gaussian beam.
    """
    return wavelength / (np.pi * w0)

def div_to_w0(div,wavelength=1550e-9):
    """
    Convert from half-angle divergence to gaussian waist radius.

        Parameters:
        div : float
            The half-angle divergence of the Gaussian beam.
        
        Returns:
        float
            The waist radius of the Gaussian beam.
    """
    return wavelength / (np.pi * div)

def arcsec_to_rad(ang):
    """
    Convert from arcseconds to radians.

        Parameters:
        ang : float
            The angle in arcseconds.
        
        Returns:
        float
            The angle in radians.
    """

    return ang * np.pi / 648000

def rad_to_arcsec(ang):
    """
    Convert from radians to arcseconds.

        Parameters:
        ang : float
            The angle in radians.
        
        Returns:
        float
            The angle in arcseconds.
    """
    return ang * 648000 / np.pi