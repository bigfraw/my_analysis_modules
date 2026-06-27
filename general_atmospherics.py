import sys
sys.path.insert(0, sys.path[0] + '\\..')

print(sys.path[0])

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import scipy.integrate as integrate
# import cv2
from tqdm import tqdm
import math
from array_calcs import circle, correlate2D
from aotools.turbulence import cn2_to_r0, r0_to_cn2
from fields import gaussz, beamparam

'''
Functions for general atmospheric simulations.

NO LIGHTPIPES DEPENDENCY
'''

rytov = lambda cn2,k,l: (1.23*cn2*k**(7/6)*l**(11/6))**0.5
scint_pl = lambda cn2,k,l: np.exp(0.54*rytov(cn2,k,l)**2/(1+1.22*rytov(cn2,k,l)**(12/5))**(7/6) + 0.509*rytov(cn2,k,l)**2/(1+0.69*rytov(cn2,k,l)**(12/5))**(5/6))-1
r0_pl = lambda cn2,k,l: (0.423*k**2*cn2*l)**(-3/5)  

def v_wind(h, ws=1, Vg=10): 
    """
    Wind speed profile as a function of height above ground.

    Parameters:
        h (numpy.ndarray) :
            Height above ground, metres.
        ws (float) :
            Satellite slew rate, deg/s.
        Vg (float) :
            Ground wind speed, m/s.

    :return vs (numpy.ndarray): the wind speed profile, m/s.
    """

    vs = np.deg2rad(ws)*h + Vg + 30*np.exp(-((h-9400)/4800)**2)
    return vs

def get_c2n(height, wind_rms = 21, c2n_0 = 1.7e-14):
    """Gets the characteristic optical turbulence value (Cn^2) for each height step for each time step. Follows Hufnagel-Valley (H-V) model.
    The formal term for Cn^2 is the index of refraction structure constant, sometimes called the structure parameter.

    Args:
        wind_rms (arr): rms wind speed
        height (arr): 2D numpy array of height values at each height step for each time step (in meters). 
        cn2_0 (float): Cn^2 value at ground level (default is 1.7e-14 m^(-2/3)).

    Returns:
        c2n (arr): 2D array of optical turbulence value for each height step for each time step (in meters^(-2/3)).
    """
    c2ns = 0.00594*np.power(wind_rms/27, 2) * np.power(1e-5*height, 10) * np.exp(-height/1000) + \
            2.7e-16*np.exp(-height/1500) + \
            c2n_0*np.exp(-height/100)
    
    return c2ns

def cum_cn2(cn2, heights, height_i = None, height_f = None):

    if height_i is None:
        height_i = heights[0]
    if height_f is None:
        height_f = heights[-1]

    if height_i not in heights:
        # find closest value in heights array
        height_i = heights[np.abs(heights - height_i).argmin()]
        # print("Warning: height_i not in heights array, using closest value:", height_i)
    if height_f not in heights:
        height_f = heights[np.abs(heights - height_f).argmin()]
        # print("Warning: height_f not in heights array, using closest value:", height_f)

    cond = (heights >= height_i) & (heights <= height_f)

    return np.trapz(cn2[cond], heights[cond])

    # return integrate.quad(get_c2n, height_i, height_f)[0]

def spotsizeLT(cn2s, heights, height_f, w0, labda=1550e-9):
    '''
    Calculate the long term spot size assuming WEAK FLUCTUATIONS

    Parameters:
        cn2s (numpy.ndarray) :
            Turbulence strength profile.
        heights (numpy.ndarray) :
            Height profile.
        height_f (float) :
            Final height.
        w0 (float) :
            Beam waist.
        labda (float) :
            Wavelength.

    Returns:
        Long term spot size at plane
    '''

    Cn2_net = cum_cn2(cn2s, heights, height_f=height_f)
    rytovvar = rytov(Cn2_net, 2*np.pi/labda, height_f)**2
    beamparamval = beamparam(height_f, w0)
    W = gaussz(w0, height_f, labda)

    return W*np.sqrt(1+1.33*rytovvar*beamparamval**(5/6))

def wLT(cn2s, heights, height_f = None, cumulative = True, labda = 1550e-9):
    '''
    SHANES CODE!!!

    Calculate the long term spot size.

    NOTE: i need to adjust this to handle integrated cn2 as the input

    Parameters:
        cn2s (numpy.ndarray) : 
            Turbulence strength profile in m^(-2/3).
        heights (numpy.ndarray) :
            Height profile in meters.
        height_f (float) :
            Target height, if None will calculate for all heights.
        labda (float) :
            Wavelength.

    Returns:
        w_LTs (numpy.ndarray) :
            Long term spot size in meters.

    '''

    k = 2*np.pi/labda
    Theta_0 = 1

    # if no heights_f, calculate wLT for every height in heights

    if height_f is None:
        height_f = heights
    elif isinstance(height_f, float) or isinstance(height_f, int):
        height_f = np.array([height_f])
    elif isinstance(height_f, list):
        height_f = np.array(height_f)

    w_LTs = np.zeros_like(height_f)

    for i,height in enumerate(height_f):

    # lambda, theta parameters from Andrews and Phillips

        Lambda_0 = 2*height/(k*0.125**2)
        Lambda = Lambda_0/(Theta_0**2+Lambda_0**2)

        W = np.sqrt(2*height/(Lambda*k)) # diff limited beam size

        if cumulative:
            r0 = cn2_to_r0(cum_cn2(cn2s,heights,height_f = height), labda)
        else:
            r0 = cn2_to_r0(cn2s[i], labda)

        w_LTs[i] = W*(1+(0.35/r0)**(5/3))**(3/5)

    return w_LTs

def subtilt(phases, mask=None):
    """
    Removes tilt over non-zero pixels of datacube.
    If mask is provided, removes tilt over non-zero pixels of mask.
    Removes tilt from each slice of a cube.

    Parameters:
    phases (numpy.ndarray): Input data cube (2D or 3D array).
    mask (numpy.ndarray, optional): Mask to determine non-zero pixels.

    Returns:
    numpy.ndarray: Data cube with tilt removed.
    numpy.ndarray: Tilt

    """
    if phases.ndim == 2:
        phases = phases[..., np.newaxis]  # Add an extra dimension for consistency
        single_frame = True
    else:
        single_frame = False
       
    nframe = phases.shape[-1]
    notilt = np.zeros_like(phases)

    if mask is not None:
        pmask = mask
    else:
        pmask = np.ones_like(phases[:,:,0])

    pmask[pmask != 0] = 1

    planes = np.zeros_like(phases)
    planes_components = np.zeros((3,phases.shape[-1]))

    for k in range(nframe):
        p = phases[:,:,k].astype(float)
        y, x = np.indices(p.shape)
        x = x[pmask > 0]
        y = y[pmask > 0]
        z = p[pmask > 0]

        a, b, c = planefit(x, y, z)
        plane = a * np.indices(p.shape)[1] + b * np.indices(p.shape)[0] + c
        notilt[:,:,k] = p - plane
        planes[:,:,k] = plane
        planes_components[:,k] = [a,b,c]

    if single_frame:
        return notilt[:,:,0], planes[:,:,0], planes_components[:,0]  # Return a 2D array if input was 2D
    
    return notilt, planes, planes_components

def planefit(x, y, z):
    """
    Fit a plane to the function defined by vectors x, y, z
    Fit is given by z = f[0]*x + f[1]*y + f[2]

    x, y, z are meshgrids
    """

    # fit a plane to the data
    xx = x.flatten()
    yy = y.flatten()
    zz = z.flatten()

    A = np.c_[xx,yy,np.ones(xx.shape)]
    p, _, _, s = np.linalg.lstsq(A, zz, rcond = None) # NOTE: more stringent rcond??

    return p

def intensity(f):
    """
    Calculate the intensity of a field f.
    
    Parameters:
    f : numpy.ndarray
        The input field.
    
    Returns:
    intensity : numpy.ndarray
        The intensity of the field.
    """
    return np.abs(f)**2

def power(f, area = None):
    """
    Calculate the power of a field f.
    
    Parameters:f
    f : numpy.ndarray
        The input field.

    area : float, optional
        The area the field is defined over, default is None.
    
    Returns:
    power : float
        The power of the field.
    """

    if area is not None:
        return np.sum(intensity(f)) * area
    return np.sum(intensity(f))

def phase(f):
    """
    Calculate the phase of a field f.
    
    Parameters:
    f : numpy.ndarray
        The input field.
    
    Returns:
    phase : numpy.ndarray
        The phase of the field.
    """
    return np.angle(f)

def embed(f, n=1):
    """
    Embed a 2D or 3D array f in the center of a raster that's bigger by 2^n.
    n=1 if unspecified.
    
    Parameters:
    f : numpy.ndarray
        The input array to embed.
    n : int, optional
        The scale factor to increase the size by, default is 1.
    
    Returns:
    e : numpy.ndarray
        The embedded array.
    """
    # Calculate the size of the new embedded array
    s = np.round(2**n * np.array(f.shape[:2])).astype(int)
    if f.ndim == 3:
        e = np.zeros((s[0], s[1], f.shape[2]), dtype=f.dtype)
    else:
        e = np.zeros(s, dtype=f.dtype)
    
    # Calculate offsets to center the original array in the new larger array
    off1 = np.ceil(s[0] * (2**n - 1) / (2**(n + 1))).astype(int)
    off2 = np.ceil(s[1] * (2**n - 1) / (2**(n + 1))).astype(int)
    
    # Embed the original array in the center of the new larger array
    if f.ndim == 3:
        e[off1:off1+f.shape[0], off2:off2+f.shape[1], :] = f
    else:
        e[off1:off1+f.shape[0], off2:off2+f.shape[1]] = f
    
    return e

def unembed(f, n=1):
    """
    Extract the center piece of a 2D or 3D array f into a raster half the size.
    Undoes 'embed'.
    
    Parameters:
    f : numpy.ndarray
        The input array to unembed.
    n : int, optional
        The scale factor to reduce the size by, default is 1.
    
    Returns:
    e : numpy.ndarray
        The unembedded array.
    """
    s = np.round((2**-n) * np.array(f.shape[:2])).astype(int)
    e = np.zeros((s[0], s[1], f.shape[2]), dtype=f.dtype) if f.ndim == 3 else np.zeros(s, dtype=f.dtype)
    off1 = np.ceil(f.shape[0] * (2**n - 1) / (2**(n + 1))).astype(int)
    off2 = np.ceil(f.shape[1] * (2**n - 1) / (2**(n + 1))).astype(int)
    
    if f.ndim == 3:
        e = f[off1:off1+s[0], off2:off2+s[1], :]
    else:
        e = f[off1:off1+s[0], off2:off2+s[1]]
    
    return e

def overlap(field1, field2):
    """
    Calculate the coherence function and overlap integral between two fields.

    Inputs:
    field1 : numpy.ndarray
        The first field
    field2 : numpy.ndarray
        The second field

    Returns:
    field_corr[0][0] : float
        The overlap integral
    field_corr : numpy.ndarray
        The coherence function
    """

    field1_fft = np.fft.fft2(field1)
    field2_fft = np.fft.fft2(field2)
    field_corr = np.fft.ifft2(np.multiply(field1_fft, np.conj(field2_fft)))

    field_corr_shifted = np.fft.fftshift(field_corr)

    return field_corr[0][0], field_corr_shifted

def mcf(field):

    '''
    Compute the mutual coherence function of a field.

    Inputs:
    field : numpy.ndarray
        The input field.

    Returns:
    field_corr : numpy.ndarray
        The mutual coherence function.
    '''

    field_corr = np.real(fftshift(ifft2(np.multiply(fft2(field), np.conj(fft2(field))))))

    return field_corr

def find_blob_radius(image):

    '''
    Find the radius of a blob in a thresholded image. Used for finding the coherence length.

    Inputs:
    image : numpy.ndarray
        The input image.
    
    Returns:
    radius : float
        The radius of the blob.
    '''
    
    image = image.astype(np.uint8)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)

    (x, y), radius = cv2.minEnclosingCircle(max_contour)
    
    return radius

def r0_corr(fields):
    '''
    Compute the coherence length and coherence function from a set of fields.

    Inputs:
    fields : numpy.ndarray
        The input fields.
    
    Returns:
    r0 : float
        The coherence length in pixels.

    corr_i : numpy.ndarray
        The coherence function.
    '''

    fields = np.asarray(fields)
    mcfs = np.zeros_like(fields)

    for i in (range(mcfs.shape[0])):
        mcfs[i] = mcf(fields[i])

    corr_i = np.mean(mcfs, axis=0)
    corr_i /= np.max(corr_i)

    corr = np.copy(corr_i)
    corr[corr < 1/np.e] = 0
    corr[corr >= 1/np.e] = 1
    
    r0 = 2.1*(find_blob_radius(corr))
    
    return r0, corr_i

def scintillation_index(fields, unbound = False, mask = None):
    '''
    Compute the scintillation index from a set of fields.

    Inputs:
    fields : numpy.ndarray
        The input fields.
    unbound : bool, optional
        Whether to calculate the scintillation index for a single pixel or over the whole field.
    mask : numpy.ndarray, optional
        The mask to apply to the fields.
    
    Returns:
    scintillation : float
        The scintillation index.
    '''

    intensities = np.abs(fields)**2
    if mask is not None:
        intensities = np.multiply(intensities, mask)
    if unbound:
        # find intensities of the central pixel
        powers = intensities[:,int(intensities.shape[1]/2),int(intensities.shape[2]/2)]
    else:
        powers = np.sum(intensities, axis=(1,2))
    scintillation = np.var(powers) / np.mean(powers)**2

    return scintillation

def scintillation_index_1D(powers):
    '''
    Compute the scintillation index from a set of powers.

    Inputs:
    powers : numpy.ndarray
        The input powers.
    
    Returns:
    scintillation : float
        The scintillation index.
    '''

    scintillation = np.var(powers) / np.mean(powers)**2

    return scintillation

def r0_difflambda(r0, lambda1, lambda2):
    '''
    Calculate the Fried parameter for a different wavelength.

    Inputs:
    r0 : float
        The Fried parameter at the original wavelength.
    lambda1 : float
        The original wavelength.
    lambda2 : float
        The new wavelength.
    
    Returns:
    r0_new : float
        The Fried parameter at the new wavelength.
    '''
    
    r0_new = r0 * (lambda2 / lambda1)**(6/5)

    return r0_new

def r0_net(r0s):
    '''
    Calculate the net fried parameter for N layers of turbulence (with their own fried parameters).

    Inputs:
    r0s : numpy.ndarray
        The fried parameters of the N layers of turbulence.

    Returns:
    r0_net : float
        The net fried parameter.
    '''
    if not isinstance(r0s, np.ndarray):
        r0s = np.array(r0s)
    r0_net = (r0s**(-5/3)).sum()**(-3/5)

    return r0_net

def kol_ref_index(sz):

    if isinstance(sz, int):
        s = [sz, sz]
    else:
        s = sz

    ph = np.zeros((s[0], s[1], 2))

    [x, y] = np.meshgrid(np.arange(-s[1], s[1]), np.arange(-s[0], s[0]))

    f = 2 * np.pi * np.random.rand(2 * s[0], 2 * s[1])
    pconv = (((x / s[1]) ** 2 + (y / s[0]) ** 2) * (s[0] ** 2 + s[1] ** 2)) ** (-11 / 12)
    pconv[s[0], s[1]] = 0
    psub = np.zeros((2 * s[0], 2 * s[1]), dtype=complex)
    p1 = np.floor(np.array(s) / 2).astype(int)
    p2 = p1 + np.array(s) - 1

    def Sinc(x):
        return np.sinc(x / np.pi)

def kolphase(sz, method):
    """
    ph = kolphase(sz, method)
    ph = kolphase([s1, s2], method)
    Computes a pair of Kolmogorov phase screens using the FT of
    random complex numbers with appropriate amplitudes.
    Screens are computed on a grid of size 2s, with s x s or s1 x s2 sized
    pieces cut out from the center and returned. This helps overcome
    the problem of under-representing tilt.

    ph = kolphase(s, 'lane')
    Uses the subharmonic method of Lane, Glindemann, & Dainty:
    "Simulation of a Kolmogorov phase screen," Waves in Random Media 2,
    209-224 (1992)
    This explicitly adds in undersampled low-frequency components with
    spatial scales larger than s to prevent a lack of low-frequency
    power. More accurate than 'roddier'.

    ph = kolphase(s, 'roddier')
    Uses the method of Roddier: "The Effects of Atmospheric Turbulence
    in Optical Astronomy," Progress in Optics, 19, 281-376 (1981)
    Random tilts are explicitly added to the phase screens to give
    a reasonable approximation of the overall Kolmogorov power spectrum
    at low frequency. Faster than 'lane'.

    size(ph) = [s, s, 2]
    """
    if isinstance(sz, int):
        s = [sz, sz]
    else:
        s = sz

    ph = np.zeros((s[0], s[1], 2))

    [x, y] = np.meshgrid(np.arange(-s[1], s[1]), np.arange(-s[0], s[0]))

    f = 2 * np.pi * np.random.rand(2 * s[0], 2 * s[1])
    pconv = (((x / s[1]) ** 2 + (y / s[0]) ** 2) * (s[0] ** 2 + s[1] ** 2)) ** (-11 / 12)
    pconv[s[0], s[1]] = 0
    psub = np.zeros((2 * s[0], 2 * s[1]), dtype=complex)
    p1 = np.floor(np.array(s) / 2).astype(int)
    p2 = p1 + np.array(s) - 1

    def Sinc(x):
        return np.sinc(x / np.pi)

    if method == 'lane':
        for n in range(1, 6):
            w = 3 ** -(2 * n)
            for k in range(-1, 2):
                ks = k * 3 ** -n
                sx = Sinc(np.pi * (x - ks))
                t = 2 if k == 0 else 1
                for l in range(-1, t + 1, t):
                    ls = l * 3 ** -n
                    sy = Sinc(np.pi * (y - ls))
                    psub += w * sx * sy * np.exp(1j * np.random.rand() * 2 * np.pi)
    elif method == 'roddier':
        scale = 3.30
        xt = np.random.randn(2) * scale / s[1]
        yt = np.random.randn(2) * scale / s[0]
    else:
        raise ValueError('Unknown method specified')

    sc = np.fft.fft2(np.fft.fftshift((pconv + psub) * np.exp(1j * f)))
    ph[:, :, 0] = np.real(sc[p1[0]:p2[0] + 1, p1[1]:p2[1] + 1])
    ph[:, :, 1] = np.imag(sc[p1[0]:p2[0] + 1, p1[1]:p2[1] + 1])

    if method == 'roddier':
        ph[:, :, 0] += xt[0] * x[p1[0]:p2[0] + 1, p1[1]:p2[1] + 1] + yt[0] * y[p1[0]:p2[0] + 1, p1[1]:p2[1] + 1]
        ph[:, :, 1] += xt[1] * x[p1[0]:p2[0] + 1, p1[1]:p2[1] + 1] + yt[1] * y[p1[0]:p2[0] + 1, p1[1]:p2[1] + 1]

    ph[:, :, 0] -= np.mean(ph[:, :, 0])
    ph[:, :, 1] -= np.mean(ph[:, :, 1])

    return ph

def zernike(sz, **kwargs):
    """
    z = zernike(n, m, sz)
    Generates the (n, m) Zernike polynomial, where n >= m, on the inscribed circle
    in a square raster of side 'sz'.

    Parameters:
        n:   Radial order of polynomial; must be non-negative
        m:   Azimuthal order: in the range m = -n:2:n
        j:   ANSI index
        sz:  Linear size of output raster

    Returns:
        z:  2D raster
    """

    if 'j' in kwargs:
        j = kwargs.get('j', 0)
        n = math.ceil((-3 + np.sqrt(9 + 8*j))/2)
        m = int(2*j - n*(n+2))
    elif 'm' in kwargs and 'n' in kwargs:
        m = kwargs.get('m', 0)
        n = kwargs.get('n', 0)
    else:
        raise ValueError('Either j or n and m must be specified')

    ma = abs(m)
    if n < ma or n < 0:
        raise ValueError('n must be non-negative and n >= |m|')

    s = (n - ma) / 2
    if s - math.floor(s) != 0:
        raise ValueError('n - m must be even')

    c1 = -((sz - 1) / 2)
    c2 = sz + c1 - 1

    x, y = np.meshgrid(np.arange(c1, c2 + 1), np.arange(c1, c2 + 1))
    rho = np.sqrt(x**2 + y**2) / sz * 2
    theta = np.arctan2(y, x)

    R = np.zeros_like(rho)
    for k in range(int(s) + 1):
        R += (-1)**k * math.factorial(n - k) / (
            math.factorial(k) * math.factorial((n + ma) // 2 - k) * math.factorial(int(s) - k)
        ) * rho**(n - 2 * k)

    R *= circle(sz, sz)#, [(sz -1) / 2, (sz -1) / 2]) # changed from (sz+1)//2 to (sz-1)/2

    if m < 0:
        z = np.sqrt(2 * (n + 1)) * R * np.sin(ma * theta)
    elif m > 0:
        z = np.sqrt(2 * (n + 1)) * R * np.cos(ma * theta)
    else:
        z = np.sqrt(n + 1) * R

    return z

def ao_lowpass(screens, subapp, sf=1, mask = None, shift_fft =  False):
    """
    Emulate AO compensation by applying a low pass filter based on the subaperture size.

    Additionally apply aperture over the screens.

    Parameters:
        screens : np.ndarray
            The screens to be filtered, shape (N, N, n_frames).
        subapp : int
            The number of subapertures.
        sf : float, optional
            Scale factor for the filter size, default is 1.

    Returns:
        filtered : np.ndarray
            The filtered screens, shape (N, N, n_frames).
    """

    N = screens.shape[0]

    length = N / subapp * sf
    fil_mask = circle(N, length)#, (N//2, N//4))
    fil_mask /= np.sum(fil_mask)  # normalize the mask
    # make fil_mask the same size as screens
    # if fil_mask.ndim == 2:
    #     fil_mask = np.repeat(fil_mask[..., np.newaxis], screens.shape[-1], axis=-1)

    # add dims to match screens. screen can be 2D, 3D or 4D but mask is always 2D

    counter = 2
    while fil_mask.ndim < screens.ndim:
        fil_mask = fil_mask[..., np.newaxis]
        fil_mask = np.repeat(fil_mask, screens.shape[counter], axis=counter)
        counter+=1
    

    mask_fft = (np.fft.fft2(fil_mask, axes = (0, 1)))
    screens_fft = (np.fft.fft2(screens, axes= (0, 1)))
    screens_fft *= np.conj(mask_fft)
    filtered = np.fft.fftshift(np.real((np.fft.ifft2(screens_fft, axes = (0,1)))), axes = (0,1))

    # auto align 

    if shift_fft:
        # create soft edge mask
        soft_edge = circle(N, N*0.90, edge='soft')
        circle2 = circle(N, 0.5*N, edge='hard')
        soft_edge[circle2==1] = 0
        soft_edge /= np.max(soft_edge)
        soft_edge += circle2

        # 2d correlate
        orig = screens[...,0]*soft_edge
        fil = filtered[...,0]*soft_edge
        corr = correlate2D(orig, fil)
        peak = np.where(corr == np.max(corr))
        peak = np.asarray([peak[0][0], peak[1][0]])
        shift = np.array(peak) - np.array(corr.shape) // 2 - 1
        filtered = np.roll(filtered, shift, axis=(0, 1))

    if mask is not None:
        filtered = np.multiply(filtered, mask[..., np.newaxis])

    return filtered

def expected_variance(r0s, diameter, calc_net = False):

    """
    Calculate the theoretical expected variance of the phase screens over the groundstation aperture.
    Parameters:
        r0s (list or numpy.ndarray) :
            Fried parameters of the atmosphere in meters.
        diameter (float) :
            Diameter of the groundstation aperture in meters.
        calc_net (bool, optional) :
            If True, calculates the net Fried parameter from the provided r0s.
    Returns:
    var (numpy.ndarray) :
        Expected variance of the phase screens over the aperture.
    var_notilt (numpy.ndarray) :
        Expected variance of the phase screens over the aperture without tilt.
    """ 

    # if r0s is a float, convert it to a list
    if isinstance(r0s, float):
        if isinstance(diameter, (list, np.ndarray)):
            r0s = [r0s] * len(diameter)
        else:
            r0s = [r0s]
    # elif isinstance(r0s, (list, np.ndarray)):
    #     if len(r0s) != len(diameter):
    #         raise ValueError("If r0s is a list or array, it must have the same length as diameter.")

    r0s = list(r0s)
    if calc_net:
        r0s.append(r0_net(r0s))
    r0s = np.array(r0s)

    var = np.array(1.0299 * (diameter / r0s)**(5/3))
    var_notilt = np.array(0.134*(diameter/r0s)**(5/3))
    
    return var, var_notilt

class ZernikeDecomposer:
    """Zernike decomposer for phase screens, with precomputed basis and reconstructor."""

    def __init__(self, n_modes, Nd, ignore_piston=True, Nd_vis=None):
        self.n_modes = n_modes
        self.Nd = Nd
        self.Nd_vis = Nd_vis if Nd_vis is not None else Nd
        self.zernike_basis, self.reconstructor, self.infl, self.mode_variances = self._generate_zernike_basis()
        self.zernike_basis_vis = self._build_vis_basis()

    def _generate_zernike_basis(self):
        """Generate Zernike basis functions for the given number of modes."""

        basis = np.zeros((self.Nd, self.Nd, self.n_modes))
        mask = circle(self.Nd, self.Nd)
        infl = np.zeros((int(np.sum(mask)), self.n_modes))
        for modenum in range(self.n_modes):
            vec = zernike(self.Nd, j=modenum)#/np.sum(mask)**0.5
            basis[..., modenum] = vec
            infl[:, modenum] = vec[mask > 0]
        reconstructor = np.linalg.pinv(infl)
        mode_variances = np.var(basis[mask > 0, :], axis=0)

        return basis, reconstructor, infl, mode_variances

    def _build_vis_basis(self):
        if self.Nd_vis == self.Nd:
            return self.zernike_basis
        basis = np.zeros((self.Nd_vis, self.Nd_vis, self.n_modes))
        for j in range(self.n_modes):
            basis[..., j] = zernike(self.Nd_vis, j=j)
        return basis
    
    def decompose(self, screen, mask = None, recon = None):
        """Decompose a screen into Zernike coefficients. Accepts up to 4D input"""
        if mask is None:
            mask = circle(self.Nd, self.Nd)
        screen_flat = screen[mask > 0,...]
        if recon is None:
            recon = self.reconstructor
        coeffs = np.tensordot(recon, screen_flat, axes=([-1], [0]))
        # coeffs = self.reconstructor @ screen_flat
        return coeffs
    
    def reconstruct_screen(self, screen):
        """Reconstruct a screen from Zernike coefficients and calculate MSE."""
        basis = self.zernike_basis
        coeffs = self.decompose(screen)
        
        recon = basis @ coeffs
        mse = np.var(recon, axis = (0,1))
        return recon, mse
    
    def reconstruct_screen_from_coeffs(self, coeffs):
        """Reconstruct a screen from given Zernike coefficients."""
        basis = self.zernike_basis
        # recon = basis @ coeffs
        recon = np.tensordot(basis, coeffs, axes=([-1], [0]))

        return recon

class ZernikeSlopeDecomposer:
    """Zernike reconstructor using X/Y slope measurements (e.g. Shack-Hartmann WFS).

    Influence matrix columns are [dZ_j/dx, dZ_j/dy] over mask pixels, stacked vertically.
    """

    def __init__(self, n_modes, Nd, Nd_vis=None):
        self.n_modes = n_modes
        self.Nd = Nd
        self.Nd_vis = Nd_vis if Nd_vis is not None else Nd
        self.mask = circle(Nd, Nd)
        self._n_pix = int(np.sum(self.mask > 0))
        self.zernike_basis, self.infl, self.reconstructor = self._build()
        self.zernike_basis_vis = self._build_vis_basis()

    def _build(self):
        mask_flat = self.mask > 0
        basis = np.zeros((self.Nd, self.Nd, self.n_modes))
        infl = np.zeros((2 * self._n_pix, self.n_modes))
        for j in range(self.n_modes):
            z = zernike(self.Nd, j=j)
            basis[..., j] = z
            dy, dx = np.gradient(z)  # np.gradient: axis-0 = row = y, axis-1 = col = x
            infl[:self._n_pix, j] = dx[mask_flat]
            infl[self._n_pix:, j] = dy[mask_flat]
        return basis, infl, np.linalg.pinv(infl)

    def _build_vis_basis(self):
        if self.Nd_vis == self.Nd:
            return self.zernike_basis
        basis = np.zeros((self.Nd_vis, self.Nd_vis, self.n_modes))
        for j in range(self.n_modes):
            basis[..., j] = zernike(self.Nd_vis, j=j)
        return basis

    def decompose(self, slopes_x_or_phase, slopes_y=None, mask=None, recon=None):
        """Reconstruct Zernike coefficients from slope maps or a phase screen.

        Pass (slopes_x, slopes_y) or a single phase screen (slopes computed internally).
        """
        if slopes_y is None:
            dy, dx = np.gradient(slopes_x_or_phase)
            slopes_x, slopes_y = dx, dy
        else:
            slopes_x = slopes_x_or_phase
        m = self.mask > 0 if mask is None else mask > 0
        s = np.concatenate([slopes_x[m], slopes_y[m]])
        if recon is None:
            recon = self.reconstructor
        return recon @ s

    def reconstruct_screen(self, slopes_x_or_phase, slopes_y=None, mask=None):
        """Return (phase_screen, coeffs) reconstructed from slope maps or a phase screen."""
        coeffs = self.decompose(slopes_x_or_phase, slopes_y, mask)
        return np.tensordot(self.zernike_basis, coeffs, axes=([-1], [0])), coeffs


if __name__ == "__main__":
    # Example usage
    
    heights = np.linspace(0,30e3,1000)
    v_wind_mean = np.mean(v_wind(heights))
    cn2s = get_c2n(heights,wind_rms=v_wind_mean)

    # find cumulative cn2 with height

    net_cn2s= []
    w_lts = []
    w_lts2 = []
    w_dls =[]
    for i,height in enumerate(heights):
        if i == 0:
            continue
        net_cn2s.append(cum_cn2(cn2s, heights, height_f = height))
        w_lts.append(wLT(cn2s, heights, height_f = height))
        w_lts2.append(spotsizeLT(cn2s, heights, height_f = height, w0 = 0.1))
        w_dls.append(gaussz(0.1, height))

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(heights[1:], net_cn2s, label='Net Cn2')
    plt.plot(heights, cn2s, label='Cn2')
    plt.xlabel('Height (m)')
    plt.ylabel('Cn2 (m^(-2/3))')
    plt.yscale('log')
    plt.title('Cumulative Cn2 with Height')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(heights[1:], w_lts)
    plt.plot(heights[1:], w_lts2, linestyle='--')
    plt.plot(heights[1:], w_dls, linestyle=':')

    plt.xlabel('Height (m)')
    plt.ylabel('W_L_T (m)')
    plt.title('W_L_T with Height')
    plt.grid()

    plt.show()

