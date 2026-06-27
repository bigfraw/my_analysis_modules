'''
Beam field related functions
'''

import numpy as np
import addcopyfighandler

def zR(w0, lamda=1550e-9):
    '''
    Rayleigh range for a Gaussian beam

    Parameters:
        w0 : float
            Beam waist radius at the focus (z=0)
        lamda : float
            Wavelength of the light
    Returns:
        float
            Rayleigh range
    '''
    return np.pi * w0**2 / lamda

def gaussz(w0, z, lamda=1550e-9):
    '''
    Gaussian beam radius at a distance z

    Parameters:
        w0 : float
            Beam waist radius at the focus (z=0)
        z : float
            Distance from the focus
        lamda : float
            Wavelength of the light

    Returns:
        float
            Gaussian beam radius at distance z
    '''
    
    zR = np.pi * w0**2 / lamda  # Rayleigh range
    wz = w0 * np.sqrt(1 + (z / zR)**2)
    return wz

def beamparam(dist, w0, labda=1550e-9):
    '''
    Find the Fresnel ratio at a given distance (from the beam waist)

    Parameters:
        dist (float) : Distance from transmitter to plane.
        w0 (float) : Beam waist.
        labda (float) : Wavelength.

    Returns:
    Fresnel ratio at plane
    '''
    beamparam0 = 2*dist/(2*np.pi/labda*w0**2)
    return beamparam0/(1+beamparam0**2)

def gaussfield1D(x, z, w0, power=1, lamda=1550e-9):
    '''
    Generate a 1D Gaussian beam electric field profile at distance z.

    Parameters:
        x (numpy.ndarray): 1D array of spatial coordinates.
        w0 (float): Beam waist radius at the focus (z=0).
        z (float): Distance from the focus.
        lamda (float): Wavelength of the light.

    Returns:
        numpy.ndarray: Complex electric field profile of the Gaussian beam at distance z.
    '''

    if isinstance(z, np.ndarray):
        z[z== 0] = 1e-10  # Avoid division by zero at the waist
    elif z == 0:
        z = 1e-10  # Avoid division by zero at the waist
    wz = gaussz(w0, z, lamda)
    k = 2 * np.pi / lamda
    zR = np.pi * w0**2 / lamda  # Rayleigh range
    Rz = z * (1 + (zR/z)**2)  # Radius of curvature
    gouy_phase = np.arctan(z / (np.pi * w0**2 / lamda))  # Gouy phase

    E = (w0 / wz) * np.exp(-x**2 / wz**2) * np.exp(-1j * (k*z + k * x**2 / (2 * Rz) - gouy_phase))
    E /= np.sum(np.abs(E)**2/power)**0.5

    return E

def gaussfieldRz(R,z,w0,lamda=1550e-9, power = None):
    '''
    Generate a radial Gaussian beam electric field profile at distance z.
    Parameters:
        R (numpy.ndarray): 1D array of radial spatial coordinates.
        w0 (float): Beam waist radius at the focus (z=0).
        z (float): Distance from the focus.
        lamda (float): Wavelength of the light.
        power (float): If provided, fixes the total power of the beam to this value
    '''

    if isinstance(z, np.ndarray):
        z[z== 0] = 1e-10  # Avoid division by zero at the waist
    elif z == 0:
        z = 1e-10  # Avoid division by zero at the waist

    wz = gaussz(w0, z, lamda)
    k = 2 * np.pi / lamda
    zR = np.pi * w0**2 / lamda  # Rayleigh range
    Rz = z * (1 + (zR/z)**2)  # Radius of curvature
    gouy_phase = np.arctan(z / zR)  # Gouy phase

    E = (w0 / wz) * np.exp(-(R**2) / wz**2) * np.exp(-1j * (k*z + k * (R**2) / (2 * Rz) - gouy_phase))
    if power is not None:
        # E *= np.sqrt(2*power/ (np.pi * w0**2))
        # E *= 1/ w0**0.5'
        E /= np.sum(np.abs(E)**2/power)**0.5
    return E

def gaussfield2D(X, Y, z, w0, lamda=1550e-9):
    '''
    Generate a 2D Gaussian beam electric field profile at distance z.

    Parameters:
        X (numpy.ndarray): 2D array of x spatial coordinates.
        Y (numpy.ndarray): 2D array of y spatial coordinates.
        w0 (float): Beam waist radius at the focus (z=0).
        z (float): Distance from the focus.
        lamda (float): Wavelength of the light.

    Returns:
        numpy.ndarray: Complex electric field profile of the Gaussian beam at distance z.
    '''
    if z == 0:
        z = 1e-10  # Avoid division by zero at the waist
    wz = gaussz(w0, z, lamda)
    k = 2 * np.pi / lamda
    R = z * (1 + (np.pi * w0**2 / (lamda * z))**2)  # Radius of curvature
    gouy_phase = np.arctan(z / (np.pi * w0**2 / lamda))  # Gouy phase

    E = (w0 / wz) * np.exp(-(X**2 + Y**2) / wz**2) * np.exp(-1j * (k * (X**2 + Y**2) / (2 * R) - gouy_phase))
    return E

def planewavefield(X, Y, z, lamda=1550e-9):
    '''
    Generate a 2D plane wave electric field profile at distance z.

    Parameters:
        X (numpy.ndarray): 2D array of x spatial coordinates.
        Y (numpy.ndarray): 2D array of y spatial coordinates.
        z (float): Distance from the source.
        lamda (float): Wavelength of the light.

    Returns:
        numpy.ndarray: Complex electric field profile of the plane wave at distance z.
    '''
    k = 2 * np.pi / lamda
    inten = 1 * np.ones_like(X)
    E = inten * np.exp(-1j * k * z)
    return E

def sphericalwavefield(X, Y, z, lamda=1550e-9):
    '''
    Generate a 2D spherical wave electric field profile at distance z.

    Parameters:
        X (numpy.ndarray): 2D array of x spatial coordinates.
        Y (numpy.ndarray): 2D array of y spatial coordinates.
        z (float): Distance from the source.
        lamda (float): Wavelength of the light.

    Returns:
        numpy.ndarray: Complex electric field profile of the spherical wave at distance z.
    '''
    k = 2 * np.pi / lamda
    r = np.sqrt(X**2 + Y**2 + z**2)
    E = np.exp(-1j * k * r) / r
    return E

def overlap(field1, field2):

    numerator = np.abs(np.sum(field1 * np.conj(field2)))**2
    denominator = np.sum(field1 * np.conj(field1)) * np.sum(field2 * np.conj(field2))

    return (numerator/denominator).astype(np.float64)

def overlap_field(field1, field2):

    denominator = np.sum(field1 * np.conj(field1)) * np.sum(field2 * np.conj(field2))
    field = field1*np.conj(field2)

    return (field/denominator)**0.5

def fresnel_propagator(field, z, dx, labda=1550e-9):
    '''
    Takes a complex field and propagates it a distance z using the Fresnel approximation
    (angular spectrum method).

    Parameters:
        field (numpy.ndarray): 2D array representing the complex field to be propagated.
        z (float): Distance to propagate the field.
        dx (float): Grid spacing (metres).
        labda (float): Wavelength (metres).

    Returns:
        numpy.ndarray: Propagated complex field.
    '''
    k = 2 * np.pi / labda

    # Spatial frequencies in rad/m
    fx = np.fft.fftfreq(field.shape[1], d=dx)
    fy = np.fft.fftfreq(field.shape[0], d=dx)
    kx, ky = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy)

    # Fresnel transfer function: exp(ikz) * exp(-iz(kx²+ky²)/2k)
    H = np.exp(1j * k * z) * np.exp(-1j * z * (kx**2 + ky**2) / (2 * k))

    field_ft = np.fft.fft2(field)
    return np.fft.ifft2(field_ft * H)


if __name__ == "__main__":

    # 1D Gaussian beam profile example
    import matplotlib.pyplot as plt
    import lpmodes as lp

    grid_size = 256
    max_plot_radius = 15

    core_n = 1.466
    cladding_n = 1.44692 
    core_radius = 4.1
    wavelength = 1.55
    modes = lp.find_modes(core_radius, core_n, cladding_n, wavelength)
    mode = modes[0]
    mode_plot = mode.plot_amplitude(grid_size, max_plot_radius)
    mode_plot = mode_plot[mode_plot.shape[0]//2,:]
    mode_plot /= np.sum(np.abs(mode_plot))
    lengths = np.linspace(-15,15,grid_size)

    gauss_profile = gaussfield1D(lengths*1e-6, 0, w0=5.25e-6, lamda = 1550e-9, power=1)
    gauss_inten = np.abs(gauss_profile)**2
    gauss_inten /= np.sum(gauss_inten)

    plt.figure(dpi=150)
    plt.plot(lengths, mode_plot)
    plt.plot(lengths, np.abs(gauss_profile)**2, '--')
    plt.show()

    exit()

    x = np.linspace(-5e-3, 5e-3, 500)
    y = x.copy()
    X,Y = np.meshgrid(x,y)
    field = gaussfield2D(X, Y, 0, 8e-4)
    plt.figure(figsize=(6,5))
    plt.pcolormesh(x*1e3, y*1e3, np.abs(field)**2, shading='auto')
    # equal axis scale
    plt.axis('equal')
    plt.show()
    exit()
    
    r = np.linspace(-5e-3, 5e-3, 500)
    z = np.linspace(0, 10, 500)  # distances in meters
    R,Z = np.meshgrid(r, z)
    w0 = 8e-4
    f = gaussfieldRz(R,Z,w0,lamda=1550e-9)

    

    #calcualte wavefront from phase
    param = np.abs(f)**2
    # param = np.real(f)

    plt.figure(figsize=(0.5,1))
    plt.plot(r*1e3,param[250,:])
    # add vertical lines for w0
    plt.axvline(x=w0*1e3, color='r', linestyle='--', label='w0')
    plt.axvline(x=-w0*1e3, color='r', linestyle='--')
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('r (mm)')
    # plt.ylabel('Intensity (a.u.)')
    # plt.tight_layout()
    plt.show()

    # for row in range(phase.shape[0]):
    #     phase[row,:] = np.unwrap(phase[row,:]) - phase[row,0]
        # phase[row,:] -= np.mean(phase[row,:])
    plt.figure(figsize=(6,3))
    plt.pcolormesh(Z,R*1e3, param, shading='auto')
    # plt.axis('equal')
    plt.xlabel('z (m)')
    plt.ylabel('r (mm)')
    # plt.colorbar(label='Unwrapped phase (rad)')
    plt.colorbar(label='Re(E) (a.u)')
    plt.tight_layout()
    plt.show()