import numpy as np
from scipy.fft import fftn, ifftn, fftshift
from scipy.interpolate import RegularGridInterpolator

"""
Misc array calculations
"""

def correlate2D(a, b):
    """
    Computes the 2D cross-correlation of two arrays a and b.

    Will correlate over the first two dimensions of the arrays.
    
    Parameters:
        a (numpy.ndarray): First input array.
        b (numpy.ndarray): Second input array.
    
    Returns:
        numpy.ndarray: Cross-correlation of a and b.
    """
    if a.ndim != b.ndim:
        raise ValueError("Input arrays must have the same number of dimensions.")

    fft_a = np.fft.fftn(a,axes=(0, 1))
    fft_b = np.fft.fftn(b,axes=(0, 1))
    cross_correlation = np.fft.ifftn(fft_a * np.conj(fft_b), axes=(0, 1))
    cross_correlation = np.fft.fftshift(cross_correlation)  # Shift zero frequency component to the center

    return np.real(cross_correlation) # imag components is ~zero

def correlate1D(a, b):
    """
    Computes the 1D cross-correlation of two arrays a and b.

    Will correlate over the last dimension of the arrays.
    
    Parameters:
        a (numpy.ndarray): First input array.
        b (numpy.ndarray): Second input array.
    """

    if a.ndim != b.ndim:
        raise ValueError("Input arrays must have the same number of dimensions.")

    fft_a = np.fft.fft(a)
    fft_b = np.fft.fft(b)
    cross_correlation = np.fft.ifft(fft_a * np.conj(fft_b))
    cross_correlation = np.fft.fftshift(cross_correlation)  # Shift zero frequency component to the center

    return np.real(cross_correlation) # imag components is ~zero

def interpolate2d(input, output_shape,method = 'linear'):
    '''
    Interpolates a 2D array to a new shape using linear interpolation.

    Parameters:
        input (numpy.ndarray): The input 2D array to be interpolated.
        output_shape (int): The desired output shape for both dimensions of the interpolated array.
        method (str): The interpolation method to use.

    Returns:
        numpy.ndarray: The interpolated 2D array with the specified output shape.
    '''

    # TODO: Add support for n>2 input arrays.

    input_shape = input.shape

    # Handle output_shape parameter
    if isinstance(output_shape, (int, np.integer)):
        new_shape_2d = (output_shape, output_shape)
    elif isinstance(output_shape, (tuple, list)) and len(output_shape) == 2:
        new_shape_2d = tuple(output_shape)
    else:
        raise ValueError("output_shape must be an int or tuple/list of 2 elements")
    
    # output_shape = np.array([output_shape, output_shape])

    if input.ndim == 2:
        
        x = np.linspace(0, input_shape[1] - 1, input_shape[1])
        y = np.linspace(0, input_shape[0] - 1, input_shape[0])
        x_new = np.linspace(0, input_shape[1] - 1, new_shape_2d[1])
        y_new = np.linspace(0, input_shape[0] - 1, new_shape_2d[0])
        interpolating_function = RegularGridInterpolator((y, x), input, method=method, bounds_error=False, fill_value=0)
        y_new, x_new = np.meshgrid(y_new, x_new, indexing='ij')
        output = interpolating_function((y_new.ravel(), x_new.ravel())).reshape(new_shape_2d)

        return output
    
    elif input.ndim > 2:

        # Reshape input to 3D array with last dimension as slices
        original_shape = input.shape
        reshaped_input = input.reshape(original_shape[0], original_shape[1], -1)
        n_slices = reshaped_input.shape[-1]

        # Calculate output shape
        output_full_shape = new_shape_2d + original_shape[2:]
        output_reshaped = np.zeros((new_shape_2d[0], new_shape_2d[1], n_slices), dtype=input.dtype)

        # Create coordinate grids
        x = np.linspace(0, original_shape[1] - 1, original_shape[1])
        y = np.linspace(0, original_shape[0] - 1, original_shape[0])
        x_new = np.linspace(0, original_shape[1] - 1, new_shape_2d[1])
        y_new = np.linspace(0, original_shape[0] - 1, new_shape_2d[0])
        y_new_mesh, x_new_mesh = np.meshgrid(y_new, x_new, indexing='ij')
        coords_new = np.column_stack([y_new_mesh.ravel(), x_new_mesh.ravel()])
        
        # Process each 2D slice
        for i in range(n_slices):
            slice_2d = reshaped_input[:, :, i]
            
            interpolating_function = RegularGridInterpolator(
                (y, x), slice_2d, method=method, bounds_error=False, fill_value=0
            )
            interpolated_slice = interpolating_function(coords_new).reshape(new_shape_2d)
            output_reshaped[:, :, i] = interpolated_slice
        
        # Reshape back to original dimensions
        output = output_reshaped.reshape(output_full_shape)
        return output 

def embed_square(embed_factor, arr_2d):
        """Embed a 2D array in a larger square array centered at the origin."""
        ny, nx = arr_2d.shape
        n_base = max(ny, nx)
        n_embed = int(np.ceil(embed_factor * n_base))
        if n_embed < n_base:
            n_embed = n_base
        if n_embed % 2 != 0:
            n_embed += 1

        out = np.zeros((n_embed, n_embed), dtype=arr_2d.dtype)

        y0 = (n_embed - ny) // 2
        x0 = (n_embed - nx) // 2
        out[y0:y0 + ny, x0:x0 + nx] = arr_2d
        return out

def pad_to_dim(arr, target_dim):
    """Pad a 2D array to a target dimension with zeros, centering the original array."""
    ny, nx = arr.shape
    if ny > target_dim or nx > target_dim:
        raise ValueError("Input array is larger than target dimension.")
    
    out = np.zeros((target_dim, target_dim), dtype=arr.dtype)
    
    y0 = (target_dim - ny) // 2
    x0 = (target_dim - nx) // 2
    out[y0:y0 + ny, x0:x0 + nx] = arr
    return out

def circle(s, d, center=None, edge='hard'):
    """
    Create an array of zeros of size s with a circle of diameter d filled with ones
    either in the center of the array or with center at [x, y]. The edge of the circle
    is defined by edge as either 'hard' or 'soft'.
    
    Parameters:
        s (int): Size of the array (s x s).
        d (int): Diameter of the circle.
        center (list, optional): Center of the circle [x, y]. Defaults to center of the array.
        edge (str, optional): Edge type of the circle ('hard' or 'soft'). Defaults to 'hard'.
    
    Returns:
    numpy.ndarray: Array with the circle.
    """
    
    if center is None:
        m = (s-1) / 2
        center = [m, m]
    elif len(center) != 2:
        raise ValueError('Error in center definition')
    # check if s is an array
    if isinstance(s, (list, tuple, np.ndarray)):
        c = np.zeros(s[0],s[1])
    
    c = np.zeros((s, s))
    y, x = np.ogrid[:s, :s]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    if edge == 'hard':
        mask = dist_from_center <= (d / 2)
    elif edge == 'soft':
        mask = dist_from_center <= (d / 2)
        c[mask] = 1 - (dist_from_center[mask] / (d / 2))
        return c
    else:
        raise ValueError('Edge must be either "hard" or "soft"')
    
    c[mask] = 1
    return c

def rectangle(s, l1, l2, center = None):
    """
    draw a rectangle

    Parameters:
        s (int): size of the array (s x s)
        l1 (int): length of the rectangle in the x direction
        l2 (int): length of the rectangle in the y direction
        center (list, optional): center of the rectangle [x, y]. Defaults to center of the array.

    Returns:
        numpy.ndarray: array with the rectangle
    """

    if center is None:
        m = (s-1)/2
        center = [m,m]
    
    c = np.zeros((s,s))
    y,x = np.ogrid[:s,:s]

    mask = (np.abs(x - center[0]) <= l1/2) & (np.abs(y - center[1]) <= l2/2)
    c[mask] = 1

    return c

def fit_parabola(img, d=1):

    """
    This function does sub-pixel centroiding through a parabolic fit on a bright spot on img. It returns the coordinates of the turning point of the parabola.

    Inputs:
    img: np.array
        A 2D array of the WFS frame split into subapertures
    d: int
        The distance from the max point to use for the parabola fit

    
    Returns:
    coms: np.array
        A 2D array of the coordinates of the turning points of the parabolas
    """

    max_val = np.max(img)
    max_index = np.where(img == max_val)
    y_max, x_max = max_index
    y_max = y_max[0]
    x_max = x_max[0]
    x = np.arange(x_max-d, x_max+1+d)
    y = np.arange(y_max-d, y_max+1+d)

    # if min(x) < 0 or max(x) >= data_split[i].shape[1] or min(y) < 0 or max(y) >= data_split[i].shape[0]:
    #     raise ValueError("d_parab too large for r")

    # flatten the values of x and y
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    # find the z value corresponding to each x,y pair
    zz = img[y[0]:y[-1]+1, x[0]:x[-1]+1].flatten()

    # fit a 2d parabola to the data

    # construct a matrix A corresponding to the equation z = ax^2 + by^2 + cx + dy + e for each x,y pair
    A = np.c_[xx**2, yy**2, xx, yy, np.ones(xx.shape)]
    # solve the equation Ap = zz for p (least squares)
    p = np.linalg.lstsq(A, zz, rcond=None)[0]

    xcoord = -p[2]/(2*p[0]) # x turning point coord = -b/2a for x values
    ycoord = -p[3]/(2*p[1]) # y turning point coord = -b/2a for y values

    centroid = [xcoord, ycoord]  

    return centroid

def sub_pixel_roll(data, shift, axis = -1, method='cubic', upsample = 5):
    """
    Roll an array by a sub-pixel amount along a specified axis using interpolation.

    Parameters:
        data (numpy.ndarray): Input array to be rolled. Of dimensions (X, Y, t)
        shift (float): Amount to roll the array by (can be fractional).
        axis (int): Axis along which to roll the array. Default is -1 (last axis).
        method (str): Interpolation method to use ('linear', 'nearest', 'cubic'). Default is 'cubic'.
        upsample (int): Upsampling factor for fractional shifts. Default is 10.

    Returns:
        numpy.ndarray or list: Rolled array with the same type as the input.
    """

    shift_int = int(np.floor(shift))
    shift_frac = shift - shift_int
    data = np.roll(data, shift_int, axis=axis)
    if shift_frac != 0:
        upsample = upsample
        temp = interpolate2d(data, output_shape=np.ceil(data.shape[0]*upsample).astype(int), method=method)
        roll_amt = np.round(shift_frac * upsample).astype(int)
        temp = np.roll(temp, roll_amt, axis=axis)
        data = interpolate2d(temp, output_shape=data.shape[0], method=method)
    return data
    

if __name__ == "__main__":

    import aotools
    import matplotlib.pyplot as plt
    # Example usage
    phases_test = aotools.phasescreen.ft_phase_screen(r0=10, N=256, delta=0.1, L0=100, l0=0.01,seed=0)
    
    phases_downsampled = interpolate2d(phases_test, 64)

    plt.subplot(2,2,1)
    plt.imshow(phases_test)
    plt.title('Original')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(phases_downsampled)
    plt.title('Upsampled')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.plot(phases_test[0,:])
    plt.ylabel('Phase (rad)')
    plt.title('Slice of original')
    plt.subplot(2,2,4)
    plt.plot(phases_downsampled[0,:])
    plt.ylabel('Phase (rad)')
    plt.title('Slice of upsampled')

    plt.show()