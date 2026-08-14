from __future__ import division
import warnings
import numpy
from codes import auxiliars

def my_xderiv(x, y, data, n = 1):
    
    '''
    Return the horizontal derivative in x direction for n order in Fourier domain.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - float - order of the derivative
    
    Output:
    xder - numpy 2D array - derivative in x direction
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have same shape!")
    
    # Condition for the order of the derivative
    if n <= 0.:
        raise ValueError("Order of the derivative must be positive and nonzero!")
    
    if n == 0.:
        res = data
    else:    
        # Calculate the wavenuber in x direction
        _, kx = auxiliars.my_wavenumber(y, x)
        # Apply the Fourier transform
        xder = numpy.fft.fft2(data)*((kx*1j)**(n))
        # Calculating the inverse transform
        res = numpy.real(numpy.fft.ifft2(xder))
    # Return the final output
    return res.reshape(res.size)

def my_yderiv(x, y, data, n = 1):
    
    '''
    Return the horizontal derivative in y direction for n order in Fourier domain.
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - float - order of the derivative
    
    Output:
    yder - numpy 2D array - derivative in y direction
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have same shape!")
    # Condition for the order of the derivative
    if n <= 0.:
        raise ValueError("Order of the derivative must be positive and nonzero!")
        
    if n == 0.:
        res = data
    else:    
        # Calculate the wavenuber in y direction
        ky, _ = auxiliars.my_wavenumber(y, x)
    
        # Apply the Fourier transform
        yder = numpy.fft.fft2(data)*((ky*1j)**(n))
        # Calculate the inverse transform
        res = numpy.real(numpy.fft.ifft2(yder))
    
    # Return the final output
    return res.reshape(res.size)
    
def my_zderiv(x, y, data, n = 1):
    
    '''
    Return the vertical derivative in z direction for n order in Fourier domain.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    n - float - order of the derivative
    
    Output:
    zder - numpy 2D array - derivative in z direction    
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have same shape!")
    
    # Condition for the order of the derivative
    if n < 0.:
        raise ValueError("Order of the derivative must be positive!")
    
    if n == 0.:
        res = data
    else:    
        # Calculate the wavenuber in z direction
        ky, kx = auxiliars.my_wavenumber(y, x)
    
        # Apply the Fourier transform
        zder = numpy.fft.fft2(data)*(numpy.sqrt(kx**2 + ky**2)**(n))
    
        # Calculate the inverse transform
        res = numpy.real(numpy.fft.ifft2(zder))
    
    # Return the final output
    return res.reshape(res.size)
    
def my_hgrad(x, y, data):
    
    '''
    Return the horizontal gradient amplitude (HGA) for a potential data on a regular 
    grid. All calculation is done by using Fourier domain.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    
    Output:
    hga - numpy 2D array - horizontal gradient amplitude
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have same shape!")
    
    # Computes the horizontal derivatives
    diffx = my_xderiv(x, y, data)
    diffy = my_yderiv(x, y, data)
    
    # Calculates the total gradient
    hgrad = (diffx**2 + diffy**2)**(0.5)
    
    # Return the final output
    return hgrad

def my_totalgrad(x, y, data):
    
    '''
    Return the total gradient amplitude (TGA) for a potential data on a regular grid.
    
    Inputs:
    x - numpy 2D array - grid values in x direction
    y - numpy 2D array - grid values in y direction
    data - numpy 2D array - potential data
    
    Output:
    tga - numpy 2D array - total gradient amplitude
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape != data.shape:
        raise ValueError("All inputs must have same shape!")

    # Calculates the x derivative
    diffx = my_xderiv(x, y, data)
    # Calculates the y derivative
    diffy = my_yderiv(x, y, data)
    # Calculates the z derivative
    diffz = my_zderiv(x, y, data)

    # Calculates the total gradient
    res = (diffx**2 + diffy**2 + diffz**2)**(0.5)
    
    # Return the final output
    return res.reshape(res.size)