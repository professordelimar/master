#------------------------------------------------------------------------------------
# Title: Grav-Mag Codes
# Author: Nelson Ribeiro Filho
# Description: Source codes that will be necessary during the masters course.
# Collaborator: Rodrigo Bijani
#------------------------------------------------------------------------------------
import numpy
from codes import auxiliars
#------------------------------------------------------------------------------------
def my_potential(x, y, z, prism, rho):
    '''
    This function calculates the gravitational potential due to a rectangular prism. It is calculated 
    solving a numerical integral approximated by using the gravity field G(x,y,z), once G can be written 
    as minus the gradient of the gravitational potential. This function recieves all obsevation points 
    for an array or a grid and also the value for height of the observation, which can be a simple float 
    number (as a level value) or a 1D array. It recieves the values for the prism dimension in X, Y and Z 
    directions.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    potential - numpy array - gravitational potential due to a solid prism
    '''

    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    rho = rho*1000.
    
    # Creating the zeros vector to allocate the result
    potential = numpy.zeros_like(x)
    
    # Solving the integral as a numerical approximation
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = (xp[i]*yp[j]*auxiliars.my_log(zp[k] + r)
                          + yp[j]*zp[k]*auxiliars.my_log(xp[i] + r)
                          + xp[i]*zp[k]*auxiliars.my_log(yp[j] + r)
                          - 0.5*xp[i]**2 *
                          auxiliars.my_atan(zp[k]*yp[j], xp[i]*r)
                          - 0.5*yp[j]**2 *
                          auxiliars.my_atan(zp[k]*xp[i], yp[j]*r)
                          - 0.5*zp[k]**2*auxiliars.my_atan(xp[i]*yp[j], zp[k]*r))
                potential += ((-1.)**(i + j + k))*result*rho
    
    # Multiplying the values for 
    potential *= G
        
    # Return the final output
    return potential

def my_prism_gx(x, y, z, prism, rho):
    '''
    This function is a Python implementation for the X horizontal component for the gravity field due to 
    a rectangular prism, which has initial and final positions equals to xi and xf, yi and yf, for the X 
    and Y directions. This function also recieve the obsevation points for an array or a grid and also the 
    value for height of the observation, which can be a simple float number (as a level value) or a 1D array.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        rho - -float - density value
        
    Output:
    gx - numpy array - vertical component for the gravity atraction
    '''

    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]    
    
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    rho = rho*1000.
    
    # Numpy zeros array to update the result
    gx = numpy.zeros_like(x)
    
    # Compute the value for Gx
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(yp[j]*auxiliars.my_log(zp[k] + r) + 
                           zp[k]*auxiliars.my_log(yp[j] + r) - 
                           xp[i]*auxiliars.my_atan(zp[k]*yp[j], xp[i]*r))
                gx += ((-1.)**(i + j + k))*result*rho

                # Multiplication for all constants and conversion to mGal
    gx *= G*si2mGal
    
    # Return the final output
    return gx

def my_prism_gy(x, y, z, prism, rho):
    '''
    This function is a Python implementation for the Y horizontal  component for the gravity field due 
    to a rectangular prism, which has initial and final positions equals to xi and xf, yi and yf, for 
    the X and Y directions. This function also recieve the obsevation points for an array or a grid and 
    also the value for height of the observation, which can be a simple float number (as a level value) 
    or a 1D array.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    gy - numpy array - vertical component for the gravity atraction
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]  
   
    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0
    rho = rho*1000.
    
    # Numpy zeros array to update the result
    gy = numpy.zeros_like(x)
    
    # Compute the value for Gy
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(zp[k]*auxiliars.my_log(xp[i] + r) + 
                           xp[i]*auxiliars.my_log(zp[k] + r) - 
                           yp[j]*auxiliars.my_atan(xp[i]*zp[k], yp[j]*r))
                gy += ((-1.)**(i + j + k))*result*rho
                
    # Multiplication for all constants and conversion to mGal
    gy *= G*si2mGal
    
    # Return the final output
    return gy

def my_prism_gz(x, y, z, prism, rho):
    '''
    This function is a Python implementation for the vertical component for the gravity field due to a 
    rectangular prism, which has initial and final positions equals to xi and xf, yi and yf, for the X 
    and Y directions. This function also recieve the obsevation points for an array or a grid and also 
    the value for height of the observation, which can be a simple float number (as a level value) or a 
    1D array.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - density value
        
    Output:
    gz - numpy array - vertical component for the gravity atraction
    '''

    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Definitions for all distances
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]  

    # Definition - some constants
    G = 6.673e-11
    si2mGal = 100000.0    
    rho = rho*1000
    # Numpy zeros array to update the result
    gz = numpy.zeros_like(x)
    
    # Compute the value for Gz
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                result = -(xp[i]*auxiliars.my_log(yp[j] + r) + 
                           yp[j]*auxiliars.my_log(xp[i] + r) -
                           zp[k]*auxiliars.my_atan(xp[i]*yp[j], zp[k]*r))
                gz += ((-1.)**(i + j + k))*result*rho
                
    # Multiplication for all constants and conversion to mGal
    gz *= G*si2mGal
    
    # Return the final output
    return gz

def my_3Dpotential(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''
    It calculates the gravitational potential due to a set of prisms. Xp and Yp represent 
    the horizontal positions of all prisms; Zt and Zb repesent the depth of top 
    and bottom of each prism. Dx and Dy are the horizontal dimensions. All space 
    coordinates are in meteres. Rho indicates the density in g/cm^3.
    
    Inputs:
    x, y, z - numpy array - coordinates and level of each observation
    xp, yp - numpy array - horizontal coordinates of each prism
    zt, zb - numpy array - depth of top and bottom of each prism
    dx, dy - float - horizontal dimension of each prism
    rho - numpy array - values of density
    
    Output:
    pot - numpy array - gravity signal of all prisms
    '''
    
    # Assert all conditions for observation points
    if x.shape != y.shape:
        raise ValueError("All observations inputs must have same shape!")
    # Assert all conditions for each prism in horizontal and vertical positions
    if xprism.shape != yprism.shape:
        raise ValueError("All prisms horizontal inputs must have same shape!")
    if top.shape != bottom.shape:
        raise ValueError("All prisms vertical inputs must have same shape!")
        
    # Create the zero array
    pot = numpy.zeros_like(x)
    
    # Condition for density
    if density.shape == xprism.shape:
        for k in range(xprism.size):
            pot += potential(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                       yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                       top[k], bottom[k]), density[k])
    else:
        for k in range(xprism.size):
            pot += prism_potential(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                             yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                             top[k], bottom[k]), density)
    # Return the final output
    return pot

def my_3Dgx(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''
    It calculates the gravity signal due to a set of prisms. Xp and Yp represent 
    the horizontal positions of all prisms; Zt and Zb repesent the depth of top 
    and bottom of each prism. Dx and Dy are the horizontal dimensions. All space 
    coordinates are in meteres. Rho indicates the density in g/cm^3.
    
    Inputs:
    x, y, z - numpy array - coordinates and level of each observation
    xp, yp - numpy array - horizontal coordinates of each prism
    zt, zb - numpy array - depth of top and bottom of each prism
    dx, dy - float - horizontal dimension of each prism
    rho - numpy array - values of density
    
    Output:
    gx - numpy array - gravity signal of all prisms in x direction
    '''
    
    # Assert all conditions for observation points
    if x.shape != y.shape:
        raise ValueError("All observations inputs must have same shape!")
    # Assert all conditions for each prism in horizontal and vertical positions
    if xprism.shape != yprism.shape:
        raise ValueError("All prisms horizontal inputs must have same shape!")
    if top.shape != bottom.shape:
        raise ValueError("All prisms vertical inputs must have same shape!")
        
    # Create the zero array
    gx = numpy.zeros_like(x)
    
    # Condition for density
    if density.shape == xprism.shape:
        for k in range(xprism.size):
            gx += prism_gx(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                     yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                     top[k], bottom[k]), density[k])
    else:
        for k in range(xprism.size):
            gx += prism_gx(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                     yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                     top[k], bottom[k]), density)
            
    # Return the final output
    return gx

def my_3Dgy(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''
    It calculates the gravity signal due to a set of prisms. Xp and Yp represent 
    the horizontal positions of all prisms; Zt and Zb repesent the depth of top 
    and bottom of each prism. Dx and Dy are the horizontal dimensions. All space 
    coordinates are in meteres. Rho indicates the density in g/cm^3.
    
    Inputs:
    x, y, z - numpy array - coordinates and level of each observation
    xp, yp - numpy array - horizontal coordinates of each prism
    zt, zb - numpy array - depth of top and bottom of each prism
    dx, dy - float - horizontal dimension of each prism
    rho - numpy array - values of density
    
    Output:
    gy - numpy array - gravity signal of all prisms in y direction
    '''
    
    # Assert all conditions for observation points
    if x.shape != y.shape:
        raise ValueError("All observations inputs must have same shape!")
    # Assert all conditions for each prism in horizontal and vertical positions
    if xprism.shape != yprism.shape:
        raise ValueError("All prisms horizontal inputs must have same shape!")
    if top.shape != bottom.shape:
        raise ValueError("All prisms vertical inputs must have same shape!")
        
    # Create the zero array
    gy = numpy.zeros_like(x)
    
    # Condition for density
    if density.shape == xprism.shape:
        for k in range(xprism.size):
            gy += prism_gy(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                     yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                     top[k], bottom[k]), density[k])
    else:
        for k in range(xprism.size):
            gy += prism_gy(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                     yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                     top[k], bottom[k]), density)
    # Return the final output
    return gz

def my_3Dgz(x, y, z, xprism, yprism, top, bottom, deltax, deltay, density):
    '''
    It calculates the gravity signal due to a set of prisms. Xp and Yp represent 
    the horizontal positions of all prisms; Zt and Zb repesent the depth of top 
    and bottom of each prism. Dx and Dy are the horizontal dimensions. All space 
    coordinates are in meteres. Rho indicates the density in g/cm^3.
    
    Inputs:
    x, y, z - numpy array - coordinates and level of each observation
    xp, yp - numpy array - horizontal coordinates of each prism
    zt, zb - numpy array - depth of top and bottom of each prism
    dx, dy - float - horizontal dimension of each prism
    rho - numpy array - values of density
    
    Output:
    gz - numpy array - gravity signal of all prisms in z direction
    '''
    
    # Assert all conditions for observation points
    if x.shape != y.shape:
        raise ValueError("All observations inputs must have same shape!")
    # Assert all conditions for each prism in horizontal and vertical positions
    if xprism.shape != yprism.shape:
        raise ValueError("All prisms horizontal inputs must have same shape!")
    if top.shape != bottom.shape:
        raise ValueError("All prisms vertical inputs must have same shape!")
        
    # Create the zero array
    gz = numpy.zeros_like(x)
    
    # Condition for density
    if density.shape == xprism.shape:
        for k in range(xprism.size):
            gz += my_prism_gz(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                     yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                     top[k], bottom[k]), density[k])
    else:
        for k in range(xprism.size):
            gz += my_prism_gz(x, y, z, (xprism[k] - deltax/2., xprism[k] + deltax/2., 
                                     yprism[k] - deltay/2., yprism[k] + deltay/2., 
                                     top[k], bottom[k]), density)
            
    # Return the final output
    return gz

def my_prism_tf(x, y, z, prism, mag, inc, dec, incs = None, decs = None, azim = 0.):
    '''
    This function calculates the total field anomaly produced by a rectangular prism located under 
    surface; it is a Python implementation for the Subroutin MBox which is contained on Blakely (1995). 
    It recieves: the coordinates of the positions in all directions, the elements of the prims, the 
    angle directions and the elements of the field. That function also uses the auxiliarsilary function 
    DIR_COSSINE to calculate the projections due to the field F and the source S.
    
    Inputs:
    x, y - numpy arrays - observation points in x and y directions
    z - numpy array/float - height for the observation
    prism - numpy array - all elements for the prims
        prism[0, 1] - initial and final coordinates at X (dimension at X axis!)
        prism[2, 3] - initial and final coordinates at Y (dimension at Y axis!)
        prism[4, 5] - initial and final coordinates at Z (dimension at Z axis!)
        prism[6] - magnetic intensity
    directions - numpy array - elements for source directions
        directions[0] - float - source inclination
        directions[1] - float - source declination
    field - numpy array - elementes for regional field
        field[0] - float - magnetic field inclination
        field[1] - float - magnetic field declination
        
    Output:
    tfa - numpy array - calculated total field anomaly
    
    X and Y represents North and East; Z is positive downward.
    Ps. Z can be a array with all elements for toppography or a float point as a flight height.
    '''    
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
        
    # Stablish some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    if incs == None:
        incs = inc
    if decs == None:
        decs = dec
  
    # Calculate the directions for the source magnetization and for the field
    Ma, Mb, Mc = auxiliars.my_dircos(incs, decs, azim) # s -> source
    Fa, Fb, Fc = auxiliars.my_dircos(inc, dec, azim) # f -> field

    # Aranges all values as a vector
    MF = [Ma*Fb + Mb*Fa, 
          Ma*Fc + Mc*Fa, 
          Mb*Fc + Mc*Fb, 
          Ma*Fa, 
          Mb*Fb, 
          Mc*Fc]
    
    # Limits for initial and final position along the directions
    A = [prism[1] - x, prism[0] - x]
    B = [prism[3] - y, prism[2] - y]
    H = [prism[5] - z, prism[4] - z]
    
    # Create the zero array to allocate the total field result
    tfa = numpy.zeros_like(x)
    
    # Loop for controling the signal of the function    
    for k in range(2):
        mag *= -1
        H2 = H[k]**2
        for j in range(2):
            Y2 = B[j]**2
            for i in range(2):
                X2 = A[i]**2
                AxB = A[i]*B[j]
                R2 = X2 + Y2 + H2
                R = numpy.sqrt(R2)
                HxR = H[k]*R
                tfa += ((-1.)**(i + j))*mag*(0.5*(MF[2])*auxiliars.my_log((R - A[i])/(R + A[i])) + 0.5*(MF[1])*
                                             auxiliars.my_log((R - B[j])/(R + B[j])) - (MF[0])*auxiliars.my_log(R + H[k]) -
                                             (MF[3])*auxiliars.my_atan(AxB, X2 + HxR + H2) -
                                             (MF[4])*auxiliars.my_atan(AxB, R2 + HxR - X2) +
                                             (MF[5])*auxiliars.my_atan(AxB, HxR))
    # Multiplying for constants conversion
    tfa *= t2nt*cm
    
    # Return the final output
    return tfa

def prism_bx(x, y, z, prism, inc, dec, incs = None, decs = None, azim = 0.):
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
        
    # Stablish some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    # Condition for directions
    if incs == None:
        incs = inc
    if decs == None:
        decs = dec
  
    # Calculate the directions for the source magnetization
    mx, my, mz = auxiliars.my_dircos(incs, decs, azim)
    
    # Create the zero array
    bx = numpy.zeros_like(x)

    # Calculate the x - component
    bx += (kernelxx(x, y, z, prism)*mx + 
           kernelxy(x, y, z, prism)*my + 
           kernelxz(x, y, z, prism)*mz)
    # Conversion
    bx *= cm*t2nt
    
    # Return the final output
    return bx


def prism_by(x, y, z, prism, inc, dec, incs = None, decs = None, azim = 0.):
    # Shape condition
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
        
    # Stablish some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    # Condition for directions
    if incs == None:
        incs = inc
    if decs == None:
        decs = dec
  
    # Calculate the directions for the source magnetization
    mx, my, mz = auxiliars.my_dircos(incs, decs, azim)
    
    # Create the zero array
    by = numpy.zeros_like(x)

    # Calculate the y - component
    by += (kernelxy(x, y, z, prism)*mx + 
           kernelyy(x, y, z, prism)*my + 
           kernelyz(x, y, z, prism)*mz)
    # Conversion
    by *= cm*t2nt
    # Return the final output
    return by

def prism_bz(x, y, z, prism, inc, dec, incs = None, decs = None, azim = 0.):
    # Shape condition
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
        
    # Stablish some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    # Condition for directions
    if incs == None:
        incs = inc
    if decs == None:
        decs = dec
  
    # Calculate the directions for the source magnetization
    mx, my, mz = auxiliars.my_dircos(incs, decs, azim)
    
    # Create the zero array
    bz = numpy.zeros_like(x)

    # Calculate the z - component
    bz += (kernelxz(x, y, z, prism)*mx + 
           kernelyz(x, y, z, prism)*my + 
           kernelzz(x, y, z, prism)*mz)
    # Conversion
    bz *= cm*t2nt
    # Return the final output
    return bz
#------------------------------------------------------------------------------------
def my_kernelxx(x, y, z, prism):
    result = numpy.zeros_like(x)
    # Defines computation points
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = -auxiliars.my_atan(zp[k]*yp[j], xp[i]*r)
                result += ((-1.)**(i + j + k))*kernel
    return result

def my_kernelyy(x, y, z, prism):
    result = numpy.zeros_like(x)
    # Defines computation points
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = -auxiliars.my_atan(zp[k]*xp[i], yp[j]*r)
                result += ((-1.)**(i + j + k))*kernel
    return result

def my_kernelzz(x, y, z, prism):
    result = numpy.zeros_like(x)
    # Defines computation points
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = -auxiliars.my_atan(yp[j]*xp[i], zp[k]*r)
                result += ((-1.)**(i + j + k))*kernel
    return result

def my_kernelxy(x, y, z, prism):
    result = numpy.zeros_like(x)
    # Defines computation points
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = auxiliars.my_log(zp[k] + r)
                result += ((-1.)**(i + j + k))*kernel
    return result

def my_kernelxz(x, y, z, prism):
    result = numpy.zeros_like(x)
    # Defines computation points
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = auxiliars.my_log(yp[j] + r)
                result += ((-1.)**(i + j + k))*kernel
    return result

def my_kernelyz(x, y, z, prism):
    result = numpy.zeros_like(x)
    # Defines computation points
    xp = [prism[1] - x, prism[0] - x]
    yp = [prism[3] - y, prism[2] - y]
    zp = [prism[5] - z, prism[4] - z]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = numpy.sqrt(xp[i]**2 + yp[j]**2 + zp[k]**2)
                kernel = auxiliars.my_log(xp[i] + r)
                result += ((-1.)**(i + j + k))*kernel
    return result 