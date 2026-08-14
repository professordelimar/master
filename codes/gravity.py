from __future__ import division
import numpy
import scipy

def my_WGS84():
    '''    
    This function returns the following parameters defining the reference elipsoid WGS84:
    a = semimajor axis [m]
    f = flattening
    GM = geocentric gravitational constant of the Earth (including the atmosphere) [m**3/s**2]
    omega = angular velocity [rad/s]
    
    Output:
    a, f, GM, omega    
    '''
    a = 6378137.0
    f = 1.0/298.257223563
    GM = 3986004.418e8
    omega = 7292115e-11
    # Return the final output
    return a, f, GM, omega

def my_GRS80():
    '''    
    This function returns the following parameters defining the reference elipsoid GRS80:
    a = semimajor axis [m]
    f = flattening
    GM = geocentric gravitational constant of the Earth (including the atmosphere) [m**3/s**2]
    omega = angular velocity [rad/s]
    
    Output:
    a, f, GM, omega    
    '''
    a = 6378137.0
    f = 1.0/298.257222101
    GM = 3986005.0e8
    omega = 7292115e-11
    
    # Return the final output
    return a, f, GM, omega

def my_somigliana(phi, a = None, f = None, GM = None, omega = None):
    '''    
    This function calculates the normal gravity by using the Somigliana's formula.    
    Inputs:
    a: float containing the semimajor axis [m]
    f: float containing the flattening
    GM: float containing the geocentric gravitational constant of the Earth (including the atmosphere) [m**3/s**2]
    omega: float containing the angular velocity [rad/s]
    phi: array containing the geodetic latitudes [degree]    
    Output:
    gamma: array containing the values of normal gravity on the surface of the elipsoid for each geodetic latitude [mGal]
    '''    
    # WSG84
    if a == None:
        a = 6378137.0
    if f == None:
        f = 1.0/298.257223563
    if GM == None:
        GM = 3986004.418*(10**8)
    if omega == None:
        omega = 7292115*(10**-11)
    # Calculation    
    b = a*(1.0-f)
    a2 = a**2
    b2 = b**2
    E = numpy.sqrt(a2 - b2)
    elinha = E/b
    bE = b/E
    Eb = E/b
    atg = numpy.arctan(Eb)
    q0 = 0.5*((1+3*(bE**2))*atg - (3*bE))
    q0linha = 3.0*(1+(bE**2))*(1-(bE*atg)) - 1
    m = (omega**2)*(a2)*b/GM
    aux = elinha*q0linha/q0
    gammaa = (GM/(a*b))*(1-m-(m/6.0)*aux)
    gammab = (GM/a2)*(1+(m/3.0)*aux)
    aux = numpy.deg2rad(phi)
    s2 = numpy.sin(aux)**2
    c2 = numpy.cos(aux)**2
    # the 10**5 converts from m/s**2 to mGal
    gamma = (10**5)*((a*gammaa*c2) + (b*gammab*s2))/numpy.sqrt((a2*c2) + (b2*s2))
    # Final output
    return gamma
    
def my_closedform(phi, h, a = None, f = None, GM = None, omega = None):
    '''    
    This function calculates the normal gravity by using a closed-form formula.
    
    Reference: Li, X., & Götze, H. J. (2001). Ellipsoid, geoid, gravity, geodesy, and geophysics. Geophysics, 66(6), 1660-1668.
    
    Inputs: 
    a: float containing the semimajor axis [m]
    f: float containing the flattening
    GM: float containing the geocentric gravitational constant of the Earth (including the atmosphere) [m**3/s**-2]
    omega: float containing the angular velocity [rad/s]
    phi: array containing the geodetic latitudes [degree]
    h: array containing the normal heights [m]
    
    Output:
    gamma: array containing the values of normal gravity on the surface of the elipsoid for each geodetic latitude [mGal]
    '''
    # WSG84
    if a == None:
        a = 6378137.0
    if f == None:
        f = 1.0/298.257223563
    if GM == None:
        GM = 3986004.418*(10**8)
    if omega == None:
        omega = 7292115*(10**-11)
    # Calculation
    b = a*(1.0-f)
    a2 = a**2
    b2 = b**2
    E = numpy.sqrt(a2 - b2)
    E2 = E**2
    bE = b/E
    Eb = E/b
    atanEb = numpy.arctan(Eb)
    phirad = numpy.deg2rad(phi)
    tanphi = numpy.tan(phirad)
    cosphi = numpy.cos(phirad)
    sinphi = numpy.sin(phirad)
    beta = numpy.arctan(b*tanphi/a)
    sinbeta = numpy.sin(beta)
    cosbeta = numpy.cos(beta)
    zl = b*sinbeta+h*sinphi
    rl = a*cosbeta+h*cosphi
    zl2 = zl**2
    rl2 = rl**2
    dll2 = rl2-zl2
    rll2 = rl2+zl2
    D = dll2/E2
    R = rll2/E2
    cosbetal = numpy.sqrt(0.5*(1+R) - numpy.sqrt(0.25*(1+R**2) - 0.5*D))
    cosbetal2 = cosbetal**2
    sinbetal2 = 1-cosbetal2
    bl = numpy.sqrt(rll2 - E2*cosbetal2)
    bl2 = bl**2
    blE = bl/E
    Ebl = E/bl
    atanEbl = numpy.arctan(Ebl)
    q0 = 0.5*((1+3*(bE**2))*atanEb - (3*bE))
    q0l = 3.0*(1+(blE**2))*(1-(blE*atanEbl)) - 1
    W = numpy.sqrt((bl2+E2*sinbetal2)/(bl2+E2))
    gamma = GM/(bl2+E2) - cosbetal2*bl*omega**2
    gamma += (((omega**2)*a2*E*q0l)/((bl2+E2)*q0))*(0.5*sinbetal2 - 1./6.)
    # the 10**5 converts from m/s**2 to mGal
    gamma = (10**5)*gamma/W
    # Final output
    return gamma

def my_freeair_correction(orthometric):
    '''
    It calculates the freeair correction and return the freeair anomaly.
    
    Input: 
    orthometric - 1D numpy array - orthometric height
    
    Output:
    fac - 1D numpy array - factor of freeair correction
    '''
    # Final output
    return -0.3086*orthometric

def my_bouguer_correction(topography, rho_crust = 2673., rho_oceanic = 2950., rho_water = 1040.):
    '''
    It calculates the Bouguer reduction factor with the Bouguer plateur calculation.
    
    Input:
    topo - 1D array - topography data
    
    Output:
    bgc - 1D array - Bouguer correction    
    '''
    # Setting positive and negative topography
    ocean = numpy.array(topography < 0)
    continent = numpy.array(topography > 0)
    # Calculate the Bouguer reduction
    # Over continents
    bgc_cont = numpy.zeros_like(topography)
    bgc_cont[continent] = 2.*numpy.pi*6.67408e-11*1.0e5*rho_crust*topography[continent]
    # Over oceans
    bgc_ocea = numpy.zeros_like(topography)
    bgc_ocean[ocean] = 2.*nupy.pi*6.67408e-11*1.0e5*(rho_oceanic - rho_water)*topography[ocean]
    # Final output
    return bgc_ocean + bgc_cont

def my_Airy(topography, rho_mantle = 3270., rho_crust = 2673., rho_oceanic = 2950., rho_water = 1040.):
    '''
    It calculates the isostatic crustal root depth based on Airy's hypothesis.
    If user wants to calculate the Moho depth, it should be added a reference depth.
    Hint: Use 30000 m.
    
    Input: 
    topography - 1D array - topography data
    rho_mantle - float - density of the mantle
    rho_crust - float - density of continental crust
    rho_oceanic - float - density of oceanic crust
    rho_water - float - density of the water
    
    Output:
    rooth - 1D array - isostatic Moho depth
    '''
    # Setting positive and negative topography
    ocean = numpy.array(topography < 0)
    continent = numpy.array(topography > 0)
    # Calculating the Moho depth
    # Over continents
    continental_thickness = numpy.zeros_like(topography)
    continental_thickness[continent] = rho_crust*topography[continent]/(rho_mantle - rho_crust)
    # Over oceans
    oceanic_thickness = numpy.zeros_like(topography)
    oceanic_thickness[ocean] = (rho_oceanic - rho_water)*topography[ocean]/(rho_mantle - rho_oceanic)
    # Final output
    return continental_thickness + oceanic_thickness

def my_isostatic_correction(topography, rho_crust = 2673., rho_oceanic = 2950., rho_water = 1040.):
    '''
    It calculates the isostatic reduction factor with the Bouguer plateur approximation for calculation.
    
    Input:
    topo - 1D array - topography data
    
    Output:
    isostatic - 1D array - isostatic correction    
    '''
    # Setting positive and negative topography
    ocean = numpy.array(topography < 0)
    continent = numpy.array(topography > 0)
    # Calculate the isostatic reduction
    # Over continents
    isostatic_cont = numpy.zeros_like(topography)
    isostatic_cont[continent] = 2.*numpy.pi*6.67408e-11*1.0e5*rho_crust*topography[continent]
    # Over oceans
    isostatic_ocea = numpy.zeros_like(topography)
    isostatic_ocean[ocean] = 2.*nupy.pi*6.67408e-11*1.0e5*(rho_oceanic - rho_water)*topography[ocean]
    # Final output
    return isostatic_ocean + isostatic_cont