# Preamble
import numpy as np
import scipy as sp

import scipy.linalg as sla
import scipy.sparse.linalg as ssla

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm

##############################################################################

# Function to solve boundary value problem for JJA resonator capacitively with 
# Dirichlet or Neumann boundary conditions
def JJAResClosedEigs(Nx, L, C, Cg, nEig=40, bc="neu"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size
    L
        JJA resonator inductance
    C
        JJA resonator coupling capacitance
    Cg 
        JJA resonator coupling to ground
    nEig
        Number of eigenvalues to be computed
    BC
        Parameter controlling boundary condition to be imposed. 
        Can be "dir" (Dirichlet) or "neu" (Neumann)
  
    Returns:
    --------
    spEigVals
        Computed eigenvalues
    spEigVecs
        Computed eigenvectors
    """
    
    # Vector of ones
    e = np.ones( (Nx,), dtype=complex )

    # Check boundary conditions, print error message if incorrect
    assert bc == "neu" or bc == "dir", "Incorrect boundary conditions requested!"

    # Construct (Inverse) Inductance matrix
    diagVals = np.array([2*e/L, -e/L, -e/L])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    diagVals = np.array([(Cg+2*C)*e, -C*e, -C*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)

    if bc == "neu":
        # Set boundary elements - Neumann boundary conditions
        L1D[0,0], L1D[-1,-1] = 1/L, 1/L
        # Set boundary elements - Neumann boundary conditions
        C1D[0,0], C1D[-1,-1] = Cg+C, Cg+C
     
    # Solve generalized eigenvalue problem
    spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, which='SM')

    # Sort eigenvalues and eigenvectors
    sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
    spEigVals = spEigVals[sIVec]
    spEigVecs = spEigVecs[:,sIVec]
    
    return np.sqrt(spEigVals), spEigVecs


##############################################################################

# Function to solve boundary value problem for JJA resonator capacitively coupled 
# to open transmission lines at both ends
def JJAResOpenEigs(Nx, L, C, Cg, l, c, CL, CR, kRefVec, errTol=1e-5, itNum=5, disp="on"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size
    L
        JJA resonator inductance
    C
        JJA resonator coupling capacitance
    Cg 
        JJA resonator coupling to ground
    l
        Left/right transmission line inductance
    c
        Left/right transmission line capacitance to ground
    CL
        Coupling capacitance between left transmission line and JJA resonator
    CR
        Coupling capacitance between right transmission line and JJA resonator
    kRefVec
        Vector of reference frequencies for eigenfunction calculation
    errTol
        Tolerance for iterative solver convergnce
    itNum
        Maximum number of iterations for eigenvalue solver
    disp
        Parameter to control printing of status messages. Either "on" or "off"
   
    Returns:
    --------
    eigVals
        Computed eigenvalues
    eigVecs
        Computed eigenvectors
    errVals
        Errors in convergence of computed eigenvalues
    """
    
    # Number of eigenvalues to be computed
    nCom = len(kRefVec)
    
    # Storage vectors and matrices
    eigVecs = np.zeros( (Nx,nCom), dtype=complex )
    eigVals = np.zeros( (nCom,), dtype=complex )
    errVals = np.zeros( (nCom,) )
    
    # Vector of ones
    e = np.ones( (Nx,), dtype=complex )

    # Construct (Inverse) Inductance matrix
    diagVals = np.array([2*e/L, -e/L, -e/L])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    diagVals = np.array([(Cg+2*C)*e, -C*e, -C*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)

    # Loop over eigenvalues to be computed
    for j in range(nCom):
    
        # Set frequency reference
        kRef = kRefVec[j]
        
        # Set error 
        err = 1
        # Loop over iterations
        n = 0 
        while err > errTol and n <= itNum:

            # Factor for implementing outgoing boundary condition
            βL = (2+1j*kRef)/(2-1j*kRef) # Outgoing wavevector term
            ζL = CL/( -(kRef**2)*(c+CL) + (1/l)*(1-βL) )
            βR = (2+1j*kRef)/(2-1j*kRef) # Outgoing wavevector term
            ζR = CR/( -(kRef**2)*(c+CR) + (1/l)*(1-βR) )

            # Implement outgoing boundary conditions - left side resonator EOM
            C1D[0,0]   = Cg+C+CL+CL*(kRef**2)*ζL
            L1D[0,0]   = 1/L

            # Implement outgoing boundary conditions - right side resonator EOM
            C1D[-1,-1] = Cg+C+CR+CR*(kRef**2)*ζR
            L1D[-1,-1] = 1/L

            # Solve generalized eigenvalue problem
            nEig = 1
            spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, sigma=(kRef**2) )

            # Sort eigenvalues and eigenvectors
            sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
            spEigVals = spEigVals[sIVec]
            spEigVecs = spEigVecs[:,sIVec]
            kVals = np.sqrt(spEigVals)
    
            # Calculate error and set new reference
            err = np.abs( kRef - np.real(kVals[0]) )
            kRef = np.real(kVals[0])
            if disp == "on":
                print('Mode: ' + str(j) + ', error: ' + str(err) + ', it. num.: ' + str(n) )
    
            # Increment iteration
            n = n + 1
            
        # Store computed eigenvalue, eigenvector, and convergence error
        eigVals[j] = kVals[0]
        errVals[j] = err
        eigVecs[:,j] = spEigVecs[:,0]
    
        
    # Return eigenvalue and eigenvector
    return eigVals, eigVecs, errVals

    
##############################################################################


# Function to solve boundary value problem for JJA resonator capacitively coupled 
# to open transmission lines at both ends. Solves for eigenvalues iteratively by 
# starting from closed resonator case
def JJAResOpenIterEigs(Nx, L, C, Cg, l, c, CL, CR, nEig=40, errTol=1e-5, itNum=5, capItNum=5, disp="on"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends. Starts with closed resonator and iteratively reaches
    open resonator solution.
    
    Parameters:
    -----------
    Nx
        Grid size
    L
        JJA resonator inductance
    C
        JJA resonator coupling capacitance
    Cg 
        JJA resonator coupling to ground
    l
        Left/right transmission line inductance
    c
        Left/right transmission line capacitance to ground
    CL
        Coupling capacitance between left transmission line and JJA resonator
    CR
        Coupling capacitance between right transmission line and JJA resonator
    nEig
        Number of eigenvalues to be found
    errTol
        Tolerance for iterative solver convergnce
    itNum
        Maximum number of iterations for eigenvalue solver
    capItNum
        Number of iterations over end capacitances
    disp
        Parameter to control printing of status messages. Either "on" or "off"
   
    Returns:
    --------
    eigVals
        Computed eigenvalues
    eigVecs
        Computed eigenvectors
    """
    
    # Vectors for iteration over end capacitances
    CLVec = np.linspace(0, CL, capItNum)
    CRVec = np.linspace(0, CR, capItNum)
    
    # Find closed resonator eigenvalues
    refVals, _ = JJAResClosedEigs(Nx, L, C, Cg, nEig=nEig)
    refVals = refVals[1:]
    
    # Loop over vectors of end capacitances
    for n in range(1,capItNum):
        
        # Set current value of end capacitances
        CLn, CRn = CLVec[n], CRVec[n]
        
        if disp == "on":
            print('Solving for CL: ' + str(CLn) + ', CR: ' + str(CRn))
        
        # Find open resonator eigenvalues
        refVals, refVecs, _ = JJAResOpenEigs(Nx, L, C, Cg, l, c, CLn, CRn, np.real(refVals), errTol=errTol, itNum=itNum, disp=disp)
    
    
    # Return final eigenvalues
    eigVals = refVals
    eigVecs = refVecs
    
    return eigVals, eigVecs


##############################################################################


# Function to solve boundary value problem for two JJA resonators coupled capacitively via a JJ with Dirichlet or Neumann boundary conditions
def JJACoupledResClosedEigs(Nx, L, C, Cg, LJ, Csh, nEig=40, bc="neu"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size (preferably even)
    L
        JJA resonator inductance
    C
        JJA resonator coupling capacitance
    Cg 
        JJA resonator coupling to ground
    LJ
        Coupling Junction inductance
    Csh
        Coupling Junction capacitance
    nEig
        Number of eigenvalues to be computed
    BC
        Parameter controlling boundary condition to be imposed. 
        Can be "dir" (Dirichlet) or "neu" (Neumann)
  
    Returns:
    --------
    spEigVals
        Computed eigenvalues
    spEigVecs
        Computed eigenvectors
    """
    
    assert np.mod(Nx,2) == 0, "Nx should be even!"
    
    # Vector of ones
    e = np.ones( (Nx,), dtype=complex )

    # Check boundary conditions, print error message if incorrect
    assert bc == "neu" or bc == "dir", "Incorrect boundary conditions requested!"

    # Construct (Inverse) Inductance matrix
    diagVals = np.array([2*e/L, -e/L, -e/L])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    diagVals = np.array([(Cg+2*C)*e, -C*e, -C*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)

    if bc == "neu":
        # Set boundary elements - Neumann boundary conditions
        L1D[0,0], L1D[-1,-1] = 1/L, 1/L
        # Set boundary elements - Neumann boundary conditions
        C1D[0,0], C1D[-1,-1] = Cg+C, Cg+C
        
        
    # Inducatance matrix coupling via the junction
    L1D[Nx/2-1,Nx/2+1-1] = -1/LJ
    L1D[Nx/2+1-1,Nx/2+1] = -1/LJ
    L1D[Nx/2-1,Nx/2-1] = 1/L+1/LJ
    L1D[Nx/2+1-1,Nx/2+1-1] = 1/L+1/LJ
     
    # Capacitance matrix coupling via junction
    C1D[Nx/2-1,Nx/2+1-1] = -Csh
    C1D[Nx/2+1-1,Nx/2+1] = -Csh
    C1D[Nx/2-1,Nx/2-1] = Cg+C+Csh
    C1D[Nx/2+1-1,Nx/2+1-1] = Cg+C+Csh
        
    # Solve generalized eigenvalue problem
    spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, which='SM')

    # Sort eigenvalues and eigenvectors
    sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
    spEigVals = spEigVals[sIVec]
    spEigVecs = spEigVecs[:,sIVec]
    
    return np.sqrt(spEigVals), spEigVecs


##############################################################################


# Function to solve boundary value problem for JJA resonator capacitively coupled 
# to open transmission lines at both ends
def JJACoupledResOpenEigs(Nx, L, C, Cg, LJ, Csh, l, c, CL, CR, kRefVec, errTol=1e-5, itNum=5, disp="on"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size
    L
        JJA resonator inductance
    C
        JJA resonator coupling capacitance
    Cg 
        JJA resonator coupling to ground
    LJ
        Coupling Junction inductance
    Csh
        Coupling Junction capacitance
    l
        Left/right transmission line inductance
    c
        Left/right transmission line capacitance to ground
    CL
        Coupling capacitance between left transmission line and JJA resonator
    CR
        Coupling capacitance between right transmission line and JJA resonator
    kRefVec
        Vector of reference frequencies for eigenfunction calculation
    errTol
        Tolerance for iterative solver convergnce
    itNum
        Maximum number of iterations for eigenvalue solver
    disp
        Parameter to control printing of status messages. Either "on" or "off"
   
    Returns:
    --------
    eigVals
        Computed eigenvalues
    eigVecs
        Computed eigenvectors
    errVals
        Errors in convergence of computed eigenvalues
    """
    
    assert np.mod(Nx,2) == 0, "Nx should be even!"
    
    # Number of eigenvalues to be computed
    nCom = len(kRefVec)
    
    # Storage vectors and matrices
    eigVecs = np.zeros( (Nx,nCom), dtype=complex )
    eigVals = np.zeros( (nCom,), dtype=complex )
    errVals = np.zeros( (nCom,) )
    
    # Vector of ones
    e = np.ones( (Nx,), dtype=complex )

    # Construct (Inverse) Inductance matrix
    diagVals = np.array([2*e/L, -e/L, -e/L])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    diagVals = np.array([(Cg+2*C)*e, -C*e, -C*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)

    # Inducatance matrix coupling via the junction
    L1D[Nx/2-1,Nx/2+1-1] = -1/LJ
    L1D[Nx/2+1-1,Nx/2+1] = -1/LJ
    L1D[Nx/2-1,Nx/2-1] = 1/L+1/LJ
    L1D[Nx/2+1-1,Nx/2+1-1] = 1/L+1/LJ
     
    # Capacitance matrix coupling via junction
    C1D[Nx/2-1,Nx/2+1-1] = -Csh
    C1D[Nx/2+1-1,Nx/2+1] = -Csh
    C1D[Nx/2-1,Nx/2-1] = Cg+C+Csh
    C1D[Nx/2+1-1,Nx/2+1-1] = Cg+C+Csh
    
    # Loop over eigenvalues to be computed
    for j in range(nCom):
    
        # Set frequency reference
        kRef = kRefVec[j]
        
        # Set error 
        err = 1
        # Loop over iterations
        n = 0 
        while err > errTol and n <= itNum:

            # Factor for implementing outgoing boundary condition
            βL = (2+1j*kRef)/(2-1j*kRef) # Outgoing wavevector term
            ζL = CL/( -(kRef**2)*(c+CL) + (1/l)*(1-βL) )
            βR = (2+1j*kRef)/(2-1j*kRef) # Outgoing wavevector term
            ζR = CR/( -(kRef**2)*(c+CR) + (1/l)*(1-βR) )

            # Implement outgoing boundary conditions - left side resonator EOM
            C1D[0,0]   = Cg+C+CL+CL*(kRef**2)*ζL
            L1D[0,0]   = 1/L

            # Implement outgoing boundary conditions - right side resonator EOM
            C1D[-1,-1] = Cg+C+CR+CR*(kRef**2)*ζR
            L1D[-1,-1] = 1/L

            # Solve generalized eigenvalue problem
            nEig = 1
            spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, sigma=(kRef**2) )

            # Sort eigenvalues and eigenvectors
            sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
            spEigVals = spEigVals[sIVec]
            spEigVecs = spEigVecs[:,sIVec]
            kVals = np.sqrt(spEigVals)
    
            # Calculate error and set new reference
            err = np.abs( kRef - np.real(kVals[0]) )
            kRef = np.real(kVals[0])
            if disp == "on":
                print('Mode: ' + str(j) + ', error: ' + str(err) + ', it. num.: ' + str(n) )
    
            # Increment iteration
            n = n + 1
            
        # Store computed eigenvalue, eigenvector, and convergence error
        eigVals[j] = kVals[0]
        errVals[j] = err
        eigVecs[:,j] = spEigVecs[:,0]
    
        
    # Return eigenvalue and eigenvector
    return eigVals, eigVecs, errVals


##############################################################################

# Function to normalize eigenvectors
def normalizeEigVecs(eigVecs, dx):
    
    # Loop over eigenvectors
    for n in range(np.shape(eigVecs)[1]):
        # Compute norm
        prod = np.matmul( np.conj(np.transpose(eigVecs[:,n])), eigVecs[:,n] )
        norm = 1/np.sqrt(dx*prod)
        eigVecs[:,n] = norm*eigVecs[:,n]
    
    return eigVecs


##############################################################################
