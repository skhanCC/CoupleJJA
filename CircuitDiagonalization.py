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
def JJAResClosedEigs(Nx, L, C, Cg, nEig=50, bc="neu"):
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

    chiC = C/(Cg + 2*C)
    w0 = 1/np.sqrt(L* (Cg + 2*C))
    
    # Construct (Inverse) Inductance matrix
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    diagVals = np.array([e, -chiC*e, -chiC*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)

    if bc == "neu":
        # Set boundary elements - Neumann boundary conditions
        L1D[0,0], L1D[-1,-1] = 1, 1
        # Set boundary elements - Neumann boundary conditions
        C1D[0,0], C1D[-1,-1] = 1 - chiC, 1- chiC
     
    # Solve generalized eigenvalue problem
    spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, which='SM')

    # Sort eigenvalues and eigenvectors
    sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
    spEigVals = spEigVals[sIVec]
    spEigVecs = spEigVecs[:,sIVec]
   
    
    return np.sqrt(spEigVals), spEigVecs
##################
# Dispersion relation
#def wk(k):
#return np.sqrt((1 - np.cos(np.pi* k/Nx))(Cg/(2*C)+ 1 - np.cos(np.pi* k/Nx)))


##############################################################################

# Function to solve boundary value problem for JJA resonator capacitively coupled 
# to open transmission lines at both ends
def JJAResOpenEigs(Nx, L, C, Cg, LW, CW, Cin, Cout, kRefVec, errTol=1e-5, itNum=5, disp="on"):
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
    LW
        Transmission line inductance
    CW
        Transmission line capacitance to ground
    Cin
        Coupling capacitance between left transmission line and JJA resonator
    Cout
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
    
    chiC = C/(Cg + 2*C)
    chiIn = Cin/(Cg + 2*C)
    chiOut = Cout/(Cg + 2*C)
    
    # Vector of ones
    e = np.ones( (Nx,), dtype=complex )

    # Construct (Inverse) Inductance matrix
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    diagVals = np.array([e, -chiC*e, -chiC*e])
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
            ζL = Cin/( -(kRef**2)*(CW+Cin) + (1/LW)*(1-βL) )
            βR = (2+1j*kRef)/(2-1j*kRef) # Outgoing wavevector term
            ζR = Cout/( -(kRef**2)*(CW+Cout) + (1/LW)*(1-βR) )

            # Implement outgoing boundary conditions - left side resonator EOM
            C1D[0,0]   = 1- chiC+chiIn-chiIn*(kRef**2)*ζL
            L1D[0,0]   = 1

            # Implement outgoing boundary conditions - right side resonator EOM
            C1D[-1,-1] = 1- chiC+chiOut-chiOut*(kRef**2)*ζR
            L1D[-1,-1] = 1

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
def JJAResOpenIterEigs(Nx, L, C, Cg, LW, CW, CL, CR, nEig=40, errTol=1e-5, itNum=5, capItNum=5, disp="on"):
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
    LW
        Transmission line inductance
    CW
        Transmission line capacitance to ground
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
    CLVec = np.linspace(0, Cin, capItNum)
    CRVec = np.linspace(0, Cout, capItNum)
    
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
        refVals, refVecs, _ = JJAResOpenEigs(Nx, L, C, Cg, LW, CW, CLn, CRn, np.real(refVals), errTol=errTol, itNum=itNum, disp=disp)
    
    
    # Return final eigenvalues
    eigVals = refVals
    eigVecs = refVecs
    
    return eigVals, eigVecs


# ##############################################################################


# Function to solve boundary value problem for two JJA resonators coupled capacitively via a JJ with Dirichlet or Neumann boundary conditions in the Even-Odd basis
def JJACoupledClosedEigs(Nx, L, C, Cg, wJ, Csh, nEig=40, bc="neu"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size (forcedly even)
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
    NN = round(Nx/2)
    
    # Vector of ones
    e = np.ones( (NN,), dtype=complex )

    # Check boundary conditions, print error message if incorrect
    assert bc == "neu" or bc == "dir", "Incorrect boundary conditions requested!"
    
    hbar = 1.0545718E-34
    el = 1.60217662E-19
    LJ = hbar/(4*wJ*el**2)
    chiC = C/(Cg + 2*C)
    chiCsh = Csh/(Cg + 2*C)
    chiLj = L/LJ
    
    
    # Construct (Inverse) Inductance matrix EVEN
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix EVEN
    diagVals = np.array([e, -chiC*e, -chiC*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    C1D = sp.sparse.csr_matrix(C1D)

    if bc == "neu":
        # Set boundary elements - Neumann boundary conditions
        L1D[0,0], L1D[-1,-1] = 1, 1
        # Set boundary elements - Neumann boundary conditions
        C1D[0,0], C1D[-1,-1] = 1- chiC, 1- chiC 
        
    
    ########
    #ODD
    
    # Construct (Inverse) Inductance matrix ODD
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L2D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    L2D = sp.sparse.csr_matrix(L2D)

    # Construct capacitance matrix ODD
    diagVals = np.array([e, -chiC*e, -chiC*e])
    diagPos = np.array([0, -1, +1])
    C2D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    C2D = sp.sparse.csr_matrix(C2D)

    if bc == "neu":
        # Set boundary elements - Neumann boundary conditions
        L2D[0,0], L2D[-1,-1] = 1+chiLj, 1+chiLj
        # Set boundary elements - Neumann boundary conditions
        C2D[0,0], C2D[-1,-1] = 1- chiC+ chiCsh, 1- chiC+ chiCsh

        
    # Solve generalized eigenvalue problem
    spEigVals1, spEigVecs1 = ssla.eigs(L1D, k=nEig, M=C1D, which='SM')
    # Solve generalized eigenvalue problem
    spEigVals2, spEigVecs2 = ssla.eigs(L2D, k=nEig, M=C2D, which='SM')

    # Sort eigenvalues and eigenvectors
    sIVec1 = np.argsort( np.real(np.sqrt(spEigVals1)) )
    spEigVals1 = spEigVals1[sIVec1]
    spEigVecs1 = spEigVecs1[:,sIVec1]
    
    # Sort eigenvalues and eigenvectors
    sIVec2 = np.argsort( np.real(np.sqrt(spEigVals2)) )
    spEigVals2 = spEigVals2[sIVec2]
    spEigVecs2 = spEigVecs2[:,sIVec2]
    return np.sqrt(spEigVals1),np.sqrt(spEigVals2), spEigVecs1, spEigVecs2



# Function to solve boundary value problem for two coupled JJA resonators capacitively coupled 
# to open transmission lines at both ends
def JJACoupledOpenEigs(Nx, L, C, Cg,wJ, Csh, LW, CW, Cin, Cout, kRefVec1,kRefVec2, errTol=1e-5, itNum=5, disp="on"):
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
    LW
        Transmission line inductance
    CW
        Transmission line capacitance to ground
    Cin
        Coupling capacitance between left transmission line and JJA resonator
    Cout
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
    NN = round (Nx/2)
    
    # Number of eigenvalues to be computed
    nCom = len(kRefVec1)
    
    # Storage vectors and matrices
    eigVecs1 = np.zeros( (NN,nCom), dtype=complex )
    eigVals1 = np.zeros( (nCom,), dtype=complex )
    errVals1 = np.zeros( (nCom,) )
    
    eigVecs2 = np.zeros( (NN,nCom), dtype=complex )
    eigVals2 = np.zeros( (nCom,), dtype=complex )
    errVals2 = np.zeros( (nCom,) )
    
    hbar = 1.0545718E-34
    el = 1.60217662E-19
    LJ = hbar/(4*wJ*el**2)
    chiC = C/(Cg + 2*C)
    chiCsh = Csh/(Cg + 2*C)
    chiLj = L/LJ
    chiCin = Cin/(Cg + 2*C)
    chiCout = Cout/(Cg + 2*C)
    
    # Vector of ones
    e = np.ones( (NN,), dtype=complex )

    # Construct (Inverse) Inductance matrix EVEN
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix EVEN
    diagVals = np.array([e, -chiC*e, -chiC*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    C1D = sp.sparse.csr_matrix(C1D)
    
    # Construct (Inverse) Inductance matrix ODD
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L2D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    L2D = sp.sparse.csr_matrix(L2D)

    # Construct capacitance matrix ODD
    diagVals = np.array([e, -chiC*e, -chiC*e])
    diagPos = np.array([0, -1, +1])
    C2D = sp.sparse.spdiags(diagVals, diagPos, NN, NN)
    C2D = sp.sparse.csr_matrix(C2D)

    # Set boundary elements
    L1D[0,0], L1D[-1,-1] = 1, 1
    L2D[0,0], L2D[-1,-1] = 1, 1 + chiLj
    
    C1D[0,0], C2D[0,0] = 1- chiC, 1- chiC+ chiCsh
    
    # Loop over eigenvalues to be computed
    for j in range(nCom):
    
        # Set frequency reference
        kRef1 = kRefVec1[j]
        kRef2 = kRefVec2[j]
        
        # Set error 
        err1 = 1
        err2 = 1
        
        # Loop over iterations
        n = 0 
        while err1 > errTol and n <= itNum:

            # Factor for implementing outgoing boundary condition
            βL = (2+1j*kRef1)/(2-1j*kRef1) # Outgoing wavevector term
            ζL = Cin/( -(kRef1**2)*(CW+Cin) + (1/LW)*(1-βL) )
            
            # Implement outgoing boundary conditions - left side resonator EOM
            C1D[-1,-1]   = 1- chiC+chiCin-chiCin*(kRef1**2)*ζL

            
            # Solve generalized eigenvalue problem
            nEig = 1
            spEigVals1, spEigVecs1 = ssla.eigs(L1D, k=nEig, M=C1D, sigma=(kRef1**2) )
            
            # Sort eigenvalues and eigenvectors
            sIVec1 = np.argsort( np.real(np.sqrt(spEigVals1)) )
            spEigVals1 = spEigVals1[sIVec1]
            spEigVecs1 = spEigVecs1[:,sIVec1]
            kVals1 = np.sqrt(spEigVals1)
            
            
            # Calculate error and set new reference
            err1 = np.abs( kRef1 - np.real(kVals1[0]) )
            
            kRef1 = np.real(kVals1[0])
            
            if disp == "on":
                print('Mode: ' + str(j) + ', error: ' + str(err1) + ', it. num.: ' + str(n) )
            # Increment iteration
            n = n + 1
            
             # Loop over iterations
        n = 0 
        while err2 > errTol and n <= itNum:

            # Factor for implementing outgoing boundary condition
            βR = (2+1j*kRef2)/(2-1j*kRef2) # Outgoing wavevector term
            ζR = Cout/( -(kRef2**2)*(CW+Cout) + (1/LW)*(1-βR) )

            # Implement outgoing boundary conditions - right side resonator EOM
            C2D[-1,-1] = 1- chiC+chiCout-chiCout*(kRef2**2)*ζR

            # Solve generalized eigenvalue problem
            nEig = 1
            spEigVals2, spEigVecs2 = ssla.eigs(L2D, k=nEig, M=C2D, sigma=(kRef2**2) )
            
            sIVec2 = np.argsort( np.real(np.sqrt(spEigVals2)) )
            spEigVals2 = spEigVals2[sIVec2]
            spEigVecs2 = spEigVecs2[:,sIVec2]
            kVals2 = np.sqrt(spEigVals2)
            
            # Calculate error and set new reference
            err2 = np.abs( kRef2 - np.real(kVals2[0]) )
            
            kRef2 = np.real(kVals2[0])
            
            if disp == "on":
                print('Mode: ' + str(j) + ', error: ' + str(err2) + ', it. num.: ' + str(n) )
            # Increment iteration
            n = n + 1
            
        # Store computed eigenvalue, eigenvector, and convergence error
        eigVals2[j] = kVals2[0]
        errVals2[j] = err2
        eigVecs2[:,j] = spEigVecs2[:,0]
        
    # Return eigenvalue and eigenvector
    return eigVals1,eigVals2, eigVecs1,eigVecs2, errVals1,errVals2

    
##############################################################################


# Function to solve boundary value problem for two JJA resonators capacitively coupled 
# to open transmission lines at both ends. Solves for eigenvalues iteratively by 
# starting from closed resonator case
def JJACoupledOpenIterEigs(Nx, L, C, Cg, wJ, Csh, LW, CW, Cin, Cout, nEig=40, errTol=1e-5, itNum=5, capItNum=5, disp="on"):
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
    LW
        Transmission line inductance
    CW
        Transmission line capacitance to ground
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
    CLVec = np.linspace(0, Cin, capItNum)
    CRVec = np.linspace(0, Cout, capItNum)
    
    # Find two closed resonators eigenvalues
    refVals1, refVals2,_,_ = JJACoupledClosedEigs(Nx, L, C, Cg, wJ, Csh, nEig=nEig)
    refVals1, refVals2 = refVals1[1:], refVals2[1:]
    
    # Loop over vectors of end capacitances
    for n in range(1,capItNum):
        
        # Set current value of end capacitances
        CLn, CRn = CLVec[n], CRVec[n]
        
        if disp == "on":
            print('Solving for CL: ' + str(CLn) + ', CR: ' + str(CRn))
        
        # Find open resonator eigenvalues
        refVals1,refVals2, refVecs1,  refVecs2, _, _ = JJACoupledOpenEigs(Nx, L, C, Cg, wJ, Csh, LW, CW, CLn, CRn, np.real(refVals1), np.real(refVals2), errTol=errTol, itNum=itNum, disp=disp)
    
    
    # Return final eigenvalues
    eigVals1 = refVals1
    eigVecs1 = refVecs1
    eigVals2 = refVals2
    eigVecs2 = refVecs2
    
    return eigVals1,  eigVals2, eigVecs1, eigVecs2