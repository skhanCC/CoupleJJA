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
# Function to solve boundary value problem for JJA resonator array with or without JJ defect 
# with Dirichlet or Neumann boundary conditions
def JJAEigs(Nx, Cjjagnd, Ljja, Cjja, Ljjdefect, Cjjdefect, nEig, bc="neu", disp="on"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size
    Ljja
        JJA resonator inductance
    Cjja
        JJA resonator coupling capacitance
    Cjjagnd 
        JJA resonator coupling to ground
    Ljjdefect
        Defect JJ resonator inductance in the middle
    Cjjdefect
        Defect JJ resonator shunting capacitance in the middle
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
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix
    χc = Cjja/(Cjjagnd+2*Cjja)
    diagVals = np.array([e, -χc*e, -χc*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)
    if bc == "neu":
        # Set boundary elements - Neumann boundary conditions
        L1D[0,0], L1D[-1,-1] = 1, 1
        # Set boundary elements - Neumann boundary conditions
        C1D[0,0], C1D[-1,-1] = (Cjjagnd+Cjja)/(Cjjagnd+2*Cjja), (Cjjagnd+Cjja)/(Cjjagnd+2*Cjja)

    # Inductance matrix coupling via the junction
    L1D[Nx/2-1,Nx/2+1-1] = -Ljja/Ljjdefect
    L1D[Nx/2+1-1,Nx/2-1] = -Ljja/Ljjdefect
    L1D[Nx/2-1,Nx/2-1] = 1 + Ljja/Ljjdefect
    L1D[Nx/2+1-1,Nx/2+1-1] = 1 + Ljja/Ljjdefect
     
    # Capacitance matrix coupling via junction
    C1D[Nx/2-1,Nx/2+1-1] = -Cjjdefect/(Cjjagnd + 2*Cjja)
    C1D[Nx/2+1-1,Nx/2-1] = -Cjjdefect/(Cjjagnd + 2*Cjja)
    C1D[Nx/2-1,Nx/2-1] = (Cjjagnd+Cjja+Cjjdefect)/(Cjjagnd + 2*Cjja)
    C1D[Nx/2+1-1,Nx/2+1-1] = (Cjjagnd+Cjja+Cjjdefect)/(Cjjagnd + 2*Cjja)     
    
    # Solve generalized eigenvalue problem
    spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, which='SM')

    # Sort eigenvalues and eigenvectors
    sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
    spEigVals = spEigVals[sIVec]
    spEigVecs = spEigVecs[:,sIVec]
    
    return spEigVals, spEigVecs, C1D, L1D

##############################################################################
# Function to solve boundary value problem for JJA resonator array with our without JJ defect capacitively coupled 
# to open transmission lines at both ends
def JJAResOpenEigs(Nx, Lleft, Cleft, Cin, Cjjagnd, Ljja, Cjja, Ljjdefect, Cjjdefect, Lright, Cright, Cout, kRefVec, errTol=1e-5, itNum=20, disp="on"):
    """
    Function to solve boundary value problem for JJA resonator with open boundary
    conditions at both ends
    
    Parameters:
    -----------
    Nx
        Grid size
    Cjjagnd 
        JJA resonator coupling to ground
    Ljja
        JJA resonator inductance
    Cjja
        JJA resonator coupling capacitance
    Ljjdefect
        Defect JJ resonator inductance in the middle
    Cjjdefect
        Defect JJ resonator shunting capacitance in the middle
    Lleft/Lright
        Left/right transmission line inductance
    Cleft/Cright
        Left/right transmission line capacitance to ground
    Cin
        Coupling capacitance between left transmission line and JJA resonator
    Cout
        Coupling capacitance between right transmission line and JJA resonator
    kRefVec
        Vector of reference eigenvalues for iterative calculation
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
    errVals = np.zeros( (nCom,), dtype=complex )
    
    # Vector of ones
    e = np.ones( (Nx,), dtype=complex )
    
    # Construct (Inverse) Inductance matrix constant elements
    diagVals = np.array([2*e, -e, -e])
    diagPos = np.array([0, -1, +1])
    L1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    L1D = sp.sparse.csr_matrix(L1D)

    # Construct capacitance matrix constant elements
    χc = Cjja/(Cjjagnd+2*Cjja)
    diagVals = np.array([e, -χc*e, -χc*e])
    diagPos = np.array([0, -1, +1])
    C1D = sp.sparse.spdiags(diagVals, diagPos, Nx, Nx)
    C1D = sp.sparse.csr_matrix(C1D)

    # Loop over eigenvalues to be computed FOR NOW THE LOOP STARTS AT 8 TO AVOID CONVERGENCE ERROR OF SMALLER EIGENVALUES
    for j in range(8,nCom):
    #for j in range(nCom):
    
        # Set eigenvalue reference
        kRef = kRefVec[j]
       
        # Set error 
        #errre = 1
        #errim = 1
        err = 1
        # Loop over iterations
        n = 0 
        while err > errTol and n <= itNum:
        #while errre > errTol or errim > errTol:
            #if n > itNum: break
            # Terms for implementing outgoing boundary condition at left
            kdxleft = np.sqrt(kRef * Lleft * Cleft / ( Ljja * (Cjjagnd + 2*Cjja) ) )
            βLeft = (2 + 1j*kdxleft) / (2 - 1j*kdxleft) # Outgoing wavevector term
            wbarleft = Ljja*( Cjjagnd + 2*Cjja ) / ( Lleft*(Cleft + Cin) )
            ζLeft = (Cin / (Cleft+Cin)) / (1 - wbarleft/kRef*(1-βLeft))
            # Terms for implementing outgoing boundary condition at right
            kdxright = np.sqrt(kRef * Lright*Cright/(Ljja*(Cjjagnd + 2*Cjja)))
            βRight = (2+1j*kdxright)/(2-1j*kdxright) # Outgoing wavevector term
            wbarright = Ljja*(Cjjagnd + 2*Cjja)/( Lright*(Cright + Cout) )
            ζRight = ( Cout/( (Cright+Cout) ) / (1 - wbarright/kRef*(1-βRight)) )

            # Implement outgoing boundary conditions - left side resonator EOM
            χLeft = Cin/(Cjjagnd + 2*Cjja)*(1 - ζLeft)
            C1D[0,0]   = 1.0 - χc + χLeft
            L1D[0,0]   = 1.0

            # Implement outgoing boundary conditions - right side resonator EOM
            χRight = Cout/(Cjjagnd + 2*Cjja)*(1 - ζRight)
            C1D[-1,-1] = 1.0 - χc + χRight
            L1D[-1,-1] = 1.0

            # Implement JJ defect coupling terms
            χJJL = Ljja/Ljjdefect
            # Inductance matrix coupling via the junction
            L1D[Nx/2-1,Nx/2+1-1] = -χJJL
            L1D[Nx/2+1-1,Nx/2-1] = -χJJL
            L1D[Nx/2-1,Nx/2-1] = 1 + χJJL
            L1D[Nx/2+1-1,Nx/2+1-1] = 1 + χJJL
     
            # Capacitance matrix coupling via junction
            χJJC = Cjjdefect/(Cjjagnd + 2*Cjja)
            χC = Cjja/(Cjjagnd + 2*Cjja)
            C1D[Nx/2-1,Nx/2+1-1] = -χJJC
            C1D[Nx/2+1-1,Nx/2-1] = -χJJC
            C1D[Nx/2-1,Nx/2-1] = 1 - χC + χJJC
            C1D[Nx/2+1-1,Nx/2+1-1] = 1 - χC + χJJC
            # Solve generalized eigenvalue problem
            nEig = 1
            spEigVals, spEigVecs = ssla.eigs(L1D, k=nEig, M=C1D, sigma=kRef )

            # Sort eigenvalues and eigenvectors
            sIVec = np.argsort( np.real(np.sqrt(spEigVals)) )
            spEigVals = spEigVals[sIVec]
            spEigVecs = spEigVecs[:,sIVec]
            #kVals = np.sqrt(spEigVals)
            kVals = spEigVals
    
            # Calculate error and set new reference
            err = np.abs( kRef - np.real(kVals[0]) )
            #errre = np.abs( np.real(kRef**2 - spEigVals[0]) )
            #errim = np.abs( np.imag(kRef**2 - spEigVals[0]) )
            kRef = kVals[0]
            kRef = np.real(kVals[0])
            if disp == "on":
            #    print('Mode: ' + str(j) + ', errre: ' + '{:.4f}'.format(errre) + ', errim: ' + '{:.4f}'.format(errim) + ', it. num.: ' + str(n) )
                print( 'Mode: ' + str(j) + ', err: ' + str(err) + ', iteration: ', str(n) )
                print('EigVal ' + str(kVals[0]))
    
            # Increment iteration
            n = n + 1
            
        # Store computed eigenvalue, eigenvector, and convergence error 
        eigVals[j] = kVals[0]
        #errVals[j] = errre + 1j*errim
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
