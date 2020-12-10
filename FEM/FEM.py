import numpy as np
import matplotlib.pyplot as plt
mat_model=None

################################ Material Models ##########################
def exp_m(l):
    A=0.2
    B=50
    P=A*np.exp(B*(l-1))
    dPdl = A*B*np.exp(B*(l-1))
    return P, dPdl

def VM(l):
    A=0.2
    B=50
    expB = np.exp(B*(l*l+2./l-3))
    P=2*A*(l-1./l/l)*expB-A*(1-1./l**3)
    dPdl = 2*A*B*(l - 1.0/l**2)*(2*l - 2./l**2)*expB + 2*A*(1 + 2.0/l**3)*expB - 3.0*A/l**4
    return P, dPdl

def MR(l):
    mu=1.
    nu=1.
    P = mu*nu*(l-1./l/l)+mu*(1-nu)*(1-1./l/l/l)
    dPdl = mu*(nu*(1 + 2.0/l**3) + 3.0*(-nu + 1)/l**4)
    return P,dPdl
###########################################################################




def main():
    mat_model=exp_m
    L=1.0 #length of beam 
    nNodes=5 # number of nodes
    Fext=12.85 #external force in Newtons
    #printout=True
    factor = -1 #+1 for compression and -1 for extension

    nElem = nNodes-1 # number of elements

    #uniform node spacing
    X=np.linspace(0,L,nNodes)
    Le=[X[i+1]-X[i] for i in range(0,nElem)] #Element Length#

    """Boundry condition node logic"""
    BClogic=nNodes*[False] 
    BClogic[0]=True

    MAX_iter=50
    TOL=1e-9
    x = X.copy() #original guess
    eps = 1e-4

    F = np.logspace(-1,3,30) #F vs Fext
    test(x,Fext)
    #plotting
    y = [1]*len(x)
    plt.plot(x,y, "-o")
    plt.plot(0,1 , "s")
    plt.show()
    #end plotting


def test(x, F):


main()