import numpy as np
import matplotlib.pyplot as plt
import solver_ENM as solver
mat_model=None
Fext = 0.0
Le = []
L = 0.0
nNodes = 0
nElem = 0
X = []
factor = 0
BClogic = []
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
    global mat_model
    global Fext
    global Le
    global L
    global factor
    global nNodes
    global nElem
    global X
    global BClogic
    mat_model=exp_m
    L=1.0 #length of beam 
    nNodes=30 # number of nodes
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
    x1 = test(x)
    #plotting
    y = [1]*len(x)
    y1 = [1.1]*len(x)
    #print(_force_stiff(1,2,3,2))
    plt.plot(x,y, "-o")
    plt.plot(x1,y1, "-o")
    plt.plot(0,1 , "s")
    plt.show()
    #end plotting

def _force_stiff(x1,x2,le,n):
    #print("_force_stiff call", x1,x2,le,n)
    global mat_model
    lam = (x2-x1)/le
    if lam<0:
        raise ValueError("Negative Jacobian")
    p,dpdl = mat_model(lam) #where the model comes in
    #print("P",p)
    dNdxi = [-1.,1.] 
    f = np.zeros(2)
    if n ==2:
        K = np.zeros([2,2])
    for i in range(0,2):
        f[i] = dNdxi[i]**2*p 
        if n==2:
            for j in range(0,2):
                K[i,j] = dNdxi[i]*dNdxi[j]*dpdl
   
    if n ==2:
        K /= le #divide whole matrix by length
        #print("K_element",K)
        #print("f_element",f)
        return f,K
    return f, 0

def force_stiff(x, n):
    #print("Call", x,n)
    global Fext#
    global Le#
    global factor#
    global nNodes#
    global nElem #
    global BClogic#
    fglobal = np.zeros(nNodes) #empty global force vector
    Kglobal = np.zeros([nNodes,nNodes]) #empty global stiffness matrix
    for e in range(nElem):
        try:
            f,k = _force_stiff(x[e],x[e+1],Le[e] ,n) # force and stiffnes = force_stiff
        except:
            print("expetion")
            return 0
        fglobal[e:e+2] += f #global force vector
        if n==2:
            Kglobal[e:e+2,e:e+2] += k #global stiffness matrix
            #print("K_global, ln:127",Kglobal)
    fglobal[-1] += factor*Fext
    
    #for force acting along the length
    #for e in range(nElem):
    #    le = X[e+1]-X[e]
    #    fglobal[e:e+2] += factor*Fext*le

    #remove the fixed boundary condition
    fglobal_reduced=np.array(fglobal[np.logical_not(BClogic)]).transpose()
    #print("force_stiff fgobal reduced ln:135",fglobal_reduced)
    if n ==2:
        temp=np.array(Kglobal[np.logical_not(BClogic), :])
        stiff_reduced=np.array(temp[:,np.logical_not(BClogic)])
        return fglobal_reduced, stiff_reduced
    return fglobal_reduced
	

def test(x):
    c = x*1.5
    s = solver.solve(force_stiff, c)
    return s.execute(x)

main()