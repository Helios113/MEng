import numpy as np
import matplotlib.pyplot as plt
import solver_ENM as solver
import lhsmdu

mat_model=None
F_traction = 0.0#external force in Newtons
F_body = 0.0
Le = []
L = 0.0
nNodes = 0
nElem = 0
X = []
BClogic = []
################################ Material Models ##########################
def lin(l):
    A=1000
    B=50
    P=A*(l-1)
    dPdl = A
    return P, dPdl
def exp_m(l):
    A=0.2
    B=20
    P=A*np.exp(B*(l-1))
    dPdl = A*B*np.exp(B*(l-1))
    return P, dPdl

def VM(l, param):
    A=param[0]
    B=param[1]
    expB = np.exp(B*(l*l+2./l-3))
    P=2*A*(l-1./l/l)*expB-A*(1-1./l**3)
    #dPdl = 2*A*B*(l - 1.0/l**2)*(2*l - 2./l**2)*expB + 2*A*(1 + 2.0/l**3)*expB - 3.0*A/l**4
    return P

def MR(l, param):#compression is non linear
    mu=param[0]
    nu=param[1]
    P = mu*nu*(l-1./l/l)+mu*(1-nu)*(1-1./l/l/l)
    #dPdl = mu*(nu*(1 + 2.0/l**3) + 3.0*(-nu + 1)/l**4)
    return P
###########################################################################




def main():
    global mat_model
    global F_traction
    global F_body
    global Le
    global L
    global nNodes
    global nElem
    global X
    global BClogic
    mat_model=VM
    L = 2.0 #length of beam 
    nNodes = 6# number of nodes
    F_traction= -20#external force in Newtons
    F_body = -5
    #printout=True

    #factor = -1 #+1 for compression and -1 for extension

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

def _force_stiff(x1,x2,le,param):
    #print("_force_stiff call", x1,x2,le,n)
    global mat_model
    lam = (x2-x1)/le
    if lam<0:
        raise ValueError("Negative Jacobian")
    p = mat_model(lam, param) #where the model comes in
    #print("P",p)
    dNdxi = [-1.,1.] 
    f = np.zeros(2)
    for i in range(0,2):
        f[i] = dNdxi[i]*p 
    return f

def force_stiff(x, param):
    #print("Call", x,n)
    global F_traction
    global F_body#
    global Le#
    global nNodes#
    global nElem #
    global BClogic#
    fglobal = np.zeros(nNodes) #empty global force vector
    Kglobal = np.zeros([nNodes,nNodes]) #empty global stiffness matrix
    for e in range(nElem):
        try:
            f = _force_stiff(x[e],x[e+1],Le[e] ,param) # force and stiffnes = force_stiff
        except Exception as e:
            print(e)
            return 0
        fglobal[e:e+2] += f #global force vector
    
    fglobal[-1] += F_traction
    
    #for force acting along the length
    
    for e in range(nElem):
        le = X[e+1]-X[e]
        fglobal[e:e+2] += F_body*le
    
    #remove the fixed boundary condition

    fglobal_reduced=np.array(fglobal[np.logical_not(BClogic)]).transpose()
    #print("force_stiff fgobal reduced ln:135",fglobal_reduced)
    return fglobal_reduced
	
#NEeds more work but getting there
def test(x):
    #lhsmdu.sample(nNodes, 1)
    #c = np.array([k+(i/9)**2 for i,k in enumerate(x)])
    a = np.random.rand(nNodes)
    c = 2*x+(a.flatten()/10) #pick a c that dissatisfies the function alot
                                        #definitly need a random pertubation 
    
    while not np.all(c[:-1] <= c[1:]) or not np.all(c >= 0):
        a = np.random.rand(nNodes)
        c = 2*x+(a.flatten()/10)
    c[0] = 0
    #c[-1] +=1 
    print("Input c is:",c)
    print("Input x is:",x)
    s = solver.solve(force_stiff, c)
    return (s.execute(x),c)


main()
