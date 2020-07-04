import numpy as np
import matplotlib.pyplot as plt
import solver_ENM as solver
import lhsmdu
from mpl_toolkits.mplot3d import Axes3D 
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
    A=100
    P=A*(l-1)
    dPdl = A
    return P, dPdl
def exp_m(l):
    A=0.2
    B=20
    P=A*np.exp(B*(l-1))
    dPdl = A*B*np.exp(B*(l-1))
    return P, dPdl

def VM(l):
    A=2.48446
    B=0.16860
    expB = np.exp(B*(l*l+2./l-3))
    P=2*A*(l-1./l/l)*expB-A*(1-1./l**3)
    dPdl = 2*A*B*(l - 1.0/l**2)*(2*l - 2./l**2)*expB + 2*A*(1 + 2.0/l**3)*expB - 3.0*A/l**4
    return P, dPdl

def MR(l):#compression is non linear
    mu=5.289
    nu=-0.6417
    P = mu*nu*(l-1./l/l)+mu*(1-nu)*(1-1./l/l/l)
    dPdl = mu*(nu*(1 + 2.0/l**3) + 3.0*(-nu + 1)/l**4)
    return P,dPdl
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
    F_body = -20
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

    F = np.logspace(-1,3,30) 
    
    #F vs Fext
    """
    ans = None
    ans1 = None
    for i in range(100):
        x1 = test(x)
        if x1[0] is not None:
            print("Ans is", x1[0])
            ans = np.hstack((ans, x1[1].reshape(-1,1))) if ans is not None else x1[1].reshape(-1,1)
        else:
            ans1 = np.hstack((ans1, x1[1].reshape(-1,1))) if ans1 is not None else x1[1].reshape(-1,1)
    #plotting
   
    #print(ans)
    plt.plot(ans.T)
    plt.show()
    plt.plot(ans1.T)
    plt.show()
    print("Worked average",np.average(ans,axis=1))
    print("Didnt wokr",np.average(ans1,axis=1))
    #print(_force_stiff(1,2,3,2))
    """
    
    y = [1]*len(x)
    y1 = [1.1]*len(x)
    x1,c = test(x)
    l1 = []
    if x1 is not None:
        print(x1)
        for i,_ in enumerate(x1[:-1]):
            l1.append((x1[i+1]-x1[i])/Le[0])
        #print(c)
        #print(np.linalg.norm(x1-c))4
        print("Stretch",repr(l1))
        avg = np.average(l1)
        maxx = np.max(l1)
        minn = np.min(l1)
        l1-=minn
        l1/=(maxx-minn)
        l1+=1
        l1*=avg
        l1 *=avg/np.average(l1)
        print("Corrected Stretches",repr(l1.tolist()))
    else:
        print(c)
        #print(np.linalg.norm([0.,0.5978,1.1956,1.7934,2.3912]-c))
    
    """plt.plot(x,y, "-o")
    plt.plot(x1,y1, "-o")
    plt.plot(0,1 , "s")
    plt.show()
    #end plotting
    """

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
        f[i] = dNdxi[i]*p 
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
            f,k = _force_stiff(x[e],x[e+1],Le[e] ,n) # force and stiffnes = force_stiff
        except Exception as e:
            print(e)
            return 0
        fglobal[e:e+2] += f #global force vector
        if n==2:
            #print("fglobal1", fglobal)
            Kglobal[e:e+2,e:e+2] += k #global stiffness matrix
            #print("K_global, ln:127",Kglobal)
    
    fglobal[-1] += F_traction
    
    #for force acting along the length
    
    for e in range(nElem):
        le = X[e+1]-X[e]
        fglobal[e:e+2] += F_body*le
    
    #remove the fixed boundary condition

    fglobal_reduced=np.array(fglobal[np.logical_not(BClogic)]).transpose()
    #print("force_stiff fgobal reduced ln:135",fglobal_reduced)
    if n ==2:
        #print("fglobal2", fglobal)
        temp=np.array(Kglobal[np.logical_not(BClogic), :])
        stiff_reduced=np.array(temp[:,np.logical_not(BClogic)])
        return fglobal_reduced, stiff_reduced
    return fglobal_reduced
	
#NEeds more work but getting there
def test(x):
    #lhsmdu.sample(nNodes, 1)
    #c = np.array([k+(i/9)**2 for i,k in enumerate(x)])
    loc = []
    loc1 = []
    ans = []
    for i in np.linspace(1,10,19):
        for j in np.linspace(5,5,1):
            a = np.random.rand(nNodes)
            print(j ,i)
            c = i*x+(a.flatten()/j) #pick a c that dissatisfies the function alot
                                                    #definitly need a random pertubation 

            while not np.all(c[:-1] <= c[1:]) or not np.all(c >= 0):
                a = np.random.rand(nNodes)
                c = i*x+(a.flatten()/j)
        
            c[0] = 0
            #c[-1] +=1 
            #print("Input c is:",c)
            #print("Input x is:",x)
            s = solver.solve(force_stiff, c)
            b = s.execute(x) 
            loc.append(i)
            loc1.append(j)
            ans.append(b[1])
            if b[1] !=0 :
                print(b[0])
    print(ans)
    print(loc)
    plt.plot(loc,ans)

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(loc, loc1, ans, linewidth=0.2, antialiased=True)
    """
    plt.show()

main()
