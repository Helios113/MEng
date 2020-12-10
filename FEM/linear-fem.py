import numpy as np
import sys
import ENM as solver

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



def force_stiff(x1,x2,le):
    #x1 location of start of element
    #x2 location of end of element
    #l  original length of element? 

    global mat_model
    lam = (x2-x1)/le #get lambda linear element
    if lam<0:
        raise ValueError("Negative Jacobian")
    p,dpdl = mat_model(lam) #where the model comes in
    dNdxi = [-1.,1.] #shape function?
    f = zeros(2)
    K = zeros([2,2])
    for i in range(0,2):
        f[i] = dNdxi[i]*p # f is equal shape function * p
        for j in range(0,2):
            K[i,j] = dNdxi[i]*dNdxi[j]*dpdl # stiffness = shapefunction squared * derivative of p with respect to l

    K /= le devide whole matrix by length
    return f,K

def stretch(x):
    global nElem
    global X
    for e in range(nElem):
        lam = (x[e+1]-x[e])/(X[e+1]-X[e])
        if lam<0:
            raise ValueError("Negative Jacobian")

        print e, lam
    return

def force_vec(Fext,x):
    global nElem
    fglobal = zeros(nNodes)

    #assemble the internal force
    for e in range(nElem):
        try:
            f,k = force_stiff(x[e],x[e+1],Le[e])
        except:
            return 0
        fglobal[e:e+2] += f

    #add external force
    #fglobal[-1] += factor*Fext
	
    for e in range(nElem):
        le = x[e+1]-x[e]
        fglobal[e:e+2] += factor*Fext*le
	
    return array(fglobal[logical_not(BClogic)]).transpose()

def step(Fext,x):
    fglobal = np.zeros(nNodes) #empty global force vector
    Kglobal = np.zeros([nNodes,nNodes]) #empty global stiffness matrix
    global nElem #number of elements
    global X #original locations
    #assemble the internal force and stiffness matrices
    for e in range(nElem):
        try:
            f,k = force_stiff(x[e],x[e+1],Le[e]) # force and stiffnes = force_stiff
        except:
            return 0
        fglobal[e:e+2] += f #global force vector
        Kglobal[e:e+2,e:e+2] += k #global stiffness matrix

    #for force acting only on the end
    #fglobal[-1] += factor*Fext

    #for force acting along the length
    for e in range(nElem):
        le = X[e+1]-X[e]
        fglobal[e:e+2] += factor*Fext*le
	
    if printout:
        print fglobal,x

    if any(isnan(fglobal)):
        if printout:
            print 'NaN found in the force vector'
        return 0

    #remove the fixed boundary condition
    fglobal_reduced=array(fglobal[logical_not(BClogic)]).transpose()

    temp=array(Kglobal[logical_not(BClogic), :])
    stiff_reduced=np.array(temp[:,logical_not(BClogic)])

    #Use Iterative method
    try:    #                       stiffnes is derivative of force hence its the jacobian
        dx = np.insert(linalg.solve(stiff_reduced, -fglobal_reduced),0,0) # add zero at beggining of array to restore shape
    except:
        print 'Singular matrix', stiff_reduced
        dx = 0
    return dx

def test(x,f1,f2,fstep):
    # fstep - force step?
    # f1 - first force
    # f2 - second force
    # x - displacement guess
    Fext=0. # extension force

    while Fext<f1:
        # while the extension force is below force1 we increase Fext by fstep
        Fext += fstep
        Fext = min(Fext,f1) # Checking not to overshoot.
        i=0
        while True:
            if printout:
                print i,
            dx=step(Fext,x,c,1) #getting newton -step
            x=x+dx # adding newton step
            i += 1
            if np.linalg.norm(dx)<TOL:
                break;
            if i>MAX:
                raise ValueError("Maximum iterations reached in the first loading")
    lfinal=x[-1]/X[-1] #final extension
    if printout:
        print '-------------------------------'
        print 'Second step'

    Fext = f2 #directly going to forsce2 no load stepping.
    i=0
    while True:
        if printout:
            print i,
        dx=step(Fext,x) #newton step
        if printout:
            print dx
        x=x+dx #adding newton step
        i += 1
        if linalg.norm(dx)<TOL or i>MAX:
            break;
    lfinal=x[-1]/X[-1] # final length
    if printout:
        print '-------------------------------'
        print i, x 
        fvec = force_vec(Fext,x)
        print fvec, linalg.norm(fvec)
    #if abs(-factor*f2-mat_model(lfinal)[0])<1e-5:
    if  linalg.norm(fvec)<1e-5:
        return i
    else:
        stretch(x)
        return 100	

mat_model=exp_m
L=1.
nNodes=5
Fext=12.85
printout=True
factor = -1 #+1 for compression and -1 for extension

nElem = nNodes-1

#uniform node spacing
X=np.linspace(0,L,nNodes)
Le=[X[i+1]-X[i] for i in range(0,nElem)] #Element Length#

BClogic=nNodes*[False]
BClogic[0]=True

MAX=50
TOL=1e-9
x = X.copy() #original guess
eps = 1e-4

F2 = np.logspace(-1,3,30)

#for f1 in arange(0.,10.,0.2):
f1=0.0
for f2 in F2:
    print f1, f2, test(x,f1,f2,0.1)
print ''

