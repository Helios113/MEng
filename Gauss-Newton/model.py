import numpy as np
def func(x, coeff):
    
    return coeff[0]**3 * x ** 3 + coeff[1]**2 * x ** 2 + coeff[2]**2 * x + coeff[3]**3 + coeff[4] * np.sin(x)

    #return coeff[0]*x[:,0]**coeff[1]+coeff[2]*x[:,1]**coeff[3]  [0.1,4,0.1,2]   #only positive values of x or fix function 

    #return coeff[0]*np.exp(-x[:,0]/coeff[1])+coeff[2]*np.exp(-x[:,1]/coeff[3])

    #c1 = complex(coeff[2], coeff[1])
    #c2 = complex(coeff[5], coeff[4])
    #ans = (coeff[0]*np.exp(c1*x)+coeff[3]*np.exp(c2*x)).real
    #return ans
def der_func(x, coeff):
    return np.array([3*coeff[0]**2 * x ** 3, 2*coeff[1]* x ** 2 , 2*coeff[2] * x , 3*coeff[3]**2 , np.sin(x)]).reshape(1,-1)
    #return np.array([x[0]**coeff[1], np.log(x[0])*coeff[0]*x[0]**coeff[1], x[1]**coeff[3], np.log(x[1])*coeff[2]*x[1]**coeff[3]]).reshape(1,-1)  # only positive values of x or fix function


def der2_func(x, coeff):
    return np.array([6*coeff[0] * x ** 3, 0, 0, 0, 0,
                     0, 2*x ** 2, 0, 0, 0,
                     0, 0, 2 * x, 0, 0,
                     0, 0, 0, 6*coeff[3], 0,
                     0, 0, 0, 0, 0]).reshape(1,5,5)
    """
    return np.array([0, np.log(x[0])*x[0]**coeff[1],0,0,
                     np.log(x[0])*x[0]**coeff[1], np.log(x[0])**2*coeff[0]*x[0]**coeff[1],0,0,
                     0,0,0, np.log(x[1])*x[1]**coeff[3],
                     0,0,np.log(x[1])*x[1]**coeff[3], np.log(x[1])**2*coeff[2]*x[1]**coeff[3]]).reshape(1,4,4)"""