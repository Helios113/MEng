import numpy as np
def func(x, coeff):
    
    #return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * np.sin(x) #[-0.001, 0.1, 0.1, 2, 15]
    
    return coeff[0]**3 * x ** 3 + coeff[1]**2 * x ** 2 + coeff[2]**2 * x + coeff[3]**3 + coeff[4] * np.sin(x) #[-0.001, 0.1, 0.1, 2, 15]

    #return coeff[0]*x[:,0]**coeff[1]+coeff[2]*x[:,1]**coeff[3]  #[0.1,4,0.1,2]   #only positive values of x or fix function 

    #test function
    #return coeff[0]*x**coeff[1]+coeff[2]*x**coeff[3]

    #return coeff[0]*np.exp(-x[:,0]/coeff[1])+coeff[2]*np.exp(-x[:,1]/coeff[3]) #[4,2,1,10]

    #return (coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)+coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)).real
    


def der_func(x, coeff):

    #return np.array([x ** 3,x ** 2 ,x,1,np.sin(x)]).reshape(1,-1)

    return np.array([3*coeff[0]**2 * x ** 3, 2*coeff[1]* x ** 2 , 2*coeff[2] * x , 3*coeff[3]**2 , np.sin(x)]).reshape(1,-1)

    #return np.array([x[0]**coeff[1], np.log(x[0])*coeff[0]*x[0]**coeff[1], x[1]**coeff[3], np.log(x[1])*coeff[2]*x[1]**coeff[3]]).reshape(1,-1)  # only positive values of x or fix function
    
    #return np.array([x**coeff[1], np.log(x)*coeff[0]*x**coeff[1], x**coeff[3], np.log(x)*coeff[2]*x**coeff[3]]).reshape(1,-1)  # only positive values of x or fix function
    
    #return np.array([np.exp(-x[0]/coeff[1]),x[0]*coeff[0]*np.exp(-x[0]/coeff[1])/(coeff[1]**2),
    #                 np.exp(-x[1]/coeff[3]),x[1]*coeff[2]*np.exp(-x[1]/coeff[3])/(coeff[3]**2)]).reshape(1,-1)
    
    """
    return np.array([
        np.exp(complex(coeff[2],coeff[1])*x),coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)*1j*x,coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)*x,
        np.exp(complex(coeff[5],coeff[4])*x),coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)*1j*x,coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)*x
    ]).real.reshape(1,-1)
    """


def der2_func(x, coeff):

    #return np.zeros((1,5,5))
    
    return np.array([6*coeff[0] * x ** 3, 0, 0, 0, 0,
                     0, 2*x ** 2, 0, 0, 0,
                     0, 0, 2 * x, 0, 0,
                     0, 0, 0, 6*coeff[3], 0,
                     0, 0, 0, 0, 0]).reshape(1,5,5)
    
    
    """
    return np.array([0, np.log(x[0])*x[0]**coeff[1],0,0,
                     np.log(x[0])*x[0]**coeff[1], np.log(x[0])**2*coeff[0]*x[0]**coeff[1],0,0,
                     0,0,0, np.log(x[1])*x[1]**coeff[3],
                     0,0,np.log(x[1])*x[1]**coeff[3], np.log(x[1])**2*coeff[2]*x[1]**coeff[3]]).reshape(1,4,4)
    """
    """
    return np.array([0, np.log(x)*x**coeff[1],0,0,
                     np.log(x)*x**coeff[1], np.log(x)**2*coeff[0]*x**coeff[1],0,0,
                     0,0,0, np.log(x)*x**coeff[3],      
                     0,0,np.log(x)*x**coeff[3], np.log(x)**2*coeff[2]*x**coeff[3]]).reshape(1,4,4)

    """
    """
    return np.array([0,x[0]*np.exp(-x[0]/coeff[1])/(coeff[1]**2),0,0,
                     x[0]*np.exp(-x[0]/coeff[1])/(coeff[1]**2), -x[0]*coeff[0]*(2*coeff[1]-x[0])*np.exp(-x[0]/coeff[1])/(coeff[1]**4),0,0,
                     0,0,0,x[1]*np.exp(-x[1]/coeff[3])/(coeff[3]**2),
                     0,0,x[1]*np.exp(-x[1]/coeff[3])/(coeff[3]**2), -x[1]*coeff[2]*(2*coeff[3]-x[1])*np.exp(-x[1]/coeff[3])/(coeff[3]**4)]).reshape(1,4,4)
    """
    """
    return np.array([0,
                     np.exp(complex(coeff[2],coeff[1])*x)*1j*x,
                     np.exp(complex(coeff[2],coeff[1])*x)*x,
                     0,0,0,

                     np.exp(complex(coeff[2],coeff[1])*x)*1j*x,
                     -coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)*x**2,
                     coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)*1j*x**2,
                     0,0,0,

                     np.exp(complex(coeff[2],coeff[1])*x)*x,
                     coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)*1j*x**2,
                     coeff[0]*np.exp(complex(coeff[2],coeff[1])*x)*x**2,
                     0,0,0,

                     0,0,0,
                     0,np.exp(complex(coeff[5],coeff[4])*x)*1j*x,
                     np.exp(complex(coeff[5],coeff[4])*x)*x,
                     
                     0,0,0,
                     np.exp(complex(coeff[5],coeff[4])*x)*1j*x,
                     -coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)*x**2,
                     coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)*1j*x**2,

                     0,0,0,
                     np.exp(complex(coeff[5],coeff[4])*x)*x,
                     coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)*1j*x**2,
                     coeff[3]*np.exp(complex(coeff[5],coeff[4])*x)*x**2
                     ]).real.reshape(1,6,6)              
    """