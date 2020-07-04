import Extended.ENM as ENM
import Standard.NM as NM
import numpy as np
import matplotlib.pyplot as plt


x = np.array([7,50,1,2], dtype='float64')



res_1 = ENM.solve(x)
res_2 = NM.solve(x[:2])


ord1 = res_1[3]
ord2 = res_2[3]
print(res_1[1],res_2[1])
#plt.plot(ord1)
plt.plot(ord2)
plt.show()