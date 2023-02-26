



import numpy as np
from numpy import random

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt


def b(n_0):
    if n_0 == 0:
        return(1)
    else:
        return(1/2)
    


### parameters

lam = 300
theta = 2

n = random.poisson(lam,1)
X = random.uniform(size=(n[0],2))


### showcase X process
fig, ax = plt.subplots()
ax.plot(X[:,0],X[:,1],"o")
ax.set_aspect(1)
ax.set(xlim=(0, 1), ylim=(0, 1))
plt.show()


D = distance_matrix(X,X)
Sigma = np.exp(-theta*D)
g = np.linalg.cholesky(Sigma) @ random.normal(size=n)
y = np.zeros(n)
for i in np.arange(n):
    y[i] = random.normal(g[i])
    
    
X_0 = X[y<0,:]
g_0 = g[y<0]
y_0 = y[y<0]

X_1 = X[y>0,:]
g_1 = g[y>0]
y_1 = y[y>0]


### showcase X_0 and X_1 processes
fig, ax = plt.subplots()
ax.plot(X_0[:,0],X_0[:,1],"o",c="silver")
ax.plot(X_1[:,0],X_1[:,1],"o")
ax.set_aspect(1)
ax.set(xlim=(0, 1), ylim=(0, 1))
plt.show()




### append delete


