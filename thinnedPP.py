



import numpy as np
from numpy import random

from scipy.stats import truncnorm
from scipy.stats import norm

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt


def b(n_0):
    if n_0 == 0:
        return(1)
    else:
        return(1/2)
    
random.seed(0)

### parameters

lam = 100
theta = 2
var = 1

n = random.poisson(lam,1)
X = random.uniform(size=(n[0],2))


### showcase X process
fig, ax = plt.subplots()
ax.plot(X[:,0],X[:,1],"o")
ax.set_aspect(1)
ax.set(xlim=(0, 1), ylim=(0, 1))
plt.show()


D = distance_matrix(X,X)
Sigma = var*np.exp(-theta*D)
g = np.linalg.cholesky(Sigma) @ random.normal(size=n)
y = np.zeros(n)
for i in np.arange(n):
    y[i] = random.normal(g[i])
    
    
X_0 = X[y<0,:]
g_0 = g[y<0]
y_0 = y[y<0]
n_0 = X_0.shape[0]

X_1 = X[y>0,:]
g_1 = g[y>0]
y_1 = y[y>0]
n_1 = X_1.shape[0]




### showcase X_0 and X_1 processes
fig, ax = plt.subplots()
ax.plot(X_0[:,0],X_0[:,1],"o",c="silver")
ax.plot(X_1[:,0],X_1[:,1],"o")
ax.set_aspect(1)
ax.set(xlim=(0, 1), ylim=(0, 1))
plt.show()




### append delete

### proposals

n_0_prop = n_1

### containers

Sigma_current = var*np.exp(-theta*distance_matrix(X_1,X_1))
Sigma_inv_current = np.linalg.inv(Sigma_current)

n_0_current = 0

X_current = X_1
g_current = random.normal(size=n_1)
y_current = np.ones(n_1)



### chain

N = 2000

n_0_run = np.zeros(N)



for i in range(n_0_prop):

    if random.uniform() < b(n_0_current):
        
        s_new = random.uniform(size=2)
        d_new = distance_matrix(X_current,[s_new])
        Sigma_new = var*np.exp(-theta*d_new)
        w = Sigma_inv_current@Sigma_new
        wtb = np.inner(w[:,0],Sigma_new[:,0])
        d = var - wtb
        g_new = np.sqrt(d) * random.normal() + np.inner(w[:,0], g_current)
        y_new = truncnorm.rvs(a=-np.inf,b=-g_new,loc=g_new)
        
        acc = lam / (n_0_current+1) * (1-b(n_0_current+1))/b(n_0_current) * norm.cdf(y_new-g_new)
        
        if random.uniform() < acc:
            
            ### updates
            
            Sigma_current = np.append(np.append(Sigma_current,Sigma_new,axis=1),[np.append(Sigma_new,var)],axis=0)
            
            Sigma_inv_current = np.append(np.append(Sigma_inv_current + np.outer(w,w)/d,-w/d,axis=1),[np.append(-w/d,1/d)],axis=0)
            
            # print(Sigma_current@Sigma_inv_current)
            
            n_0_current += 1
            
            X_current = np.append(X_current, [s_new], axis=0)
            g_current = np.append(g_current, g_new)
            y_current = np.append(y_current, y_new)
            
    else:
        
        ### deletion
        ind_del = random.randint(n_1,n_1 + n_0_current)
        
        g_old = g_current[ind_del]
        y_old = y_current[ind_del]
        
        acc = n_0_current/lam * b(n_0_current-1)/(1-b(n_0_current)) / norm.cdf(y_old-g_old)


        if random.uniform() < acc:

            ### updates
            
            Sigma_current = np.delete(np.delete(Sigma_current,ind_del,axis=0),ind_del,axis=1)
            
            f = Sigma_inv_current[ind_del,ind_del]
            e = np.delete(Sigma_inv_current[ind_del],ind_del)
            
            Sigma_inv_current = np.delete(np.delete(Sigma_inv_current,ind_del,axis=0),ind_del,axis=1) - 1/f*np.outer(e,e)
            
            # print(Sigma_current@Sigma_inv_current)
            
            n_0_current -= 1
            
            X_current = np.delete(X_current, ind_del, axis=0)
            g_current = np.delete(g_current, ind_del)
            y_current = np.delete(y_current, ind_del)

    ### showcase X_0_current and X_1 processes
    fig, ax = plt.subplots()
    ax.plot(X_current[n_1:,0],X_current[n_1:,1],"o",c="silver")
    ax.plot(X_1[:,0],X_1[:,1],"o")
    ax.set_aspect(1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    plt.show()