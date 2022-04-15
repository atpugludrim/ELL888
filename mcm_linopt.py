import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog, fsolve
from scipy.stats import multivariate_normal as mvn


class Timer:
    def __enter__(this):
        this.start = time.perf_counter()
    def __exit__(this, exc_type, exc_val, exc_tb):
        print("{:.4f}s".format(time.perf_counter()-this.start))
def getdata(x1_size=5,x2_size=4):
    mu_1 = [0,1]
    mu_2 = [3,1]
    sig_1 = [[1.2,0],[0,2]]
    sig_2 = [[2,0.4],[0.4,1.2]]
    rv1 = mvn(mean=mu_1,cov=sig_1)
    rv2 = mvn(mean=mu_2,cov=sig_2)
    randsom = np.random.randint(0,1000)
    randsom = 852
    print(randsom)
    x1 = rv1.rvs(size=x1_size,random_state=randsom)
    y1 = np.array([-1 for k in range(x1_size)])[:,None]
    x2 = rv2.rvs(size=x2_size,random_state=randsom)
    y2 = np.array([1 for k in range(x2_size)])[:,None]
    x = np.vstack((x1,x2))
    y = np.vstack((y1,y2))
    std = np.std(x,0)
    x = (x-np.mean(x,0))/std

    x1_size=10
    x2_size=10
    x1 = rv1.rvs(size=x1_size)
    y1 = np.array([-1 for k in range(x1_size)])[:,None]
    x2 = rv2.rvs(size=x2_size)
    y2 = np.array([1 for k in range(x2_size)])[:,None]
    x_test = np.vstack((x1,x2))
    y_test = np.vstack((y1,y2))
    std = np.std(x,0)
    x_test = (x_test-np.mean(x,0))/std
    return x,y,x_test,y_test

def kernel(x1,x2,gamma=0.4):
    return np.exp(-gamma*np.sum(np.square(x1-x2)))

def getkernelmatrix(x):
    K = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            K[i,j] = kernel(x[i,:],x[j,:])
    return K

def get_c_for_linopt(x):
    c = [1,0]
    C = 1e1
    for k in range(x.shape[0]):
        c.append(0)
    for k in range(x.shape[0]):
        c.append(C)
    return np.array(c)

def get_Aub_bub_for_linopt(K,y):
    Aub_1 = []
    bub_1 = []
    l = len(y)
    for i, y_i in enumerate(y):
        qi_coef = np.zeros((l,))
        qi_coef[i] = 1
        Aub_1.append(np.hstack((-1,y_i*K[i,:],y_i,qi_coef)))
        bub_1.append(0)
    Aub_1 = np.array(Aub_1)
    bub_1 = np.array(bub_1)

    Aub_2 = []
    bub_2 = []
    for i, y_i in enumerate(y):
        qi_coef = np.zeros((l,))
        qi_coef[i] = -1
        Aub_2.append(np.hstack((0,-y_i*K[i,:],-y_i,qi_coef)))
        bub_2.append(-1)
    Aub_2 = np.array(Aub_2)
    bub_2 = np.array(bub_2)

    # Aub_3 = []
    # bub_3 = []
    # for i, y_i in enumerate(y):
    #     qi_coef = np.zeros((l,))
    #     qi_coef[i] = -1
    #     Aub_3.append(np.hstack((0,np.zeros((l,)),0,qi_coef)))
    #     bub_3.append(0)
    # Aub_3 = np.array(Aub_3)
    # bub_3 = np.array(bub_3)

    # Aub = np.vstack((Aub_1,Aub_2,Aub_3))
    # bub = np.hstack((bub_1,bub_2,bub_3))

    Aub = np.vstack((Aub_1,Aub_2))
    bub = np.hstack((bub_1,bub_2))
    return Aub, bub

def get_bounds_for_linopt(x):
    bounds = [(1,None),(0,None)]
    for _ in range(x.shape[0]-1):
        bounds.append((0,None))
    bounds.append((None,None))
    for _ in range(x.shape[0]):
        bounds.append((0,None))
    return bounds

x,y,x_test,y_test = getdata(50,50)
for (x1, x2), y1 in zip(x,y):
    if y1 == 1:
        plt.plot(x1,x2,'ro')
    else:
        plt.plot(x1,x2,'b^')
plt.show()

K = getkernelmatrix(x)
c = get_c_for_linopt(x)
A_ub, b_ub = get_Aub_bub_for_linopt(K,y)
bounds = get_bounds_for_linopt(x)
# minimize cTx
# such that:
# A_ub @ x <= b_ub
# bounds hold
with Timer():
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex',options={'maxiter':10000})
print(res.success)
if not res.success:
    print("Failed to optimize")


SV = []
SVl = []
lamb = res.x[1:101]
b = res.x[101]
ctr = 0
for i,l in enumerate(lamb):
    if abs(l) <= 1e-6:
        ctr+=1
    else:
        print(f"support vector: {i}")
        SV.append(i)
        SVl.append(l)

y_hat = []
for j, x_t in enumerate(x_test):
    sum_ = b
    for i in SV:
        sum_ += lamb[i]*kernel(x[i],x_t)
    if sum_ < 0:
        y_hat.append(-1)
    else:
        y_hat.append(1)
    #print(y_hat[-1],y_test[j])
y_hat = np.array(y_hat)
acc = np.sum(y_test.reshape(-1)==y_hat)/len(y_test.reshape(-1))
print(acc*100,"%")
print("Number of support vectors",len(SV))


n_ec = len(SV)
core_points = SV


# define K1 matrix
K1 = np.ones((100,n_ec+1))
k1_gamma = 5
for i in range(100):
    for j in range(n_ec):
        K1[i,j+1]=kernel(x[i],x[SV[j]],k1_gamma)

# define B0 and W0
M = 100
B0 = np.zeros((M,M))
B0[:50,50:] -= K[:50,50:]/M
B0[:50,:50] += K[:50,:50]*(1/50-1/100)
B0[50:,50:] += K[50:,50:]*(1/50-1/100)
B0[50:,:50] -= K[50:,:50]/M

W0 = np.zeros((M,M))
np.fill_diagonal(W0,np.diagonal(K))
W0[:50,:50] -= K[:50,:50]/50
W0[50:,50:] -= K[50:,50:]/50

D = 0.0001

# define P and Q
P = np.matmul(K1.T,np.matmul(B0,K1))
Q = np.matmul(K1.T,np.matmul(W0+D*np.eye(M),K1))

w, v = eigsh(P,k=1,M=Q,which='LA')

# get scaling factor
q = np.matmul(K1,v)
qq = np.diag(q.reshape(-1))

# get modified kernel
opt_K = np.matmul(qq,np.matmul(K,qq))

c = get_c_for_linopt(x)
A_ub, b_ub = get_Aub_bub_for_linopt(opt_K,y)
bounds = get_bounds_for_linopt(x)
# minimize cTx
# such that:
# A_ub @ x <= b_ub
# bounds hold
with Timer():
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='revised simplex',options={'maxiter':10000})
print(res.success)
if not res.success:
    print("Failed to optimize")

SV = []
SVl = []
lamb = res.x[1:101]
b = res.x[101]
ctr = 0
for i,l in enumerate(lamb):
    if abs(l) <= 1e-6:
        ctr+=1
    else:
        print(f"support vector: {i}")
        SV.append(i)
        SVl.append(l)

y_hat = []
for k,x_t in enumerate(x_test):
    sum_ = b
    for i in SV:
        sum_2 = v[0,0]
        for j,p in enumerate(core_points):
            sum_2 += v[j+1,0]*kernel(x_t,x[p],k1_gamma)
        sum_ += lamb[i]*q[i]*sum_2*kernel(x[i],x_t)
    if sum_ < 0:
        y_hat.append(-1)
    else:
        y_hat.append(1)
    #print(y_hat[-1],y_test[k])
y_hat = np.array(y_hat)
acc = np.sum(y_test.reshape(-1)==y_hat)/len(y_test.reshape(-1))
print(acc*100,"%")
print("Number of support vectors",len(SV))

# Steps for Kernel PCA
# MEAN SHIFT
one_m = np.ones((M,M))/M
K_centred = K - np.matmul(one_m,K) - np.matmul(K,one_m) + np.matmul(one_m,np.matmul(K,one_m))

w, v = eigsh(K_centred,k=2,which='LA')
v[:,0] *= 1/np.sqrt(w[0])
v[:,1] *= 1/np.sqrt(w[1])

for i in range(x.shape[0]):
    proj = np.zeros((2,))
    for j in range(x.shape[0]):
        proj[0] += v[j,0]*K[i,j]
        proj[1] += v[j,1]*K[i,j]
    if y[i] == 1:
        plt.plot(proj[0],proj[1],'ro')
    else:
        plt.plot(proj[0],proj[1],'b^')
plt.title("With non-optimized kernel")
plt.show()
# Steps for Kernel PCA
# MEAN SHIFT
one_m = np.ones((M,M))/M
K_centred = opt_K - np.matmul(one_m,opt_K) - np.matmul(opt_K,one_m) + np.matmul(one_m,np.matmul(opt_K,one_m))

w, v = eigsh(K_centred,k=2,which='LA')
v[:,0] *= 1/np.sqrt(w[0])
v[:,1] *= 1/np.sqrt(w[1])

for i in range(x.shape[0]):
    proj = np.zeros((2,))
    for j in range(x.shape[0]):
        proj[0] += v[j,0]*opt_K[i,j]
        proj[1] += v[j,1]*opt_K[i,j]
    if y[i] == 1:
        plt.plot(proj[0],proj[1],'ro')
    else:
        plt.plot(proj[0],proj[1],'b^')
plt.title("With optimized kernel")
plt.show()
