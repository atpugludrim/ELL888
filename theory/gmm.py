import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# =============================
ss = np.random.randint(1000,4000,4)
print(ss)
sample1=np.random.normal(loc=25,scale=4,size=ss[0])
sample2=np.random.normal(loc=70,scale=4,size=ss[1])
sample3=np.random.normal(loc=40,scale=3,size=ss[2])
sample4=np.random.normal(loc=10,scale=10,size=ss[3])
sample = np.hstack((sample1,sample2,sample3,sample4))
    # =========================
plt.hist(sample1,bins=25,lw=1,edgecolor='k',alpha=0.6,color='C2')
plt.hist(sample2,bins=25,lw=1,edgecolor='k',color='w',alpha=0.6)
plt.hist(sample3,bins=25,lw=1,edgecolor='k',color='C0',alpha=0.6)
plt.hist(sample4,bins=25,lw=1,edgecolor='k',alpha=0.6,color='C1')
plt.show()
# =============================
sample = sample.reshape((sample.shape[0],1))
model = GaussianMixture(n_components=4, init_params='random', max_iter=1000)
model.fit(sample)
    # =========================
y1 = model.predict_proba(sample)
y2 = model.predict(sample)
# =============================
idx = np.random.randint(0,sample.shape[0],5)
print(y1.shape)
print(idx)
print(y1[idx])
print(np.unique(y2))
