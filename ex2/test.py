import numpy as np
a = np.random.randn(4,3)
b = 1*(a>=0.5).sum()
print(a)
print(b)
# m = np.mean(a,axis = 0)
# s = np.std(a, axis=0)
# print(a)
# print(m)
# n = np.zeros(a.shape)
# n = (a[:,1:]-m[1:])/s[1:]
# print(np.mean(n, axis=0))
# print(np.std(n,axis=0))

# b = np.array([[1,2,3,4,5,6]])
# print(b)
# ssum = (b**2).sum()
# print(ssum)

# c = np.linspace(-1,1.5)
# d = c[0]
# print(d.shape)
# print(d.shape[0])