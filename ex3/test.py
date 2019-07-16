import numpy as np
#b = (np.array(range(15))+1)/10
#a = np.reshape(b,(5,3), order='F')
#e = np.append(np.ones((5,1)),np.reshape((np.array(range(15))+1)/10,(5,3), order='F'),axis=1)
#print(e)

#Y_t = np.zeros(a.shape)
#Y_t = np.array([[1],[0],[1],[0],[1]])>=0.5
#print(Y_t)
#diff = ((a-b)**2).sum()
#print(diff)
# b = np.zeros(a.shape)

# print(c)
# b[a==c]=1
# print(b)

a = np.random.randint(0,10,(10,4))
d = np.zeros((10,1))
print(a)
c = np.max(a,axis=1, keepdims=True)
b=np.argmax(a,axis=1)

print(b)