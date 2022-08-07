#!/usr/bin/env python
# coding: utf-8

# # Numpy

# In[97]:


import numpy as np


# In[110]:


arr = np.array([1,2,3,4])
print(arr)


# In[118]:


arr[len(arr) - 2]


# In[3]:


arr


# In[4]:


arr.ndim


# In[5]:


arr.shape


# In[43]:


z = np.array([[[2,3,1,4],[9,6,5,6]]])


# In[44]:


print(z)


# In[67]:


np.reshape(z,(4,2))


# In[46]:


z.reshape(2,4)


# In[6]:


len(arr)


# In[7]:


ar = np.array([
    [10,20,30,40],
    [30,60,90,120]
])


# In[8]:


ar


# In[9]:


ar.ndim


# In[10]:


ar.shape


# In[11]:


len(ar)


# In[12]:


ar.size


# In[13]:


type(ar)


# In[43]:


ar.dtype


# In[14]:


k = np.array(56)


# In[15]:


print(k)


# In[16]:


l = np.array([[[2,3,1,4],[9,6,5,6]],[[20,12,34,56],[13,25,46,32]]])


# In[17]:


print (l)


# In[18]:


l.ndim


# In[19]:


print(type (l))


# In[20]:


print(l.ndim)


# In[21]:


arr = np.array([1,2,3,4])


# In[23]:


print(arr)


# In[35]:


r = np.asarray(arr,dtype="float",order = "c")


# In[36]:


print(r)


# In[31]:


v = np.array ([[1,2,3,4],[3,4,5,6]])


# In[32]:


print(v)


# In[37]:


r = np.asarray(v, dtype="int", order = "f")


# In[38]:


print(r)


# In[81]:


d = np.eye(3, dtype= "int")


# In[82]:


print (d)


# In[85]:


w = np.full((5,4),333)


# In[86]:


print(w)


# In[92]:


q = np.ones((5,5), dtype = "int")


# In[93]:


print(q)


# In[102]:


np.full((3,3), 'cat')


# In[108]:


np.random.randint(2,9,(4,5))


# In[103]:


kv = np.arange(0, 100, 2, dtype="int")


# In[104]:


print(kv)


# In[105]:


ka = np.linspace(0,50,10)


# In[106]:


print(ka)


# In[107]:


ka.dtype


# # slicing of array

# [start index : stop index : step size]
# 
# [ : : ] = will take all the elements
# 
# Note: start index will include and stop index will exclude

# In[2]:


arr = np.array([1,2,3,4])
arr[::]


# In[3]:


arr[::2]


# In[7]:


arr[1:3:] #the 3 rd index will excluded when we call it as a stop index


# In[6]:


arr[2::]


# 2d array

# In[9]:


ar = np.array([[10,20,30,40],[30,60,90,120]])
print(ar)


# In[11]:


ar[0,0:3]


# In[12]:


ar[1,:]


# In[14]:


ar[1,1:4]


# In[15]:


ar[0:2 , 0:2]


# 3d array

# In[16]:


h = np.array([[[2,3,1,4],[9,6,5,6]],[[20,12,34,56],[13,25,46,32]]])


# In[18]:


h[0,0,0:4]


# In[22]:


h[0:2,0:2,0:2]


# # Axis

# axis = 0 means, coloum wise & axis = 1 means, row wise.

# In[24]:


bar =np.array([
    [2,3,5,7],
    [3,6,8,0],
    [1,12,3,44],
    [22,32,21,76]
])

print(bar)


# In[25]:


bar.shape 


# In[26]:


np.sort(bar,0) # axis = 0


# In[28]:


np.sort(bar,1) # axis = 1


# In[125]:


bar[3,3]


# In[126]:


bar[2]


# In[128]:


bar[1:3]


# In[129]:


bar[-2,-1]


# In[131]:


bar[1:3 ,[-2,-1]]


# In[132]:


bar[0,0] = 99
print(bar)


# In[133]:


bar[1] = bar[2]
print(bar)


# In[135]:


bar [[0,1,2,3],[0,1,2,3]] = [11,11,11,11]
print(bar)


# # 3 dimension array

# In[137]:


zoo = np.array([
[   [12,34],
    [32,13],
    [67,77],
],
[   [43,42],
    [87,65],
    [46,98],    
]
])

print(zoo)


# In[139]:


zoo.ndim


# In[140]:


zoo.shape


# In[142]:


zoo[1,:,1] = 999
print(zoo)


# In[146]:


zoo[0,:,0] = 555
print(zoo)


# In[148]:


zoo[1,:,0] = 333
print(zoo)


# In[149]:


zoo[0,:,1] = 222
print(zoo)


# # Array operation

# In[150]:


fo= np.array([[10,20,30],[40,50,60]])
bo= np.array([[70,80,90],[100,110,120]])


# In[151]:


print(fo)


# In[152]:


print(bo)


# In[153]:


fo + bo


# In[154]:


bo - fo


# In[155]:


fo * bo


# In[156]:


fo / bo


# Added 5 to every element of the matrix fo

# In[158]:


fo + np.full((2,3),5)


# In[159]:


fo + 5


# # Broadcasting

# allows arithmetic operations of arrays that are different size and shape

# In[29]:


a = np.array([2,3,4,5])
print(a)


# In[30]:


a + 2


# In[34]:


m = np.array([[1,2],[3,4],[5,6]])
print(m)


# In[37]:


m.shape


# In[35]:


n = np.array([5,7])


# In[38]:


n.shape


# In[36]:


m+n


# In[39]:


x = np.array([[10],[20],[30]])


# In[40]:


y = np.array([1,2,3])


# In[41]:


x+y


# #[1 1 1]     #[10 20 30]
# #[2 2 2]  #+ #[10 20 30]
# #[3 3 3]     #[10 20 30]

# # join & split two different arrays

# In[1]:


import numpy as np


# In[2]:


a = np.arange(6).reshape(2,3)


# In[3]:


a


# In[5]:


b = np.arange(7,13).reshape(2,3)
print(b)


# In[6]:


np.concatenate((a,b), axis = 0)


# In[7]:


np.concatenate((a,b),axis = 1)


# In[8]:


np.stack((a,b))


# In[9]:


np.stack((a,b), axis = 1)


# In[10]:


np.vstack((a,b))


# In[11]:


np.hstack((a,b))


# In[67]:


import numpy as np


# In[54]:


a = np.array ([[1,2], [3,4]])
b = np.array([[9,9], [8,8]])


# In[41]:


a


# In[47]:


b


# In[48]:


np.concatenate((a,b))


# In[56]:


p = np.arange (1,11)


# In[57]:


p


# In[58]:


np.split(p,2)


# In[61]:


np.hsplit(p,2)


# In[63]:


q = np.arange(1,13).reshape(6,2)
print(q)


# In[64]:


np.vsplit(q,2)


# In[65]:


np.hsplit(q,2)


# In[66]:


np.split(q,3)


# # sort

# In[68]:


a = np.array ([[50,40,30],[10,20,60]])
a


# In[69]:


np.sort(a)


# # inserting elements

# In[70]:


a = np.arange(1,12)


# In[71]:


a


# In[72]:


np.insert(a,11,12)


# In[73]:


np.insert(a,9,100)


# In[74]:


np.insert(a,(1,3,5,7),1000)


# In[78]:


a = np.array([[1,2],[3,4]])
a


# In[79]:


np.insert(a,1,23,axis = 0)


# In[80]:


np.insert(a,1,[23,24],axis = 0)


# In[81]:


a = np.arange(1,21)
a


# In[82]:


np.append(a,21)


# In[91]:


b= np.arange(6).reshape(2,3)
b


# In[92]:


np.append(b,[[4,5,6]],axis = 0)


# In[96]:


np.delete(a,[[4,5,6]],axis = 0)


# # Matrix

# In[98]:


a = np.array([[2,4],[6,8]])
b = np.array([[12,23],[14,25]])
print(a)
print(b)


# In[99]:


a+b


# In[100]:


a.dot(b)  #matrix multiplication or


# In[101]:


np.dot(a,b)  #dot product of arrays


# In[102]:


a.T


# In[106]:


a = np.matrix ([[1,2],[6,9]])
a


# In[108]:


b = np.matrix ([[2,4],[7,8]])
b


# In[109]:


a+b


# In[110]:


a - b 


# In[111]:


a.dot(b)


# In[112]:


a/b


# In[113]:


a.T


# In[114]:


b.T


# In[116]:


# linear algebra of matrices (inverse, power of a matrix, linear equations, determinants...)


# In[118]:


np.linalg.inv(b) # inverse of a matrix


# In[119]:


np.linalg.matrix_power(b,n=0) #identity matrix


# In[120]:


np.linalg.matrix_power(b,n=1) #matrix itself


# In[121]:


np.linalg.matrix_power(b,n=2) #squared matrix


# In[123]:


np.linalg.matrix_power(b,n= - 1) # inverse of the matrix


# In[124]:


np.linalg.matrix_power(b,n= -2)


# In[126]:


np.linalg.inv(b) #inverse


# # mathemetical & statistical operation

# shape of two array must be equal, second array must have atleast one dimension and the number & elements in that dimension should be equal to first array. ex: a = (3,3) mill not go with b = (2,2),or (2,2,2) they will go with anything b = (3,) like this. it is called broadcasting.

# In[127]:


a = ([[1,2],[3,4]])
b = ([5,6])
print(a)
print(b)


# In[131]:


np.add(a,b)


# In[132]:


np.subtract(a,b)


# In[133]:


np.multiply(a,b)


# In[134]:


np.mod(a,b) # is is the reminder of all divisor


# In[135]:


np.power(a,b)


# In[137]:


np.power(a,3)


# In[138]:


np.reciprocal(a)


# In[139]:


a = np.array ([[12,13,45],[23,21,34],[23,43,47]])
a


# In[141]:


np.amin(a) #min element of a array


# In[142]:


np.amax(a) #max element of a array


# In[144]:


np.amin(a,axis = 0) #gives the smallest value of each row


# In[146]:


np.amin(a,axis = 1)


# In[145]:


np.amax(a,axis = 1) #max value


# In[147]:


np.amax(a,axis = 0)


# In[148]:


np.average(a)


# In[149]:


np.mean(a)


# In[150]:


np.mean(a,axis =0)


# In[151]:


np.mean(a,axis =1)


# In[152]:


a


# In[153]:


np.median(a)


# In[155]:


np.median(a, axis = 0)


# In[156]:


np.median(a, axis =1)


# In[157]:


np.var(a)


# In[158]:


np.std(a)


# In[ ]:




