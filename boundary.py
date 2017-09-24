from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import itertools
import math as m
from numpy.linalg import inv
Data=pd.read_csv('../credit.txt', sep=',',header=None)
iter_newton=12
X,Y,R=[],[],[]
f0,f1,f2=[],[],[]
cost=[]
lamba=0 #lambda
alpha=0.01  #learning rate

weigth=[0,1,1,1,1,1]
L=np.diag(weigth)

#print(L)

for i in Data.itertuples():
	tmp=np.ones(3)
	#tmp[0]=1
	tmp[1]=i[1]
	tmp[2]=i[2]
	#tmp[3]=i[3]
	if i[3]==1:
		plt.plot(i[1],i[2],'go')
	else:
		plt.plot(i[1],i[2],'ro')
	X.append(tmp)
	Y.append(i[3])

#print(w0,w1,w2)
def create_R(X,W,n):
	#print(X.shape, W.shape)
	X1=np.zeros((n,6))
	for i in range(n):
		x1=X[i,[1]]
		x2=X[i,[2]]
	#	print(x1,x2)
		x=np.array([1,x1,x2,x1*x2,x1**2,x2**2])
		X1[i]=x
	X1=np.matrix(X1)
	y=X1*W
		
	for i in y:
		h=1/(1+m.exp(-i))
		a=h*(1-h)
		f0.append(a)

	
	R=np.diag(f0)
	return R,X1
	
 
def newton_raphson(H,X,W,Y,n):   #A  X
	
	h=np.zeros((100,1))
	Y=np.matrix(Y)
	fx=X*W
	K1=(inv(H)*(X.transpose())) 
	for i in range(n):
		h[i]=1/(1+(m.exp(-fx[i])))
	h=h-Y.transpose()
	W=W-(K1*h)
	return W

def plot_boundary(W,x1,x2):
	#print('plot boundary')
	X=np.array([1,x1,x2,x1*x2,x1**2,x2**2])
	X=np.matrix(X)
	y=X*W
	
	h=1/(1+(m.exp(-y)))
	
	if(h>0.25677):  #0.3
 		return 1
	else:
		return 0
	

X=np.matrix(X)
W=np.matrix([1,0.05,0.05,0.05,0.05,0.05])
W=W.transpose()

n=(len(Data))
R,X1=create_R(X,W,n)
H=((X1.transpose())*R*X1)+(lamba*L)/n 

#print(H.shape)
for i in range(iter_newton):
	W=newton_raphson(H,X1,W,Y,n)
#print(W)

x1=np.linspace(1,6,100)
x2=np.linspace(1,6,100)
X,Y=np.meshgrid(x1,x2)
Z=np.zeros((len(x1),len(x1)))
for i in range(len(x1)):
	for j in range(len(x2)):
		
		Z[i,j]=plot_boundary(W,x1[i],x2[j])
		
plt.contour(x1-1,x2+1,Z,[0,1])
plt.show()
print('*****************************************************************')
#classification(W[0],W[1],W[2], X,n)