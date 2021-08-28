from numpy import *
import numpy as np # always need it
import scipy as sp # often use it
import pylab as pl # do the plots
import ot # ot
import ot.plot
import matplotlib.pyplot as plt
import math
import os


L = 1000000
H = 10000
g = 10
f = 0.0001
theta0 = 300
dt= 1000
s = -0.000003
a = g*s/(f*theta0)
N = 0.000025
t = sinh(0.25)/cosh(0.25)
A1= -(1-0.25/t)
A2 =math.sqrt((0.25-t)*(1/t-0.25))




n = 2500
n1 = 400
B= np.repeat(1/n1, n1)
D = np.repeat(1/n, n)
c= math.sqrt(n)

x= random.rand(n)
z= random.rand(n)
x = (x-0.5)*2*L
z=  z* H

X = np.zeros(n1)
Z = np.zeros(n1)
x1= random.rand(n1)
z1= random.rand(n1)
x1 = (x1-0.5)*2*L
z1=  z1* H

eta = -(math.pi)*f/(math.sqrt(N))*(A1*cosh(0.5*z1/H)*sin(math.pi*x1/L)+A2*sinh(0.5*z1/H)*cos(math.pi*x1/L))
tao = A1*sinh(0.5*z1/H)*cos(math.pi*x1/L)-A2*cosh(0.5*z1/H)*sin(math.pi*x1/L)
X = x1+5/3*math.sqrt(N)/(f*f)*eta
Z = N/(f*f)*(z1+H/2)+5/3*math.sqrt(N)/(f*f)*tao

#X = x1+(1.6*g*g*H*H/(300*f*f*L*L*theta0))*sin(math.pi*((x1/L)+(z1/H)))*(1/L)
#Z = (g/(theta0*f*f))*((40*z1/H)+285)+(1.6*g*g*H*H/(300*f*f*L*L*theta0))*sin(math.pi*((x1/L)+(z1/H)))*(1/H)

y = zeros([n1,2])
y[0:n1,0]=2*L
# compute t_ij
def C(cafe_pos,A):
    AA= cafe_pos-y
    BB= cafe_pos+y
    # minimize the distance  
    M1 = ot.dist(cafe_pos,A)
    M2 = ot.dist(AA,A)
    M3 = ot.dist(BB,A) 
    M4 = np.minimum(M1,np.minimum(M2,M3))
    M  = np.array(M4) 
    T  = ot.emd(B, D, M)
    return(T)
# closed to zero 
def G(K):
    H = (np.vstack((K,K-2*L,K+2*L))).T
    Indicator = 1*(H**2 == tile((H**2).min(axis=-1).reshape(n,1), 3))
    Q = sum(H*Indicator,axis=-1)
    return(Q)
# compute a*dt*sum_j(t_ij*(x_j-X_i))
def R(S,T):
    W = np.zeros(n1)
    for i in range(n1):
        Q1 = x-S[i]
        Q2 = G(Q1)
        W[i]=a*dt*(T[i,:]@(Q2.T)*n1)
    return(W) 
# Kinetic Energy
def F1(D,T):
    W = np.zeros(n1)
    for i in range(n1):
        Q1 = x-D[i]
        Q2 = G(Q1)
        W[i]=0.5*(T[i,:]@(Q2*Q2))
    WW=sum(W)
    return(WW) 
# Potential Energy
def F2(P,T):
    W = np.zeros(n1)
    for i in range(n1):
        Q1 = z-H/2
        Q2 = P[i]*Q1
        W[i]=T[i,:]@Q2
    WW=sum(W)
    return(WW) 



Time = 2000
# initialize the Energy
E = np.zeros(Time)
Kin = np.zeros(Time)
Pot = np.zeros(Time)
# Euler method
for k in range(Time):
    cafe_pos = np.zeros((n1,2))
    # update the cafe_pos(the X,Z in the last iteration)
    cafe_pos = (np.vstack((X,Z))).T
    A = (np.vstack((x,z))).T
    
    # update t_ij
    T = C(cafe_pos,A)
    
    
   
    
    # update the X,Z 
   
        
    Z=Z+R(X,T)
    X=X-a*dt*(T@z*n1-H/2) 
    X= mod(X+L,2*L)-L
    # +/-
    Kin[k]=F1(X,T)
    Pot[k]=-F2(Z,T)
    E[k] = Kin[k]+Pot[k] 


    # save figure for every 10 time step   
    if k%50 == 0:
       

       Xhat = (T.T)@X*n
       Xhat= mod(Xhat+L,2*L)-L
       Zhat = (T.T)@Z*n
       v = f*G(Xhat-x)
       thetaprime = Zhat*f*f*theta0/g 

       plt.figure()
       plt.plot(X[:],Z[:],'.')
       plt.savefig('Geostrophic points %d.png'%k, dpi=200)

       plt.figure()
       plt.scatter(x,z,c=v,alpha=1)
       plt.colorbar()
       plt.savefig('velocity %d.png'%k, dpi=200)


       plt.figure()
       plt.scatter(x,z,c=thetaprime,alpha=1)
       plt.colorbar()
       plt.savefig('temperature %d.png'%k, dpi=200)


       plt.figure()
       plt.plot(E[:k])
       plt.plot(Kin[:k])
       plt.plot(Pot[:k])
       plt.legend(['Energy', 'Kinetic Energy', 'Potential Energy'], loc='upper left')
       plt.savefig('Total Energy %d.png'%k, dpi=200)
    








