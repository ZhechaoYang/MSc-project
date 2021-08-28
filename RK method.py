from numpy import *
import numpy as np # always need it
import scipy as sp # often use it
import pylab as pl # do the plots
import ot # ot
import ot.plot
import matplotlib.pyplot as plt
import math

n= 400
B = np.repeat(1/n, n)
D = np.repeat(1/n, n)
c= math.sqrt(n)
S= arange(0,1,1/c)
x,z = meshgrid (S,S)
x = x.reshape(x.size)
z = z.reshape(z.size)
L = 1000000
H = 10000
g = 10
f = 0.0001
theta0 = 300
dt= 1000
s = -0.000003
a = g*s/(f*theta0)
x = (x-0.5)*2*L
z = z* H-H/2

N = 0.000025
t = sinh(0.25)/cosh(0.25)
A1= -(1-0.25/t)
A2 =math.sqrt((0.25-t)*(1/t-0.25))

X = np.zeros(n)
Z = np.zeros(n)
eta = -(math.pi)*f/(math.sqrt(N))*(A1*cosh(0.5*z/H)*sin(math.pi*x/L)+A2*sinh(0.5*z/H)*cos(math.pi*x/L))
tao = A1*sinh(0.5*z/H)*cos(math.pi*x/L)-A2*cosh(0.5*z/H)*sin(math.pi*x/L)
X = x+5/3*math.sqrt(N)/(f*f)*eta
Z = N/(f*f)*(z+H/2)+5/3*math.sqrt(N)/(f*f)*tao

y = zeros([n,2])
y[0:n,0]=2*L
def C(cafe_pos,A):
    AA= cafe_pos-y
    BB= cafe_pos+y
    # minimize the distance  
    M1 = ot.dist(cafe_pos,A)
    M2 = ot.dist(AA,A)
    M3 = ot.dist(BB,A)
    M4 = np.minimum(M1,np.minimum(M2,M3))
    M  =   M4*M4
    M  = np.array(M) 
    T  = ot.emd(B, D, M)
    return(T)

def G(K):
    H = (np.vstack((K,K-2*L,K+2*L))).T
    Indicator = 1*(H**2 == tile((H**2).min(axis=-1).reshape(n,1), 3))
    Q = sum(H*Indicator,axis=-1)
    return(Q)

def R(S,T):
    W = np.zeros(n)
    for i in range(n):
        Q1 = x-S[i]
        Q2 = G(Q1)
        W[i]=a*(T[i,:]@(Q2.T)*n)
    return(W) 
 # Kinetic Energy

def F1(D,T):
    W = np.zeros(n)
    for i in range(n):
        Q1 = x-D[i]
        Q2 = G(Q1)
        W[i]=0.5*(T[i,:]@(Q2*Q2))
    WW=sum(W)
    return(WW) 
# Potential Energy
def F2(P,T):
    W = np.zeros(n)
    for i in range(n):
        Q1 = z
        Q2 = P[i]*Q1
        W[i]=T[i,:]@Q2
    WW=sum(W)
    return(WW)    

Time = 500
# initialize the Energy
E = np.zeros(Time)
Kin = np.zeros(Time)
Pot = np.zeros(Time)

for k in range(Time):
    cafe_pos = np.zeros((n,2))
    cafe_pos = (np.vstack((X,Z))).T
    A = (np.vstack((x,z))).T
    # update the cafe_pos(the X,Z in the last iteration)
    T = C(cafe_pos,A)
   
    k1x=-a*(T@z*n-H/2)
    W1 = R(X,T)
    k1z= W1   
    XP = X+dt/2*k1x 
    ZP = Z+dt/2*k1z
    XP = mod(XP+L,2*L)-L
    cafe_pos = (np.vstack((XP,ZP))).T
    T = C(cafe_pos,A)
    
    k2x=-a*(T@z*n-H/2)
    W2 = R(XP,T)
    k2z= W2     
    XP = X+dt/2*k2x 
    ZP = Z+dt/2*k2z
    XP = mod(XP+L,2*L)-L
    cafe_pos = (np.vstack((XP,ZP))).T
    T = C(cafe_pos,A)
    
    k3x=-a*(T@z*n-H/2)
    W3 = R(XP,T)
    k3z= W3    
    XP = X+dt*k3x 
    ZP = Z+dt*k3z
    XP = mod(XP+L,2*L)-L
    cafe_pos = (np.vstack((XP,ZP))).T
    T = C(cafe_pos,A)  
    
   
    k4x=-a*(T@z*n-H/2)
    W4 = R(XP,T)
    k4z= W4
    
    X=X+dt*(k1x/6+k2x/3+k3x/3+k4x/6)
    Z=Z+dt*(k1z/6+k2z/3+k3z/3+k4z/6)
        
    X= mod(X+L,2*L)-L 
    Kin[k]=F1(X,T)
    Pot[k]=-F2(Z,T)
    E[k] = Kin[k]+Pot[k] 
    
     
Xhat = (T.T)@X*n
Xhat= mod(Xhat+L,2*L)-L
Zhat = (T.T)@Z*n
v = f*G(Xhat-x)
thetaprime = Zhat*f*f*theta0/g

plt.figure()
plt.plot(X[:],Z[:],'.')
plt.show()

plt.figure()
plt.pcolor(v.reshape(int(c),int(c)))
plt.colorbar()
plt.show()

plt.figure()
plt.pcolor(thetaprime.reshape(int(c),int(c)))
plt.colorbar()
plt.show()

plt.figure()
plt.plot(E)
plt.plot(Kin)
plt.plot(Pot)
plt.legend(['Energy', 'Kinetic', 'Potential'], loc='upper left')
plt.show()