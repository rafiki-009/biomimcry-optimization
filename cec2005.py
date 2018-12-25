import math
import numpy
from scipy import linalg
import scipy.io as sio
import random

def prod( it ):
    p= 1
    for n in it:
        p *= n
    return p

def Ufun(x,a,k,m):
    y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
    return y

def rot_matrix(dim,c):
    A = numpy.random.normal((dim,dim))
    P = cGram_Schmidt(A)
    Q = cGram_Schmidt(A)
    u = numpy.random((1,dim))
    D = c*((u - min(u))/(max(u)-min(u)))
    D = numpy.diag(D)
    M = P@D@Q
    return M

"""Classical Gram-Schmidt (CGS) algorithm"""
def cGram_Schmidt(A):
    m, n = A.shape
    R = numpy.zeros((n, n))
    Q = numpy.empty((m, n))
    R[0, 0] = linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]
    for k in range(1, n):
        R[:k-1, k] = numpy.dot(Q[:m, :k-1].T, A[:m, k])
        z = A[:m, k] - numpy.dot(Q[:m, :k-1], R[:k-1, k])
        R[k, k] = linalg.norm(z) ** 2
        Q[:m, k] = z / R[k, k]
    return Q, R

'''1.Shifted Sphere Function'''
def F1(x):
    f_bias = -450
    dim = len(x)
    o = -10*numpy.ones(dim)
    x = x - o
    s = numpy.sum(x**2)
    return s + f_bias

'''2.Shifted Schwefel's Problem 1.2'''
def F2(x):
    f_bias = -450
    dim=len(x)
    o = -10*numpy.ones(dim)
    x = x - o
    s=0
    for i in range(1,dim):
        s = s + (numpy.sum(x[0:i]))**2
    return s + f_bias

'''3.Shifted Rotated High Conditioned Elliptic Function'''
def F3(x):
    f_bias = -450
    dim = len(x)
    o = 0*numpy.ones(dim)
    if dim == 2: mat_contents = sio.loadmat('elliptic_M_D2.mat')
    elif dim == 10: mat_contents = sio.loadmat('elliptic_M_D10.mat')
    elif dim == 30: mat_contents = sio.loadmat('elliptic_M_D30.mat')
    elif dim == 50: mat_contents = sio.loadmat('elliptic_M_D50.mat')

    z = (x - o)@mat_contents['M']
    s = 0
    for i in range(1,dim+1):
        s = s + ((10**6)**((i-1)/(dim-1)))*z[i-1]**2
    return s+f_bias

'''4.Shifted Schwefel's Problem 1.2 with Noise in Fitness'''
def F4(x):
    f_bias = -450
    dim = len(x)
    o = -10*numpy.ones(dim)
    x = x - o
    s = 0
    for i in range(0,dim):
        s = s + (numpy.sum(x[0:i]))**2
    noise = 1 + 0.4*abs(random.random())
    return s*noise + f_bias

'''5.Schwefel's Problem 2.6'''
def F5(x):
    dim=len(x)
    f_bias = 0
    opt_content = sio.loadmat("schwefel_206_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    A = opt_content['A']
    A = A[0:dim,0:dim]
    o[0:int(dim/4)] = -100
    o[max(int(0.75*dim),1):dim] = 100
    B = A*numpy.transpose(o)
    f = numpy.max(abs(A*numpy.transpose(x)-B))
    return f + f_bias

'''6.Shifted Rosenbrock's Function'''
def F6(x):
    dim = len(x)
    f_bias = 390
    opt_content = sio.loadmat("rosenbrock_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    x = x - o
    f = numpy.sum(100*(x[1:dim]-(x[0:dim-1]**2))**2+(x[0:dim-1]-1)**2)
    return f + f_bias

'''7.Shifted Rotated Griewank's Function'''
def F7(x):
    f_bias = -180
    dim = len(x)
    opt_content = sio.loadmat("griewank_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    c = 3
    if dim ==2: mat_contents = sio.loadmat('griewank_M_D2.mat')
    elif dim ==10: mat_contents = sio.loadmat('griewank_M_D10.mat')
    elif dim ==30: mat_contents = sio.loadmat('griewank_M_D30.mat')
    elif dim ==50: mat_contents = sio.loadmat('griewank_M_D50.mat')
    else:
        M=rot_matrix(dim,c)
        M = M*(1 + 0.3*numpy.random.normal((dim,dim)))
    M = mat_contents["M"]
    x = x - o
    x = x@M
    f = 1
    for i in range(1,dim+1):
        f=f*math.cos(x[i-1]/math.sqrt(i))
    f= numpy.sum(x**2)/4000 - f +1
    return f +f_bias

'''8.Shifted Rotated Ackley's Function with Global Optimum on Bounds'''
def F8(x):
    dim = len(x)
    f_bias = -140
    opt_content = sio.loadmat("ackley_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    o[1::2] = -32*o[::2]
    c = 100
    if dim ==2: mat_contents = sio.loadmat('ackley_M_D2.mat')
    elif dim ==10: mat_contents = sio.loadmat('ackley_M_D10.mat')
    elif dim ==30: mat_contents = sio.loadmat('ackley_M_D30.mat')
    elif dim ==50: mat_contents = sio.loadmat('ackley_M_D50.mat')
    else: M = rot_matrix(dim,c)
    x= x-o
    M = mat_contents["M"]
    x = x@M
    f=numpy.sum(x**2)
    f = 20 - 20*math.exp(-0.2*math.sqrt(f/dim)) - math.exp(numpy.sum(numpy.cos(2*math.pi*x))) + math.exp(1)
    return f + f_bias

'''9.Shifted Rastringn's Function'''
def F9(x):
    dim = len(x)
    opt_content = sio.loadmat("rastrigin_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    f_bias = -330
    x = x - o
    f = numpy.sum(x**2 - 10*numpy.cos(2*math.pi*x) + 10)
    return f + f_bias

'''10.Shifted Rotated Rastringn's Function'''
def F10(x):
    dim = len(x)
    f_bias = -330
    opt_content = sio.loadmat("rastrigin_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    c = 2
    if dim ==2: mat_contents = sio.loadmat('rastrigin_M_D2.mat')
    elif dim ==10: mat_contents = sio.loadmat('rastrigin_M_D10.mat')
    elif dim ==30: mat_contents = sio.loadmat('rastrigin_M_D30.mat')
    elif dim ==50: mat_contents = sio.loadmat('rastrigin_M_D50.mat')
    else:M = rot_matrix(dim,c)
    M = mat_contents["M"]
    x = x - o
    x = x@M
    f = numpy.sum(x**2 - 10*numpy.cos(2*math.pi*x) + 10)
    return f + f_bias

'''11.Shifted Rotated Weierstrass Function'''
def F11(x):
    dim = len(x)
    f_bias = 90
    opt_content = sio.loadmat("weierstrass_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    c = 5
    if dim ==2: mat_contents = sio.loadmat('weierstrass_M_D2.mat')
    elif dim ==10: mat_contents = sio.loadmat('weierstrass_M_D10.mat')
    elif dim ==30: mat_contents = sio.loadmat('weierstrass_M_D30.mat')
    elif dim ==50: mat_contents = sio.loadmat('weierstrass_M_D50.mat')
    else:M = rot_matrix(dim,c)
    M = mat_contents["M"]
    x = x - o
    x = x@M
    x = x + 0.5
    a = 0.5
    b = 3
    kmax = 20
    c1 = a**numpy.arange(kmax+1)
    c2 = 2*math.pi*b**numpy.arange(kmax+1)
    c = numpy.sum(c1*numpy.cos(c2*0.5))
    f =0
    for i in range(dim):
        f = f + numpy.sum(c1*numpy.cos(c2*x[i]))
    f = f - c*dim
    return f + f_bias

'''12.Schwefel's problem'''
def F12(x):
    dim = len(x)
    f_bias = -460
    opt_content = sio.loadmat('schwefel_213_data.mat')
    alpha = opt_content['alpha']
    alpha = alpha[0,0:dim]
    a = opt_content['a']
    a = a[0:dim,0:dim]
    b = opt_content['b']
    b = b[0:dim,0:dim]
    A = numpy.sum(a*numpy.sin(alpha) + b*numpy.cos(alpha))
    B = numpy.sum(a*numpy.sin(x) + b*numpy.cos(x))
    f = numpy.sum((A-B)**2)
    return f + f_bias

'''13.Expanded Extended Griewank's plus Rosenbrock's Function (F8F2)'''
def F13(x):
    dim = len(x)
    f_bias = -130
    opt_content = sio.loadmat("EF8F2_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    x = x - o +1
    fit =0
    for i in range(dim-1):
        fit = fit + F8F2(x[i],x[i+1])
    fit = fit + F8F2(x[-1],x[0])
    return fit + f_bias

def F8F2(a,b):
    f2 = 100*(a**2 - b**2) + (1-a)**2
    f = 1 + f2**2/4000 - math.cos(f2)
    return f

'''14.Expanded Rotated Extended Scaffer's F6'''
def F14(x):
    dim = len(x)
    f_bias = -300
    opt_content = sio.loadmat("E_ScafferF6_func_data.mat")
    o = opt_content['o']
    o = o[0,0:dim]
    c = 3
    if dim ==2: mat_contents = sio.loadmat('E_ScafferF6_M_D2.mat')
    elif dim ==10: mat_contents = sio.loadmat('E_ScafferF6_M_D10.mat')
    elif dim ==30: mat_contents = sio.loadmat('E_ScafferF6_M_D30.mat')
    elif dim ==50: mat_contents = sio.loadmat('E_ScafferF6_M_D50.mat')
    else:M = rot_matrix(dim,c)
    M = mat_contents["M"]
    x = x - o
    x = x@M
    f = 0
    for i in range(dim-1):
        f = f + fscaffer(x[i],x[i+1])
    f = f + fscaffer(x[-1],x[0])
    return f + f_bias

def fscaffer(x1,x2):
    f = 0.5 + (math.sin(math.sqrt(x1**2 + x2**2))**2 - 0.5)/(1 + 0.001*(x1**2 + x2**2))**2
    return f


def getFunctionDetails(a):

    # [name, lb, ub, dim]
    param = {  0: ["F1",-100,100],
               1 : ["F2",-100,100],
               2 : ["F3",-100,100],
               3 : ["F4",-100,100] ,
               4 : ["F5",-100,100],
               5 : ["F6",-100,100],
               6 : ["F7",-1.28,1.28],
               7 : ["F8",-32,32],
               8 : ["F9",-5,5],
               9 : ["F10",-5,5],
               10 : ["F11",-0.5,0.5] ,
               11 : ["F12",-100,100],
               12 : ["F13",-3,1],
               13 : ["F14",-100,100],
               14 : ["F15",-5,5],

            }
    return param.get(a, "nothing")
