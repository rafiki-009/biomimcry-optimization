import numpy
import math
from solution import solution
import random
import time

def S_func(r):
    f = 0.5
    l = 1.5
    o = f*math.exp(-r/l) - math.exp(-r)
    return o

def IGOA(objf,lb,ub,dim,N,Max_iter):

    if dim%2 != 0:
        dim = dim +1

    convergence_curve = numpy.zeros(Max_iter)
    Positions = numpy.random.uniform(0,1,(N,dim))*(ub-lb) + lb
    TargetPosition = numpy.zeros(dim)
    TargetFitness = float("inf")
    TargetPosition_history = numpy.zeros((Max_iter,dim))
    memberFitness = numpy.zeros(N)

    #*****************************
    s=solution()
    print("IGOA is optimizing  \""+objf.__name__+"\"")
    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    #*****************************


    for i in range(0,N):
        memberFitness[i] = objf(Positions[i,:])

    TargetFitness = numpy.amin(memberFitness)
    TargetPosition = Positions[numpy.argmin(memberFitness),:]
    convergence_curve[0] = TargetFitness
    TargetPosition_history[0,:] = TargetPosition


    cMax = 1
    cMin = 0.00001
    sigma = 3.5

    l=1
    while l < Max_iter:
        Positions_temp = numpy.zeros((N,dim))
        c = math.exp(-0.5*(l/(Max_iter/sigma))**2)
        c1 = 2*math.exp(-(4*(l/Max_iter))**2)
        temp = Positions
        for i in range(0,N):
            S_i_total = numpy.zeros(dim)
            for j in range(0,N):
                if i != j:
                    Dist = numpy.linalg.norm(temp[i,:] - temp[j,:])
                    r_ij_vec = (temp[j,:]-temp[i,:])/(Dist+numpy.finfo(float).eps)
                    xj_xi= 2 + Dist%2
                    s_ij = ((ub-lb)*c/2)*S_func(xj_xi)*r_ij_vec
                    S_i_total = S_i_total + s_ij
            Positions_temp[i,:] = c*S_i_total + TargetPosition
            if i<=N/2:
                for j in range(0,dim):
                    c2 = random.random()
                    c3 = random.random()
                    if c3 < 0.5 :
                        Positions_temp[i,j] = TargetPosition[j] + c1*((ub-lb)*c2 + lb)
                    else:
                        Positions_temp[i,j] = TargetPosition[j] - c1*((ub-lb)*c2 + lb)
            else:
                point1 = Positions_temp[i-1,:]
                point2 = Positions_temp[i,:]
                Positions_temp[i,:] = (point1 + point2)/2
        Positions = Positions_temp

        for i in range(0,N):
            Positions[i,:] = numpy.clip(Positions[i,:],lb,ub)
            memberFitness[i] = objf(Positions[i,:])

        if convergence_curve[l-1] < numpy.amin(memberFitness):
            TargetFitness = convergence_curve[l-1]
            TargetPosition = TargetPosition_history[l-1,:]
        else:
            TargetFitness = numpy.amin(memberFitness)
            TargetPosition = Positions[numpy.argmin(memberFitness),:]

        convergence_curve[l] = TargetFitness
        TargetPosition_history[l,:] = TargetPosition
        print(['At iteration '+ str(l)+ ' the best fitness is '+ str(TargetFitness)])
        l = l+1

    timerEnd=time.time()
    s.best = TargetFitness
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="IGOA"
    s.objfname=objf.__name__

    return s
