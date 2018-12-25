import math
import numpy
import time
from solution import solution

def S_func(r):
    f = 0.5
    l = 1.5
    o = f*math.exp(-r/l) - math.exp(-r)
    return o

def GOA(objf,lb,ub,dim,N,Max_iter):

    if dim%2 != 0:
        dim = dim +1

    GrassHopperPositions = numpy.random.uniform(0,1,(N,dim)) *(ub-lb)+lb
    GrassHopperFitness = numpy.zeros(N)
    convergence_curve = numpy.zeros(Max_iter)
    TargetFitness = float("inf")
    TargetPosition = numpy.zeros(dim)
    TargetPosition_history = numpy.zeros((Max_iter,dim))

    for i in range(0,N):
        GrassHopperFitness[i] = objf(GrassHopperPositions[i,:])

    TargetFitness = numpy.amin(GrassHopperFitness)
    TargetPosition = GrassHopperPositions[numpy.argmin(GrassHopperFitness),:]
    TargetPosition_history[0,:] = TargetPosition
    convergence_curve[0] = TargetFitness
    #********************************************************************
    s=solution()
    print("GOA is optimizing  \""+objf.__name__+"\"")
    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    #********************************************************************
    cMax = 1
    cMin = 0.00001
    sigma = 3

    l=1
    while l<Max_iter:

        GrassHopperPositions_temp = numpy.zeros((N,dim))

        #c = cMax-l*((cMax-cMin)/Max_iter)
        c = math.exp(-0.5*(l/(Max_iter/sigma))**2)

        temp = GrassHopperPositions
        for i in range(0,N):

            S_i_total = numpy.zeros(dim)
            for j in range(0,N):
                if i != j:

                    Dist = numpy.linalg.norm(temp[i,:] - temp[j,:])
                    r_ij_vec = (temp[j,:]-temp[i,:])/(Dist+numpy.finfo(float).eps)
                    xj_xi= 2 + Dist%2
                    s_ij = ((ub-lb)*c/2)*S_func(xj_xi)*r_ij_vec
                    S_i_total = S_i_total + s_ij


            X_new = c*S_i_total + TargetPosition
            GrassHopperPositions_temp[i,:] = X_new

        GrassHopperPositions = GrassHopperPositions_temp

        for i in range(0,N):
            GrassHopperPositions[i,:] = numpy.clip(GrassHopperPositions[i,:],lb,ub)
            GrassHopperFitness[i] = objf(GrassHopperPositions[i,:])

        if convergence_curve[l-1] < numpy.amin(GrassHopperFitness):
            TargetFitness = convergence_curve[l-1]
            TargetPosition = TargetPosition_history[l-1,:]
        else:
            TargetFitness = numpy.amin(GrassHopperFitness)
            TargetPosition = GrassHopperPositions[numpy.argmin(GrassHopperFitness),:]

        convergence_curve[l] = TargetFitness
        TargetPosition_history[l,:] = TargetPosition

        print('At iteration '+ str(l)+ ' the best fitness is '+ str(TargetFitness))
        l = l+1

    timerEnd=time.time()
    s.best = TargetFitness
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="GOA"
    s.objfname=objf.__name__

    return s
