import numpy
import math
from solution import solution
import random
import time

def SSA(objf,lb,ub,dim,SearchAgents_No,Max_iter):

    convergence_curve = numpy.zeros(Max_iter)
    Positions=numpy.random.uniform(0,1,(SearchAgents_No,dim))*(ub-lb)+lb
    FoodPosition = numpy.zeros(dim)
    FoodFitness = float("inf")
    FoodPosition_history = numpy.zeros((Max_iter,dim))

    #*****************************
    s=solution()
    print("SSA is optimizing  \""+objf.__name__+"\"")
    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    #*****************************

    SalpFitness = numpy.zeros(SearchAgents_No)
    for i in range(0,SearchAgents_No):
        SalpFitness[i] = objf(Positions[i,:])

    FoodFitness = numpy.amin(SalpFitness)
    FoodPosition = Positions[numpy.argmin(SalpFitness),:]
    convergence_curve[0] = FoodFitness
    FoodPosition_history[0,:] = FoodPosition

    l=1
    while l < Max_iter:

        c1 = 2*math.exp(-(4*(l/Max_iter))**2)
        for i in range(0,SearchAgents_No):
            if i<=SearchAgents_No/2:
                for j in range(0,dim):
                    c2 = random.random()
                    c3 = random.random()
                    if c3 < 0.5 :
                        Positions[i,j] = FoodPosition[j] + c1*((ub-lb)*c2 + lb)
                    else:
                        Positions[i,j] = FoodPosition[j] - c1*((ub-lb)*c2 + lb)
            else:
                point1 = Positions[i-1,:]
                point2 = Positions[i,:]
                Positions[i,:] = (point1 + point2)/2


        for i in range(0,SearchAgents_No):
            Positions[i,:] = numpy.clip(Positions[i,:],lb,ub)
            SalpFitness[i] = objf(Positions[i,:])

        if convergence_curve[l-1] < numpy.amin(SalpFitness):
            FoodFitness = convergence_curve[l-1]
            FoodPosition = FoodPosition_history[l-1,:]
        else:
            FoodFitness = numpy.amin(SalpFitness)
            FoodPosition = Positions[numpy.argmin(SalpFitness),:]

        convergence_curve[l] = FoodFitness
        FoodPosition_history[l,:] = FoodPosition
        print(['At iteration '+ str(l)+ ' the best fitness is '+ str(FoodFitness)])
        l = l+1

    timerEnd=time.time()
    s.best = FoodFitness
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="SSA"
    s.objfname=objf.__name__

    return s
