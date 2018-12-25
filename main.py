import SSA as ssa
import GOA as goa
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import WOA as woa
import FFA as ffa
import IGOA as igoa
import cec2005
import numpy
import time
import csv
import matplotlib.pyplot as plt


def selector(algo,func_details,popSize,Iter):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=30

    if(algo=='PSO'):
        x=pso.PSO(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='SSA'):
        x=ssa.SSA(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='GOA'):
        x=goa.GOA(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='IGOA'):
        x=igoa.IGOA(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='MVO'):
        x=mvo.MVO(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='GWO'):
        x=gwo.GWO(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='MFO'):
        x=mfo.MFO(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='CS'):
        x=cs.CS(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='BAT'):
        x=bat.BAT(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='WOA'):
        x=woa.WOA(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)
    if(algo=='FFA'):
        x=ffa.FFA(getattr(cec2005, function_name),lb,ub,dim,popSize,Iter)

    return x

F1=False
F2=False
F3=False
F4=False
F5=True
F6=False
F7=False
F8=False
F9=False
F10=False
F11=False
F12=False
F13=False
F14=False


optimizer = {'PSO' : False, 'SSA' :False , 'GOA' : True , 'IGOA':False ,\
            'MVO':False,'GWO':False,'MFO':False,'CS':False,\
            'BAT':False,'WOA':False,'FFA':False}
benchmarkfunc = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14]

NumOfRuns=1
PopulationSize = 30
Iterations = 1500
Export = True
Flag1 = True
Flag2 = True

my_convergence_curves=[]
count = 0

CnvgHeader=[]
for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

for algo,algo_value in optimizer.items():
    if algo_value == True:
        ExportToFile = algo + 'Iter =' + str(Iterations) + ').csv'
        ExportToFile2 = algo + 'Iter =' + str(Iterations) + ')(Best).csv'
        for j in range(0,len(benchmarkfunc)):
            if benchmarkfunc[j]==True:
                best_values = numpy.zeros(NumOfRuns)
                for k in range (0,NumOfRuns):
                    func_details=cec2005.getFunctionDetails(j)
                    x=selector(algo,func_details,PopulationSize,Iterations)
                    best_values[k] = x.best
                    my_convergence_curves.append(x.convergence)

                    if(Export==True):
                        '''export file with 1000 iterations'''
                        with open(ExportToFile, 'a',newline='\n') as out:
                            writer = csv.writer(out,delimiter=',')
                            if (Flag1==False): # just one time to write the header of the CSV file
                                header= numpy.concatenate([["Optimizer","objfname","ExecutionTime"],CnvgHeader])
                                writer.writerow(header)
                            a=numpy.concatenate([[x.optimizer,x.objfname,x.executionTime],x.convergence])
                            writer.writerow(a)
                        out.close()
                        Flag1 = True

                        '''export file with best values'''
                        with open(ExportToFile2, 'a',newline='\n') as snap:
                            writer = csv.writer(snap,delimiter=',')
                            if (Flag2==False): # just one time to write the header of the CSV file
                                header= numpy.concatenate([["Optimizer","objfname","ExecutionTime"],['best Value']])
                                writer.writerow(header)
                            a= [x.optimizer,x.objfname,x.executionTime,x.best]
                            writer.writerow(a)
                        snap.close()
                        Flag2=True # at least one experiment

algo = ['GOA','IGOA']
for i in range(len(my_convergence_curves)):
    plt.plot(numpy.arange(0,Iterations),my_convergence_curves[i],'--',label = algo[i])
plt.title('Convergence Curves')
plt.xlabel("Number of Iterations")
plt.ylabel("Fitness Value")
plt.legend()
plt.show()
