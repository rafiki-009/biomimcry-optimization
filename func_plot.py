import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cec2005
import numpy

F1=False
F2=False
F3=False
F4=False
F5=False
F6=False
F7=False
F8=False
F9=False
F10=False
F11=False
F12=False
F13=False
F14=True
def getPlotDetails(a):
    param = {  0: ["F1",numpy.arange(-100,100,5),numpy.arange(-100,100,5)],
               1 : ["F2",numpy.arange(-100,100,5),numpy.arange(-100,100,5)],
               2 : ["F3",numpy.arange(-100,100,5),numpy.arange(-100,100,5)],
               3 : ["F4",numpy.arange(-100,0,2),numpy.arange(-100,0,2)+100] ,
               4 : ["F5",numpy.arange(-200,200,20),numpy.arange(-200,200,20)],
               5 : ["F6",numpy.arange(78,82,0.05),numpy.arange(78,82,0.05)],
               6 : ["F7",numpy.arange(-350,-250,2),numpy.arange(-350,-250,2)],
               7 : ["F8",numpy.arange(-32,32,1),numpy.arange(-32,32,1)],
               8 : ["F9",numpy.arange(-5,5,0.1),numpy.arange(-5,5,0.1)],
               9 : ["F10",numpy.arange(-5,5,0.1),numpy.arange(-5,5,0.1)],
               10 : ["F11",numpy.arange(-0.5,0.5,0.01),numpy.arange(-0.5,0.5,0.01)] ,
               11 : ["F12",numpy.arange(-3,3,0.1),numpy.arange(-3,3,0.1)],
               12 : ["F13",numpy.arange(-2,-1,0.02),numpy.arange(-2,-1,0.02)],
               13 : ["F14",numpy.arange(-90,-50,0.2),numpy.arange(-40,0,0.2)],
               14 : ["F15",numpy.arange(-5,5,0.1),numpy.arange(-5,5,0.1)],
               15 : ["F16",numpy.arange(-5,5,0.1),numpy.arange(-5,5,0.1)],

            }
    return param.get(a, "nothing")

benchmarkfunc = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14]
for i in range(0,len(benchmarkfunc)):
#    if benchmarkfunc[i]==True:
    func_axes = getPlotDetails(i)
    name = func_axes[0]
    x = func_axes[1]
    y = func_axes[2]

    L = len(x)
    f = numpy.zeros((L,L))

    for i in range(L):
        for j in range(L):
            myfunc = getattr(cec2005, name)
            f[i,j] = myfunc(numpy.array([x[i],y[j]]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x,y = numpy.meshgrid(x,y)
    surf = ax.plot_surface(x,y,f, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
