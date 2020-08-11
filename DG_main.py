import numpy as np
import matplotlib.pyplot as plt
import pdb
import DG_routines as DG
import scipy as sci
import scipy.sparse as sps
import matplotlib.animation as animation


import scipy.io


# animation function.  This is called sequentially
def animateSolution(x,time,sol_list,movie_name='pipe_flow_simulation'):
    fig = plt.figure()
    ax = plt.axes(xlim=(x[0], x[-1]), ylim=(np.min(sol_list),np.max(sol_list)))
    #ax = plt.axes(xlim=(x[0], x[-1]), ylim=(-1,1))

    #ax = plt.axes(xlim=(x[0], x[-1]), ylim=(np.min(sol_list),1003))

    plt.grid(True)
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        plt.title(str(time[i]))
        y = sol_list[i]
        line.set_data(x, y)
        return line,

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(sol_list), interval=20, blit=True)

    # save the animation as mp4 video file
    anim.save(movie_name + '.mp4',writer=writer)


alpha = 0
beta = 0

N = 2
K = 450

xmin = 0.
xmax = 1.


gamma = 1.4

DG_model_euler = DG.Euler1D(xmin=xmin, xmax=xmax, K=K, N=N)
DG_model_euler.StartUp()


xVec = np.reshape(DG_model_euler.x, (N + 1) * K, 'F')


q1init = np.ones((N+1,K))
q1init[np.argwhere(DG_model_euler.x>=0.5)[:,0],np.argwhere(DG_model_euler.x>=0.5)[:,1]] = 0.125
#q1init = np.exp(-0.5*np.power((DG_model_euler.x-0.5)/0.1,2)) + 1
q2init = np.zeros((N+1,K))
q3init = 1/(gamma-1)*np.ones((N+1,K))
q3init[np.argwhere(DG_model_euler.x>=0.5)[:,0],np.argwhere(DG_model_euler.x>=0.5)[:,1]] = 1/(gamma-1)*0.1
#q3init =  np.exp(-0.5*np.power((DG_model_euler.x-0.5)/0.1,2)) + 0.2


solq1,solq2,solq3, time = DG_model_euler.solve(q1init,q2init,q3init, FinalTime=0.2,implicit=True,stepsize=5e-5)
#%%
rho = []
u = []
for i in range(len(solq1)):
    rho.append(np.reshape(solq1[i], (N + 1) * K, 'F'))
    u.append(np.reshape(solq2[i]/solq1[i], (N + 1) * K, 'F'))

#%%

xVec = np.reshape(DG_model_euler.x, (N + 1) * K, 'F')
plt.figure()
plt.plot(xVec,u[-1])
plt.grid(True)
plt.legend(['u'])
plt.show()

plt.figure()
plt.plot(xVec,rho[-1])
plt.grid(True)
plt.legend(['rho'])
plt.show()



animateSolution(xVec,time[0:-1:2],u[0:-1:2])



'''


for K in test_vec:

    eps1 = np.concatenate((np.ones((1, np.floor(K/2).astype(int))), 2 * np.ones((1, np.floor(K/2).astype(int)))),axis=1)
    mu1 = np.ones((1, K))

    epsilon = np.ones((N+1,1))*eps1
    mu = np.ones((N+1,1))*mu1

    DG_model_maxwell = DG.DG_1D(xmin=xmin, xmax=xmax, K=K, N=N)
    DG_model_maxwell.StartUp()

    Einit = np.sin(np.pi*DG_model_maxwell.x)
    Einit[np.argwhere(DG_model_maxwell.x>0)[:,0],np.argwhere(DG_model_maxwell.x>0)[:,1]] = 0
    Hinit = np.zeros((N+1,K))

    xVec = np.reshape(DG_model_maxwell.x, (N + 1) * K, 'F')

    solE,solH, time = DG_model_maxwell.Maxwell1D(Einit,Hinit,epsilon,mu, FinalTime=10.)
    solVec = []
    for i in range(len(solE)):
        solVec.append(np.reshape(solH[i], (N + 1) * K, 'F'))
    #exactSol = []
    #for i in range(len(time)):
    #    exactSol.append(np.sin(DG_model.x-2*np.pi*time[i]))
    #error.append(1/(DG_model.x.shape[0]*DG_model.x.shape[1]) * np.linalg.norm(np.asarray(sol)-np.asarray(exactSol))**2)
xVec = np.reshape(DG_model_maxwell.x, (N + 1) * K, 'F')
plt.figure()
plt.plot(xVec,solVec[0])
plt.plot(xVec,solVec[np.floor(len(solVec)/2).astype(int)])
plt.plot(xVec,solVec[-1])
plt.show()

plt.figure()
plt.loglog(test_vec,error,'.-',markersize=15,linewidth=2)
plt.loglog(test_vec,1/np.power(test_vec,10),'.-',markersize=15,linewidth=2)
plt.show()

N = 3

xmin = 0.
xmax = 1.

error = []
test_vec = range(1,5)
K = 5
N = 2
#test_vec = [1e-1,1e-2,1e-3,1e-4]
stepsize=1e-3
implicit = True
plt.figure()
for N in test_vec:

    DG_model = DG.Advection1D(xmin=xmin, xmax=xmax, K=K, N=N)
    DG_model.StartUp()
    uinit = np.sin(DG_model.x)

    xVec = np.reshape(DG_model.x, (N + 1) * K, 'F')

    sol, time = DG_model.solve(uinit, FinalTime=.5,implicit=implicit, stepsize=stepsize,order=2)

    #pdb.set_trace()
    if implicit:
        solVec = []
        for i in range(len(sol)):
            solVec.append(np.reshape(sol[i], (N + 1) * K, 'F'))
        exactSol = []
    else:
        solVec = sol

    for i in range(len(time)):
        exactSol.append(np.sin(DG_model.x.flatten('F')-2*np.pi*time[i]))
    error.append(1/(DG_model.x.shape[0]*DG_model.x.shape[1]) * np.linalg.norm(np.asarray(sol)-np.asarray(exactSol))**2)

    plt.plot(xVec, solVec[-1])
    plt.grid(True)
plt.legend(['stepsize = {:0.5f}'.format(test_vec[0]), 'stepsize = {:0.5f}'.format(test_vec[1]),])
plt.show()

plt.figure()
plt.loglog(test_vec,error,'.-',markersize=15,linewidth=2)
plt.loglog(test_vec,1/np.power(test_vec,2),'.-',markersize=15,linewidth=2)
plt.grid(True)
plt.legend(['DG Error', 'h^2'])
plt.show()
plt.figure()
plt.plot(xVec,solVec[0])
plt.plot(xVec,solVec[np.floor(len(solVec)/2).astype(int)])
plt.plot(xVec,solVec[-1])
plt.grid(True)
plt.legend(['t={:0.1f}'.format(time[0]), 't={:0.1f}'.format(time[np.floor(len(solVec)/2).astype(int)]),'t={:0.1f}'.format(time[-1])])
plt.show()


animateSolution(xVec,time[0:-1:10],solVec[0:-1:10],movie_name='advection')

'''
