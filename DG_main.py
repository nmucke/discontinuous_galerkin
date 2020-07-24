import numpy as np
import matplotlib.pyplot as plt
import pdb
import DG_routines_jacobi as DG
import scipy as sci
import scipy.sparse as sps
import matplotlib.animation as animation


import scipy.io


# animation function.  This is called sequentially
def animateSolution(x,time,sol_list,gif_name='pipe_flow_simulation'):
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
    anim.save(gif_name + '.mp4',writer=writer)


alpha = 0
beta = 0

N = 2
#K = 50

xmin = 0.
xmax = 100.

error = []
test_vec = [1000]


for K in test_vec:

    c = 1400
    rho0 = 1000
    p0 = 1e5

    DG_model_pipe = DG.DG_1D(xmin=xmin, xmax=xmax, K=K, N=N)
    DG_model_pipe.StartUp()

    xVec = np.reshape(DG_model_pipe.x, (N + 1) * K, 'F')

    xl = 50

    mu1 = 50.
    mu2 = 60
    sigma = .01
    #q1init =  3 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((DG_model_pipe.x - mu1) / sigma, 2))
    #q1init +=  3 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((DG_model_pipe.x - mu2) / sigma, 2))
    #q1init += np.random.normal(0,1,q1init.shape)
    #q1init += rho0


    #q2init = np.zeros((N+1,K))

    #q1init = np.zeros(DG_model_pipe.x.shape)
    #q1init[np.argwhere(DG_model_pipe.x>=20)[:,0],np.argwhere(DG_model_pipe.x>=20)[:,1]] =rho0 + 1.25
    #q1init[np.argwhere(DG_model_pipe.x<20)[:,0],np.argwhere(DG_model_pipe.x<20)[:,1]] = rho0 + 3.5
    q2init = np.zeros((N+1,K))
    q1init = rho0*np.ones((N+1,K))

    solq1,solq2, time = DG_model_pipe.Pipe1D(q1init,q2init, FinalTime=0.2,c=c,rho0=rho0,p0=p0,xl=xl)
#%%
rho = []
u = []
p = []
for i in range(len(solq1)):
    rho.append(np.reshape(solq1[i], (N + 1) * K, 'F'))
    u.append(np.reshape(np.divide(solq2[i],solq1[i]), (N + 1) * K, 'F'))
    p.append(np.reshape(c*c*(solq1[i]-rho0)+p0, (N + 1) * K, 'F'))

#%%

xVec = np.reshape(DG_model_pipe.x, (N + 1) * K, 'F')
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

plt.figure()
plt.plot(xVec,p[-1])
plt.grid(True)
plt.legend(['p'])
plt.show()


#%%
#animateSolution(xVec,time[0:-1:20],u[0:-1:20])



'''
N = 5
#K = 50

xmin = 0.
xmax = 1.

error = []
test_vec = [500]# range(10,1,2)

for K in test_vec:
    gamma = 1.4

    DG_model_euler = DG.DG_1D(xmin=xmin, xmax=xmax, K=K, N=N)
    DG_model_euler.StartUp()

    xVec = np.reshape(DG_model_euler.x, (N + 1) * K, 'F')


    q1init = np.ones((N+1,K))
    q1init[np.argwhere(DG_model_euler.x>=0.5)[:,0],np.argwhere(DG_model_euler.x>=0.5)[:,1]] = 0.125
    #q1init = np.exp(-0.5*np.power((DG_model_euler.x-0.5)/0.1,2)) + 1
    q2init = np.zeros((N+1,K))
    q3init = 1/(gamma-1)*np.ones((N+1,K))
    q3init[np.argwhere(DG_model_euler.x>=0.5)[:,0],np.argwhere(DG_model_euler.x>=0.5)[:,1]] = 1/(gamma-1)*0.1
    #q3init =  np.exp(-0.5*np.power((DG_model_euler.x-0.5)/0.1,2)) + 0.2


    solq1,solq2,solq3, time = DG_model_euler.Euler1D(q1init,q2init,q3init, FinalTime=0.2)
#%%
rho = []
u = []
for i in range(len(solq1)):
    rho.append(np.reshape(solq1[i], (N + 1) * K, 'F'))
    u.append(np.reshape(solq2[i]/solq1[i], (N + 1) * K, 'F'))


xVec = np.reshape(DG_model_euler.x, (N + 1) * K, 'F')
plt.figure()
plt.plot(xVec,rho[-1])
plt.plot(xVec,u[-1])
plt.legend(['rho','u'])
plt.show()
'''

'''
N = 2
#K = 50

xmin = -5.
xmax = 5.

error = []
test_vec = [750]# range(10,1,2)
for K in test_vec:

    c = 1400
    rho0 = 1000
    p0 = 1

    DG_model_pipe = DG.DG_1D(xmin=xmin, xmax=xmax, K=K, N=N)
    DG_model_pipe.StartUp()

    xVec = np.reshape(DG_model_pipe.x, (N + 1) * K, 'F')

    mu = 0.
    sigma = .1
    #pinit = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*np.power((DG_model_pipe.x-mu)/sigma,2))
    #q1init = rho0 + (pinit-p0)/(c*c)
    q1init = rho0+1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((DG_model_pipe.x - mu) / sigma, 2))
    q2init = np.zeros((N+1,K))

    solq1,solq2, time = DG_model_pipe.Pipe1D(q1init,q2init, FinalTime=0.1,c=c,rho0=rho0,p0=p0)
#%%
rho = []
u = []
for i in range(len(solq1)):
    rho.append(np.reshape(solq1[i], (N + 1) * K, 'F'))
    u.append(np.reshape(solq2[i]/solq1[i], (N + 1) * K, 'F'))
#%%
xVec = np.reshape(DG_model_pipe.x, (N + 1) * K, 'F')
plt.figure()
plt.plot(xVec,np.reshape(q1init, (N + 1) * K, 'F'))
plt.legend(['rho','u'])
plt.show()

#%%
xVec = np.reshape(DG_model_pipe.x, (N + 1) * K, 'F')
plt.figure()
#plt.plot(xVec,rho[-1])
plt.plot(xVec,u[-1])
plt.legend(['rho','u'])
plt.show()
'''
'''

plt.figure()
plt.loglog(test_vec,error,'.-',markersize=15,linewidth=2)
plt.loglog(test_vec,1/np.power(test_vec,10),'.-',markersize=15,linewidth=2)
plt.show()

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
'''
'''

error = []
test_vec = range(1,15)
for K in test_vec:

    DG_model = DG.DG_1D(xmin=xmin, xmax=xmax, K=K, N=N)
    DG_model.StartUp()
    uinit = np.sin(DG_model.x)

    xVec = np.reshape(DG_model.x, (N + 1) * K, 'F')

    sol, time = DG_model.Advec1D(uinit, FinalTime=1.3)
    #solVec = []
    #for i in range(len(sol)):
    #    solVec.append(np.reshape(sol[i], (N + 1) * K, 'F'))
    exactSol = []
    for i in range(len(time)):
        exactSol.append(np.sin(DG_model.x-2*np.pi*time[i]))
    error.append(1/(DG_model.x.shape[0]*DG_model.x.shape[1]) * np.linalg.norm(np.asarray(sol)-np.asarray(exactSol))**2)

plt.figure()
plt.loglog(test_vec,error,'.-',markersize=15,linewidth=2)
plt.loglog(test_vec,1/np.power(test_vec,10),'.-',markersize=15,linewidth=2)
plt.show()


plt.figure()
plt.plot(xVec,solVec[0])
plt.plot(xVec,solVec[np.floor(len(solVec)/2).astype(int)])
plt.plot(xVec,solVec[-1])
plt.show()
'''

'''
Nv, VX, K, EtoV = DG.MeshGen1D(xmin,xmax,K)

va = np.transpose(EtoV[:,0])
vb = np.transpose(EtoV[:,1])
x = np.ones((Np,1))*VX[va.astype(int)] + 0.5*(np.reshape(r,(len(r),1))+1)*(VX[vb.astype(int)]-VX[va.astype(int)])

V = DG.Vandermonde1D(r,alpha,beta,N)
Dr = DG.Dmatrix1D(r,alpha,beta,N,V)

rx, J = DG.GeometricFactors(x,Dr)

NODETOL = 1e-10
fmask1 = np.where(np.abs(r+1.) < NODETOL)[0]
fmask2 = np.where(np.abs(r-1.) < NODETOL)[0]

Fmask = np.concatenate((fmask1,fmask2),axis=0)
Fx = x[Fmask,:]

nx = DG.Normals1D(Nfp,Nfaces,K)

Fscale = np.divide(1,J[Fmask,:])

EtoE, EtoF = DG.Connect1D(EtoV)

DG.BuildMaps1D(K,Np,Nfp,Nfaces,Fmask,EtoE,EtoF,x,NODETOL)

'''