import numpy as np
import pdb
import scipy.linalg as scilin
import matplotlib.pyplot as plt
#import DG_routines as DG
import scipy.optimize as opt
import time
import scipy.sparse.linalg as spla



class Onestepmethod(object):
    def __init__(self,f,y0,t0,te,N,tol):
        self.f =  f
        self.y0 = y0.astype(float)
        self.t0 = t0
        self.interval = [t0, te]
        self.grid = np.linspace(t0, te, N + 2)  # N interior points
        self.h = (te - t0) / (N + 1)
        self.N = N
        self.tol = tol
        self.m = len(y0)
        self.s = len(self.b)

    def step(self):
        t, y = [self.grid[0]], [self.y0]  # initial condition
        tim1 = t[0]
        yi = y[0]

        for ti in self.grid[1:]:
            yi = yi + self.h * self.phi(tim1, yi)
            tim1 = ti
            t.append(tim1)
            y.append(yi)

        return t, np.asarray(y)
    def solve(self):
        self.time, self.solution = self.step()

class RungeKutta_implicit(Onestepmethod):
    def phi(self, t0, y0):

        M = 10 # max number of newton iterations
        stageDer = np.array(self.s*[self.f(t0,y0)]) # initial value: Y'_0
        stageVal = self.phi_solve(t0, y0, stageDer)

        return np.array([np.dot(self.b, stageVal.reshape(self.s, self.m)[:,j]) for j in range(self.m)])

    def phi_solve(self, t0, y0, initVal):

        initVal  = opt.newton(lambda y:self.rhs(y,t0,y0),initVal)

        return  np.asarray(initVal)

    def rhs(self, initVal,t0,y0):
        RHS = np.empty(initVal.shape)
        for i in range(self.s):  # solving the s mxm systems
            RHS[i, :] = - self.F(initVal.flatten(), t0, y0)[i * self.m:(i + 1) * self.m]
        return RHS


    def F(self, stageDer, t0, y0):

        stageDer_new = np.empty((self.s, self.m))  # the i:th stageDer is on the i: th row
        for i in range(self.s):  # iterate over all stageDer
            stageVal = y0 + np.array([self.h * np.dot(self.A[i, :],stageDer.reshape(self.s, self.m)[:, j]) for j in range(self.m)])
            stageDer_new[i, :] = self.f(t0 + self.c[i] * self.h,stageVal)  # the ith stageDer is set on the ith row

        return stageDer - stageDer_new.reshape(-1)


class SDIRK(RungeKutta_implicit):
    def phi_solve(self, t0, y0, initVal):

        initVal  = self.phi_newtonstep(t0, y0, initVal)

        return initVal

    def rhs(self, initVal,t0,y0):
        RHS = np.empty(initVal.shape)
        for i in range(self.s):  # solving the s mxm systems
            RHS[i, :] = - self.F(initVal.flatten(), t0, y0)[i * self.m:(i + 1) * self.m]
        return RHS

    def phi_newtonstep(self, t0, y0, initVal):
        sol=opt.newton(lambda y:self.rhs(y,t0,y0),initVal)

        return np.asarray(sol)

class SDIRK_GMRES(RungeKutta_implicit):

    def phi_solve(self, t0, y0, initVal, M=10):

        for i in range(M):
            initVal, norm_d = self.phi_newtonstep(t0, y0, initVal)
            if norm_d < self.tol:
                #print('Newton converged in {} steps'.format(i))
                break

            elif i == M - 1:
                raise ValueError('The Newton iteration did not converge.')

        return initVal

    def JacobianProduct(self,yk,ydelta):
        epsilon = 1/(np.linalg.norm(ydelta)+np.finfo(float).eps) * (np.finfo(float).eps/2)**(1/3)
        #epsilon = np.sqrt(np.finfo(float).eps)/(np.linalg.norm(ydelta)+np.finfo(float).eps)
        return 0.5*(self.f(self.t0,yk+epsilon*ydelta)-self.f(self.t0,yk-epsilon*ydelta))/epsilon

        #return (self.f(self.t0,yk+epsilon*ydelta)-self.f(self.t0,yk))/epsilon


    def phi_newtonstep(self, t0, y0, initVal):

        x = []
        for i in range(self.s):  # solving the s mxm systems

            rhs = - self.F(initVal.flatten('F'), t0, y0)[i * self.m:(i + 1) * self.m] \
                  + np.sum([self.h * self.A[i, j] *
                    self.JacobianProduct(initVal.flatten('F')[i * self.m:(i + 1) * self.m],x[j]) for j in range(i)], axis=0)

            MatProd = lambda v: np.dot(np.eye(self.m),v) - self.h * self.A[0, 0] * self.JacobianProduct(initVal.flatten('F')[i * self.m:(i + 1) * self.m],v)

            M = spla.LinearOperator((self.m, self.m), matvec = MatProd)

            d, exitCode = spla.gmres(M, rhs,tol = 1e-5,restart=1000,maxiter=10000)

            x.append(d)

        return initVal + x, np.linalg.norm(x)

class SDIRK_tableau2s(SDIRK): # order 3
    p = (3 - np.sqrt(3))/6
    A = np.array([[p, 0], [1 - 2*p, p]])
    b = np.array([1/2, 1/2])
    c = np.array([p, 1 - p])

class SDIRK_tableau5s(SDIRK): #order 4
    A = np.array([[1/4, 0, 0, 0, 0], [1/2, 1/4, 0, 0, 0], [17/50,
    -1/25, 1/4, 0, 0],[371/1360, -137/2720, 15/544, 1/4,
    0],[25/24, -49/48, 125/16, -85/12, 1/4]])
    b = np.array([25/24, -49/48, 125/16, -85/12, 1/4])
    c = np.array([1/4, 3/4, 11/20, 1/2, 1])

class Gauss(RungeKutta_implicit): #order 6
    A=np.array([[5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],[ 5/36 +
    np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],[ 5/36 + np.sqrt(15)/30,
    2/9 + np.sqrt(15)/15, 5/36]])
    b=[5/18,4/9,5/18]
    c=[1/2-np.sqrt(15)/10,1/2,1/2+np.sqrt(15)/10]


'''
N = 10000
t0, te = 0, 5.
tol_newton = 1e-9
tol_sol = 1e-5
timeGrid = np.linspace(t0,te,N+2) #N interior points

def rhs(t,y):
    g = 9.81
    L = 1.

    RHS = np.array([y[1],-g/L*np.sin(y[0])])
    return RHS

tt = time.time()
system = SDIRK_tableau2s(lambda t,y:rhs(t,y),np.array([np.pi/2.,10]),t0,te, N,tol_newton)

system.solve()
t_vec,solution = system.time,system.solution
ttt = time.time()
print('SDIRK2 time: ' + str(ttt-tt))


plt.figure()
plt.plot(solution[0,:],solution[1,:],'-',linewidth=2.)
#plt.plot(time,solution[1,:],'.-', markersize=5,linewidth=2.)



N = 10000
timeGrid = np.linspace(t0,te,N+2) #N interior points

tt = time.time()

system = Gauss(lambda t,y:rhs(t,y),np.array([np.pi/2.,10]),t0,te, N,tol_newton)

system.solve()
t_vec,solution = system.time,system.solution
ttt = time.time()
print('Gauss time: ' + str(ttt-tt))

plt.plot(solution[0,:],solution[1,:],'--',linewidth=2.)

plt.show()
'''
"""
class ImplicitRungeKutta():
    def __init__(self, f, y0, t0, te, h, tol):
        self.p = (3 - np.sqrt(3)) / 6
        self.A = np.array([[self.p, 0], [1 - 2 * self.p, self.p]])
        self.b = np.array([1 / 2, 1 / 2])
        self.c = np.array([self.p, 1 - self.p])

        self.f = f
        self.y0 = y0.astype(float)
        self.t0 = t0
        self.interval = [t0, te]
        self.h = h
        self.tol = tol
        self.m = len(y0)
        self.s = len(self.b)


    def Y_derivative(self,t,Y):
        Y_deriv = []

        for i in range(self.s):
            Y_deriv.append(self.f(t+self.c[i]*self.h,Y[i]))

        return Y_deriv

    def Y_stage(self,t,y,y_old):

        Y_stages = []
        Y_deriv = self.Y_derivative(t,y)
        pdb.set_trace()

        for i in range(self.s):
            stage = 0
            for j in range(self.s):
                stage += self.h*self.A[i,j]*Y_deriv[j]


            Y_stages.append(stage + y_old)

        return np.asarray(Y_stages).flatten()

    def step(self,t,y_old):
        pdb.set_trace()

        G = lambda y: y-self.Y_stage(t,y,y_old)
        y_newton_sol = opt.newton_krylov(G,np.concatenate((y_old,y_old)))

        y_new = 0
        for i in range(self.s):
            y_new = self.h*self.b[i]*y_newton_sol[i]

        y_new += y_old
        return y_new

    def solve(self):

        t_vec = [self.t0]
        solution = [self.y0]

        idx = 0
        y_old = solution[idx]
        t = t_vec[idx]
        while t < te:
            y_new = self.step(t, y_old)

            solution.append(y_new)
            t_vec.append(t+self.h)
            idx += 1

            y_old = solution[idx]
            t = t_vec[idx]

        return t_vec, solution


t0, te = 0, 5.
tol_newton = 1e-9
tol_sol = 1e-5


def rhs(t, y):
    g = 9.81
    L = 1.

    RHS = np.array([y[1], -g / L * np.sin(y[0])])
    return RHS



system = ImplicitRungeKutta(lambda t, y: rhs(t, y), np.array([np.pi / 2., 10]), t0, te, h=0.001,tol=1e-6)

t_vec, solution = system.solve()
solution = np.transpose(np.asarray(solution))


plt.figure()
plt.plot(solution[0, :], solution[1, :], '-', linewidth=2.)
plt.show()
"""
