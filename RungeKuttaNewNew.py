import numpy as np
import pdb
import scipy.linalg as scilin
import matplotlib.pyplot as plt
#import DG_routines as DG
import scipy.optimize as opt
import time as timing
import scipy.sparse.linalg as spla



class ImplicitIntegrator(object):
    def __init__(self,f,u0,t0,te,N,tol=1e-8,order=3):
        self.f = f
        self.u0 = u0.astype(float)
        self.t0 = t0
        self.interval = [t0, te]
        self.timeGrid = np.linspace(t0, te, N)  # N interior points
        self.deltat = (te - t0) / (N + 1)
        self.N = N
        self.tol = tol
        self.m = len(u0)
        self.order = order
        self.time = 0
        self.Un = u0
        self.MaxNewtonIter = 50

        if order == 2:
            p = (2 - np.sqrt(2)) / 2
            b2 = (1 - 2*p)/(4*p)
            self.A = np.array([[0,0,0],[p, p, 0], [1-b2-p, b2, p]])
            self.b = np.array([1-b2-p, b2, p])
            self.c = np.array([0,2*p,1])
        elif order == 3:
            p = (3 - np.sqrt(3)) / 6
            c3 = 1
            b2 = (-2+3*c3)/(12*(c3-2*p)*p)
            b3 = (1+3*p)/(3*c3*(c3-2*p))
            a32 = c3*(c3-2*p)/(4*p)
            self.A = np.array([[0, 0, 0], [p, p, 0], [c3-a32-p, a32, p]])
            self.b = np.array([1 - b2 - b3, b2, b3])
            self.c = np.array([0, 2 * p, c3])


        self.stages = self.b.shape[0]

    def JacobianMultiply(self,U,x):
        epsilon = np.sqrt(np.finfo(float).eps)*np.max(np.maximum(np.abs(U),np.finfo(float).eps))
        dFdU = self.f(self.time,U + epsilon * x) - self.f(self.time,U)

        #epsilon = 1/np.linalg.norm(x)*np.power(np.finfo(float).eps/2,1/3)
        #dFdU = self.f(self.time, U + epsilon * x) - self.f(self.time, U - epsilon * x)

        dFdU = dFdU/(epsilon)
        return dFdU

    def LHS(self,x,U,a):

        lhs = x-self.deltat*a*self.JacobianMultiply(U,x)

        return lhs

    def RHS(self,Uk,X,s):

        rhs = -Uk + self.Un + X + self.deltat*self.A[s, s]*self.f(self.time+self.c[s]*self.deltat,Uk)

        return rhs

    def G(self,Uk, U, n):
        X = 0
        if n > 0:
            for j in range(n):
                X += self.A[n, j] * self.f(self.time + self.c[j] * self.deltat, U[j])
        g = Uk - self.Un - self.deltat * X - self.deltat * self.A[n, n] * self.f(
            self.time + self.c[n] * self.deltat, Uk)
        return g

    def ComputeStages(self):

        U = []
        U.append(self.Un)

        F = []
        F.append(self.f(self.time+self.c[0]*self.deltat,U[0]))

        Uk = self.Un

        for i in range(1,self.stages):
            k = 0
            x = np.ones(2)
            '''
            g = lambda Uk: self.G(Uk, U, i)
    
            Uk = opt.newton_krylov(g, Uk)
    
            U.append(Uk)
            '''
            X = 0
            for j in range(i):
                X += self.A[i, j] * F[j]
            X *= self.deltat
    
            while k < self.MaxNewtonIter and np.linalg.norm(x) > self.tol:
    
                lhs = lambda x: self.LHS(x,Uk,self.A[i,i])
                lhs = spla.LinearOperator((self.m, self.m), matvec=lhs)

                rhs = self.RHS(Uk,X,i)
                x,_ = spla.gmres(lhs, rhs,tol = 1e-5)
                Uk = Uk + x
                k = k+1

            F.append(self.f(self.time + self.c[i] * self.deltat, Uk))

        return F

    def UpdateState(self):

        F = self.ComputeStages()

        F_update = 0
        for i in range(self.stages):
            F_update += self.b[i]*F[i]

        Unew = self.Un + self.deltat*F_update
        '''
        Uk = self.Un
        k = 0
        x = np.ones(2)
        while k < self.MaxNewtonIter and np.linalg.norm(x) > self.tol:
            g = 9.81
            L = 1.
            J = np.array([[0, 1], [-g / L * np.cos(Uk[0]), 0]])
            lhs = np.eye(2) - self.deltat * J
            rhs = -Uk + self.Un + self.deltat * self.f(self.time + self.deltat, Uk)
            x = np.linalg.solve(lhs, rhs)
            Uk = Uk + x
            k += 1
            print(k)
        '''
        return Unew

    def solve(self):

        sol = [self.Un]
        tVec = [self.t0]

        for self.time in self.timeGrid:
            self.Un = self.UpdateState()
            tVec.append(self.time)
            sol.append(self.Un)

        return tVec, np.asarray(sol)



"""
N = 1000
t0, te = 0, 5
tol_newton = 1e-9
tol_sol = 1e-2
timeGrid = np.linspace(t0,te,N+2) #N interior points

def rhs(t,y):
    g = 9.81
    L = 1.

    RHS = np.array([y[1],-g/L*np.sin(y[0])])
    return RHS

u0 = np.array([np.pi/2.,1])
system = ImplicitIntegrator(lambda t,y:rhs(t,y),u0,t0,te,N,tol=1e-8,order=2)

t_vec,solution = system.solve()




plt.subplot(2, 1, 1)
plt.plot(t_vec,solution[:,0],'-',linewidth=2.)
plt.plot(t_vec,solution[:,1],'-',linewidth=2.)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(solution[:,0], solution[:,1], '-', lw=2)
plt.plot(u0[0],u0[1],'.',markersize=20)
plt.grid(True)
plt.show()

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
