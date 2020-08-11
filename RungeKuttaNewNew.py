import numpy as np
import pdb
import scipy.linalg as scilin
import matplotlib.pyplot as plt
#import DG_routines as DG
import scipy.optimize as opt
import time as timing
import scipy.sparse.linalg as spla



class ImplicitIntegrator(object):
    def __init__(self,f,u0,t0,te,N,tol=1e-8,order=4):
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

        if self.order == 2:
            p = (2 - np.sqrt(2)) / 2
            b2 = (1 - 2*p)/(4*p)
            self.A = np.array([[0,0,0],[p, p, 0], [1-b2-p, b2, p]])
            self.b = np.array([1-b2-p, b2, p])
            self.c = np.array([0,2*p,1])
        elif self.order == 4:
            self.A = np.array([[1/4, 0, 0, 0, 0],
                               [1 / 2, 1 / 4, 0, 0, 0],
                               [17 / 50, -1 / 25, 1 / 4, 0, 0],
                       [371 / 1360, -137 / 2720, 15 / 544, 1 / 4,
                        0], [25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4]])
            self.b = np.array([25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4])
            self.c = np.array([1 / 4, 3 / 4, 11 / 20, 1 / 2, 1])

            '''
            p = 1/4
            self.A = np.array([ [0,0,0,0,0,0],
                                [p,p,0,0,0,0],
                                [8611/62500,-1743/31250, p, 0,0,0],
                                [5012029/34652500,-654441/2922500,174375/388108,p,0,0],
                                [15267082809/155376265600,-71443401/120774400,730878875/902184768,2285395/8070912,p,0],
                                [82889/524892,0,15625/83664,69875/102672,-2260/8211,p]])
            self.b = np.array([82889/524892,0,15625/83664,69875/102672,-2260/8211,p])
            self.c = np.array([0, 1/2, 83/250,31/50,17/20,1])
            '''

        self.stages = self.b.shape[0]

    def JacobianMultiply(self,U,x):
        epsilon = np.sqrt(np.finfo(float).eps)*np.max(np.maximum(np.abs(U),np.finfo(float).eps))
        dFdU = self.f(self.time,U + epsilon * x) - self.f(self.time,U)

        #epsilon = 1/np.linalg.norm(x)*np.power(np.finfo(float).eps/2,1/3)
        dFdU = self.f(self.time, U + epsilon * x) - self.f(self.time, U - epsilon * x)

        dFdU = dFdU/(2*epsilon)
        return dFdU

    def LHS(self,x,U,a):

        lhs = x-self.deltat*a*self.JacobianMultiply(U,x)
        J = np.array([[0,1],[-1,0]])
        lhs = x-self.deltat*a*np.dot(J,x)

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
        '''
        U = []
        F = []


        g = lambda Uk: Uk - self.Un- self.deltat*self.A[0,0]*self.f(self.time+self.c[0]*self.deltat,Uk)

        Uk = opt.newton_krylov(g, Uk)

        U.append(Uk)
        F.append(self.f(self.time+self.c[0]*self.deltat,U[0]))

        '''
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
                x,_ = spla.gmres(lhs, rhs,tol = self.tol)
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
        Unew = self.ComputeStages()
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




t0, te = 0, 10

def rhs(t,y):
    g = 9.81
    L = 1.

    #RHS = np.array([y[1],-g/L*np.sin(y[0])])

    RHS = np.array([y[1],-y[0]])
    return RHS

#u0 = np.array([np.pi/2.,1])
#system = ImplicitIntegrator(lambda t,y:rhs(t,y),u0,t0,te,10000,tol=1e-8,order=4)
#t_vec,true_sol = system.solve()

y1 = lambda t: 2*np.cos(t)+3*np.sin(t)
y2 = lambda t: -2*np.sin(t) + 3*np.cos(t)

error = []
N_vec = [10,100,1000]
for N in N_vec:

    u0 = np.array([2.,3.])
    system = ImplicitIntegrator(lambda t,y:rhs(t,y),u0,t0,te,N,tol=1e-12,order=2)
    t_vec,solution = system.solve()

    true_sol = np.transpose(np.array([y1(t_vec), y2(t_vec)]))

    error.append(np.mean([np.linalg.norm(solution[i]-true_sol[i]) for i in range(N)]))

plt.figure()
plt.loglog(N_vec,error,'.-',linewidth=1.5,markersize=10)
plt.loglog(N_vec,1/np.power(N_vec,1),'.-',linewidth=1.5,markersize=10)
plt.grid(True)
plt.show()


plt.subplot(2, 1, 1)
plt.plot(t_vec,solution[:,0],'-',linewidth=2.)
plt.plot(t_vec,solution[:,1],'-',linewidth=2.)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(solution[:,0], solution[:,1], 'b-', lw=2)
plt.plot(u0[0],u0[1],'b.',markersize=20)
plt.plot(true_sol[:,0], true_sol[:,1], 'r-', lw=2)
plt.plot(u0[0],u0[1],'r.',markersize=20)
plt.grid(True)
plt.show()
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
