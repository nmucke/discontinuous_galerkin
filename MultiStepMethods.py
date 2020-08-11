import numpy as np
import pdb
import scipy.linalg as scilin
import matplotlib.pyplot as plt
# import DG_routines as DG
import scipy.optimize as opt
import time as timing
import scipy.sparse.linalg as spla


class BDF2(object):
    def __init__(self, f, u0, t0, te, N, tol=1e-8, order=2):
        self.f = f
        self.u0 = u0.astype(float)
        self.t0 = t0
        self.te = te
        self.deltat = (te - t0) / (N + 1)
        self.N = N
        self.tol = tol
        self.m = len(u0)
        self.order = order
        self.time = 0
        self.MaxNewtonIter = 50

        self.alpha = np.array([3,-4,1])/2

    def InitialStep(self):

        #Uinit = self.sol[-1] + self.deltat * self.f(self.time, self.sol[-1])

        G = lambda Unew: Unew - self.sol[-1] - self.deltat*self.f(self.time+self.deltat,Unew)
        Unew = opt.newton(G, self.sol[-1])
        return Unew

    def UpdateState(self):

        #Uinit = self.sol[-1] + self.deltat * self.f(self.time, self.sol[-1])

        G = lambda Unew: (self.alpha[0]*Unew + self.alpha[1]*self.sol[-1] + self.alpha[2]*self.sol[-2])/self.deltat - self.f(self.time+self.deltat,Unew)
        Unew = opt.newton(G, self.sol[-1])

        return Unew

    def solve(self):

        self.sol = [self.u0]
        tVec = [self.t0]
        self.time = self.t0

        for i in range(2):
            self.Un = self.InitialStep()
            self.sol.append(self.Un)
            self.time += self.deltat
            tVec.append(self.time)

        for i in range(self.N-1):
            self.Un = self.UpdateState()
            self.time += self.deltat
            tVec.append(self.time)
            self.sol.append(self.Un)
            if i % 100 == 0:
                print(self.time)

        return tVec, np.asarray(self.sol)


'''
t0, te = 0, 50.


def rhs(t, y):
    g = 9.81
    L = 1.

    # RHS = np.array([y[1],-g/L*np.sin(y[0])])

    RHS = np.array([y[1], -y[0]])
    return RHS


# u0 = np.array([np.pi/2.,1])
# system = ImplicitIntegrator(lambda t,y:rhs(t,y),u0,t0,te,10000,tol=1e-8,order=4)
# t_vec,true_sol = system.solve()

y1 = lambda t: 2 * np.cos(t) + 3 * np.sin(t)
y2 = lambda t: -2 * np.sin(t) + 3 * np.cos(t)

error = []
N_vec = [1e2,1e3,1e4]
for N in N_vec:
    N = int(N)
    u0 = np.array([2., 3.])
    system = BDF2(lambda t, y: rhs(t, y), u0, t0, te, N, tol=1e-12, order=2)
    t_vec, solution = system.solve()

    true_sol = np.transpose(np.array([y1(t_vec), y2(t_vec)]))

    error.append(np.mean([np.linalg.norm(solution[i] - true_sol[i]) for i in range(N)]))


plt.figure()
plt.loglog(N_vec, error, '.-', linewidth=1.5, markersize=10)
plt.loglog(N_vec, 1 / np.power(N_vec, 2), '.-', linewidth=1.5, markersize=10)
plt.grid(True)
plt.show()

plt.subplot(2, 1, 1)
plt.plot(t_vec, solution[:, 0], '-', linewidth=2.)
plt.plot(t_vec, solution[:, 1], '-', linewidth=2.)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(solution[:, 0], solution[:, 1], 'b-', lw=2)
plt.plot(u0[0], u0[1], 'b.', markersize=20)
plt.plot(true_sol[:, 0], true_sol[:, 1], 'r--', lw=2)
plt.plot(u0[0], u0[1], 'r.', markersize=20)
plt.grid(True)
plt.show()
'''







