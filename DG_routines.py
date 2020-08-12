import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.special import gamma
import scipy.special as sci
import scipy.sparse as sps
import scipy.integrate as integrate
import time as timing
import RungeKuttaNewNew as RK
import MultiStepMethods as multistep
import scipy.optimize as opt


def JacobiP(x,alpha,beta,N):
    xp = x

    PL = np.zeros((N+1, len(xp)))


    gamma0 = 2**(alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)
    PL[0,:] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0

    PL[1,:] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
    if N == 1:
        return PL[-1:,:]


    aold = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3));
    for i in range(1,N):
        h1 = 2 * i + alpha + beta
        anew = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        bnew = -(alpha**2-beta**2)/h1/(h1+2)
        PL[i+1,:] = 1/anew*(-aold*PL[i-1,:] + np.multiply((xp-bnew),PL[i,:]))
        aold = anew

    return PL[-1:,:]

def JacobiGQ(alpha,beta,N):
    x,w = sci.roots_jacobi(N,alpha,beta)
    return x,w

def JacobiGL(alpha,beta,N):
    x = np.zeros((N+1,1))

    if N==1:
        x[0]=-1
        x[1]=1
        x = x[:,0]

        return x
    x_int,w = JacobiGQ(alpha+1,beta+1,N-1)
    x = np.append(-1,np.append(x_int,1))
    return x

def Vandermonde1D(x,alpha,beta,N):
    V1D = np.zeros((len(x),N+1))

    for i in range(0,N+1):

        V1D[:,i] = JacobiP(x,alpha,beta,i)
    return V1D

def GradJacobiP(r,alpha,beta,N):
    dP = np.zeros((len(r),1))
    if N == 0:
        return dP
    else:
        dP[:,0] = np.sqrt(N*(N+alpha+beta+1))*JacobiP(r,alpha+1,beta+1,N-1)
    return dP


def GradVandermonde1D(r,alpha,beta,N):
    DVr = np.zeros((len(r),N+1))


    for i in range(0,N+1):

        DVr[:,i:(i+1)] = GradJacobiP(r,alpha,beta,i)
    return DVr

def Dmatrix1D(r,alpha,beta,N,V):
    Vr = GradVandermonde1D(r,alpha,beta,N)

    Dr = np.transpose(np.linalg.solve(np.transpose(V),np.transpose(Vr)))
    return Dr


def lift1D(Np,Nfaces,Nfp,V):
    Emat = np.zeros((Np,Nfaces*Nfp))
    Emat[0,0] = 1
    Emat[Np-1,1] = 1
    LIFT = np.dot(V,np.dot(np.transpose(V),Emat))
    return LIFT


def MeshGen1D(xmin,xmax,K):

    Nv = K+1

    VX = np.arange(1.,Nv+1.)

    for i in range(0,Nv):
        VX[i] = (xmax-xmin)*i/(Nv-1) + xmin

    EtoV = np.zeros((K,2))
    for k in range(0,K):
        EtoV[k,0] = k
        EtoV[k,1] = k+1

    return Nv, VX, K, EtoV

def GeometricFactors(x,Dr):
    xr = np.dot(Dr,x)
    J = xr
    rx = np.divide(1,J)

    return rx,J

def diracDelta(x):
    f = np.zeros(x.shape)
    f[np.argwhere((x<0.2e-1) & (x>-0.2e-1))] = 1
    return f

class DG_1D:
    def __init__(self, xmin=0,xmax=1,K=10,N=5):
        self.xmin = xmin
        self.xmax = xmax
        self.K = K
        self.N = N
        self.Np = N + 1
        self.NODETOL = 1e-10
        self.Nfp = 1
        self.Nfaces = 2

        self.rk4a = np.array([ 0.0,-567301805773.0/1357537059087.0,-2404267990393.0/2016746695238.0,-3550918686646.0/2091501179385.0,-1275806237668.0/842570457699.0])
        self.rk4b = np.array([ 1432997174477.0/9575080441755.0,5161836677717.0/13612068292357.0,1720146321549.0/2090206949498.0,3134564353537.0/4481467310338.0,2277821191437.0/14882151754819.0])
        self.rk4c = np.array([ 0.0,1432997174477.0/9575080441755.0,2526269341429.0/6820363962896.0,2006345519317.0/3224310063776.0,2802321613138.0/2924317926251.0])

        self.r = JacobiGL(0,0,self.N)
        self.V = Vandermonde1D(self.r,0,0,self.N)
        self.invV = np.linalg.inv(self.V)
        self.Dr = Dmatrix1D(self.r,0,0,self.N,self.V)
        self.M = np.transpose(np.linalg.solve(np.transpose(self.invV),self.invV))
        self.invM = np.linalg.inv(self.M)

        self.LIFT = lift1D(self.Np,self.Nfaces,self.Nfp,self.V)
        self.Nv, self.VX, self.K, self.EtoV = MeshGen1D(self.xmin, self.xmax, self.K)

        self.va = np.transpose(self.EtoV[:, 0])
        self.vb = np.transpose(self.EtoV[:, 1])
        self.x = np.ones((self.Np, 1)) * self.VX[self.va.astype(int)] + 0.5 * (np.reshape(self.r, (len(self.r), 1)) + 1) \
                * (self.VX[self.vb.astype(int)] - self.VX[self.va.astype(int)])

        fmask1 = np.where(np.abs(self.r + 1.) < self.NODETOL)[0]
        fmask2 = np.where(np.abs(self.r - 1.) < self.NODETOL)[0]

        self.Fmask = np.concatenate((fmask1, fmask2), axis=0)
        self.Fx = self.x[self.Fmask, :]


    def Normals1D(self):
            nx = np.zeros((self.Nfp * self.Nfaces, self.K))
            nx[0, :] = -1.0
            nx[1, :] = 1.0
            return nx

    def BuildMaps1D(self):

            nodeids = np.reshape(np.arange(0, self.K * self.Np), (self.Np, self.K), 'F')
            vmapM = np.zeros((self.Nfp, self.Nfaces, self.K))
            vmapP = np.zeros((self.Nfp, self.Nfaces, self.K))

            for k1 in range(0, self.K):
                for f1 in range(0, self.Nfaces):
                    vmapM[:, f1, k1] = nodeids[self.Fmask[f1], k1]

            for k1 in range(0, self.K):
                for f1 in range(0, self.Nfaces):
                    k2 = self.EtoE[k1, f1].astype(int)
                    f2 = self.EtoF[k1, f1].astype(int)

                    vidM = vmapM[:, f1, k1].astype(int)
                    vidP = vmapM[:, f2, k2].astype(int)

                    x1 = self.x[np.unravel_index(vidM, self.x.shape, 'F')]
                    x2 = self.x[np.unravel_index(vidP, self.x.shape, 'F')]

                    D = (x1 - x2) ** 2
                    if D < self.NODETOL:
                        vmapP[:, f1, k1] = vidP

            vmapP = vmapP.flatten('F')
            vmapM = vmapM.flatten('F')

            mapB = np.argwhere(vmapP == vmapM)
            vmapB = vmapM[mapB]

            mapI = 0
            mapO = self.K * self.Nfaces-1
            vmapI = 0
            vmapO = self.K * self.Np-1

            return vmapM.astype(int), vmapP.astype(int), vmapB.astype(int), mapB.astype(int), mapI,mapO,vmapI,vmapO

    def Connect1D(self):
        TotalFaces = self.Nfaces * self.K
        Nv = self.K + 1

        vn = [0, 1]

        SpFToV = sps.lil_matrix((TotalFaces, Nv))

        sk = 0
        for k in range(0, self.K):
            for face in range(0, self.Nfaces):
                SpFToV[sk, self.EtoV[k, vn[face]]] = 1.
                sk = sk + 1

        SpFToF = np.dot(SpFToV, np.transpose(SpFToV)) - sps.eye(TotalFaces)
        faces = np.transpose(np.nonzero(SpFToF))
        faces[:, [0, 1]] = faces[:, [1, 0]] + 1

        element1 = np.floor((faces[:, 0] - 1) / self.Nfaces)
        face1 = np.mod((faces[:, 0] - 1), self.Nfaces)
        element2 = np.floor((faces[:, 1] - 1) / self.Nfaces)
        face2 = np.mod((faces[:, 1] - 1), self.Nfaces)

        ind = np.ravel_multi_index(np.array([face1.astype(int), element1.astype(int)]), (self.Nfaces, self.K))
        EtoE = np.reshape(np.arange(0, self.K), (self.K, 1)) * np.ones((1, self.Nfaces))
        EtoE[np.unravel_index(ind, EtoE.shape, 'F')] = element2
        EtoF = np.ones((self.K, 1)) * np.reshape(np.arange(0, self.Nfaces), (1, self.Nfaces))
        EtoF[np.unravel_index(ind, EtoE.shape, 'F')] = face2
        return EtoE, EtoF

    def GeometricFactors(self):
            xr = np.dot(self.Dr, self.x)
            J = xr
            rx = np.divide(1, J)

            return rx, J

    def StartUp(self):

        self.EtoE, self.EtoF = DG_1D.Connect1D(self)

        self.vmapM, self.vmapP, self.vmapB, self.mapB,self.mapI,self.mapO,self.vmapI,self.vmapO = DG_1D.BuildMaps1D(self)

        self.nx = DG_1D.Normals1D(self)

        self.rx, self.J = DG_1D.GeometricFactors(self)

        self.alpha = 1

        self.Fscale = 1./(self.J[self.Fmask,:])

        self.a = 2*np.pi

    def minmod(self,v):

        m = v.shape[0]
        mfunc = np.zeros((v.shape[1],))
        s = np.sum(np.sign(v),0)/m

        ids = np.argwhere(np.abs(s)==1)

        if ids.shape[0]!=0:
            mfunc[ids] = s[ids] * np.min(np.abs(v[:,ids]),0)

        return mfunc

    def SlopeLimitLin(self,ul,xl,vm1,v0,vp1):

        ulimit = ul
        h = xl[self.Np-1,:]-xl[0,:]
        x0 = np.ones((self.Np,1))*(xl[0,:]+h/2)

        hN = np.ones((self.Np,1))*h

        ux = (2/hN) * np.dot(self.Dr,ul)

        ulimit = np.ones((self.Np,1))*v0 + (xl-x0)*(DG_1D.minmod(self,np.stack((ux[0,:],np.divide((vp1-v0),h),np.divide((v0-vm1),h)),axis=0)))

        return ulimit

    def SlopeLimitN(self,u):

        uh = np.dot(self.invV,u)
        uh[1:self.Np,:] = 0
        uavg = np.dot(self.V,uh)
        v = uavg[0:1,:]

        ulimit = u
        eps0 = 1e-8

        ue1 = u[0,:]
        ue2 = u[-1:,:]


        vk = v
        vkm1 = np.concatenate((v[0,0:1],v[0,0:self.K-1]),axis=0)
        vkp1 = np.concatenate((v[0,1:self.K],v[0,(self.K-1):(self.K)]))

        ve1 = vk - DG_1D.minmod(self,np.concatenate((vk-ue1,vk-vkm1,vkp1-vk)))
        ve2 = vk + DG_1D.minmod(self,np.concatenate((ue2-vk,vk-vkm1,vkp1-vk)))

        ids = np.argwhere((np.abs(ve1-ue1)>eps0) | (np.abs(ve2-ue2)>eps0))[:,1]
        if ids.shape[0] != 0:

            uhl = np.dot(self.invV,u[:,ids])
            uhl[2:(self.Np+1),:] = 0
            ul = np.dot(self.V,uhl)

            ulimit[:,ids] = DG_1D.SlopeLimitLin(self, ul,self.x[:,ids],vkm1[ids],vk[0,ids],vkp1[ids])

        return ulimit

class Advection1D(DG_1D):
    def __init__(self, xmin=0, xmax=1, K=10, N=5):
        DG_1D.__init__(self, xmin=xmin, xmax=xmax, K=K, N=N)


    def AdvecRHS1D(self,time, u):

        u = u.flatten('F')

        du = np.zeros((self.Nfp * self.Nfaces, self.K))
        du = du.flatten(('F'))
        du = np.multiply(u[self.vmapM] - u[self.vmapP],
                         self.a * self.nx.flatten('F') - 1 - self.alpha * np.abs(self.a * self.nx.flatten('F'))) / 2

        uin = -np.sin(self.a * time)
        du[self.mapI] = np.multiply(u[self.vmapI] - uin, self.a * self.nx[self.mapI, 0] - (1 - self.alpha) * np.abs(
            self.a * self.nx[self.mapI, 0])) / 2
        du[self.mapO] = 0.

        u = np.reshape(u, (self.Np, self.K), 'F')
        du = np.reshape(du, (self.Nfp * self.Nfaces, self.K), 'F')

        rhsu = -self.a * np.multiply(self.rx, np.dot(self.Dr, u)) + np.dot(self.LIFT, np.multiply(self.Fscale, du))

        return rhsu

    def AdvecRHS1DImplicit(self,time, u):

        u = u.flatten('F')

        du = np.zeros((self.Nfp * self.Nfaces, self.K))
        du = du.flatten(('F'))
        du = np.multiply(u[self.vmapM] - u[self.vmapP],
                         self.a * self.nx.flatten('F') - 1 - self.alpha * np.abs(self.a * self.nx.flatten('F'))) / 2

        uin = -np.sin(self.a * time)
        du[self.mapI] = np.multiply(u[self.vmapI] - uin, self.a * self.nx[self.mapI, 0] - (1 - self.alpha) * np.abs(
            self.a * self.nx[self.mapI, 0])) / 2
        du[self.mapO] = 0.

        u = np.reshape(u, (self.Np, self.K), 'F')
        du = np.reshape(du, (self.Nfp * self.Nfaces, self.K), 'F')

        rhsu = -self.a * np.multiply(self.rx, np.dot(self.Dr, u)) + np.dot(self.LIFT, np.multiply(self.Fscale, du))

        rhsu = rhsu.flatten('F')
        return rhsu

    def ExplicitIntegration(self, u, FinalTime):

        resu = np.zeros((self.Np,self.K))
        time = 0

        mindeltax = np.min(np.abs(self.x[0,:]-self.x[1,:]))
        CFL = 0.75
        dt = CFL/(2*np.pi) * mindeltax
        dt = 0.5*dt
        Nsteps = np.ceil(FinalTime/dt).astype(int)
        dt = FinalTime/Nsteps
        sol = [u]
        tVec = [time]
        for tstep in range(0,Nsteps):
            for INTRK in range(0,5):
                timelocal = time + self.rk4c[INTRK] * dt
                rhsu = self.AdvecRHS1D(timelocal,u)
                resu = self.rk4a[INTRK]*resu + dt*rhsu
                u = u + self.rk4b[INTRK]*resu
            sol.append(u)
            time = time+dt
            tVec.append(time)

        return sol,tVec

    def ImplicitIntegration(self, q, FinalTime, stepsize):

        #N_steps = int(FinalTime / stepsize)


        initCondition = q.flatten('F')

        #system = multistep.BDF2(self.AdvecRHS1DImplicit,initCondition,t0=0,te=FinalTime,N=N_steps,tol=1e-8,order=self.order)
        system = RK.ImplicitIntegrator(self.AdvecRHS1DImplicit,initCondition,t0=0,te=FinalTime,stepsize=stepsize,tol=1e-6,order=self.order)

        t_vec,solution = system.solve()

        return solution, t_vec

    def solve(self, q, FinalTime,implicit=False,stepsize=1e-5,order=3):

        t0 = timing.time()
        if implicit:
            self.order = order
            sol,  tVec = self.ImplicitIntegration(q, FinalTime, stepsize)
        else:
            sol,  tVec = self.ExplicitIntegration(q, FinalTime)
        t1 = timing.time()

        print('Simulation finished \nTime: {:.2f} seconds'.format(t1 - t0))

        return sol, tVec


class Maxwell1D(DG_1D):
    def __init__(self, xmin=0, xmax=1, K=10, N=5,gamma=1.4):
        DG_1D.__init__(self, xmin=xmin, xmax=xmax, K=K, N=N)
        self.gamma = 1.4
    def MaxwellRHS1D(self,time,E,H,eps,mu):

        E = E.flatten('F'); H = H.flatten('F')

        Zimp = np.sqrt(np.divide(mu,eps)); Zimp = Zimp.flatten('F')

        dE = np.zeros((self.Nfp*self.Nfaces,self.K)).flatten('F')
        dH = np.zeros((self.Nfp*self.Nfaces,self.K)).flatten('F')
        dE = E[self.vmapM] - E[self.vmapP]
        dH = H[self.vmapM] - H[self.vmapP]

        Zimpm = np.zeros((self.Nfp*self.Nfaces,self.K)).flatten('F')
        Zimpp = np.zeros((self.Nfp*self.Nfaces,self.K)).flatten('F')
        Yimpm = np.zeros((self.Nfp*self.Nfaces,self.K)).flatten('F')
        Yimpp = np.zeros((self.Nfp*self.Nfaces,self.K)).flatten('F')

        Zimpm = Zimp[self.vmapM]
        Zimpp = Zimp[self.vmapP]
        Yimpm = 1/Zimpm
        Yimpp = 1/Zimpp

        Ebc = -E[self.vmapB]
        Hbc = H[self.vmapB]
        dE[self.mapB] = E[self.vmapB] - Ebc
        dH[self.mapB] = H[self.vmapB] - Hbc


        E = np.reshape(E, (self.Np, self.K), 'F')
        H = np.reshape(H, (self.Np, self.K), 'F')
        dE = np.reshape(dE,((self.Nfp*self.Nfaces,self.K)),'F')
        dH = np.reshape(dH,((self.Nfp*self.Nfaces,self.K)),'F')

        Zimpm = np.reshape(Zimpm,((self.Nfp*self.Nfaces,self.K)),'F')
        Zimpp = np.reshape(Zimpp,((self.Nfp*self.Nfaces,self.K)),'F')
        Yimpm = np.reshape(Yimpm,((self.Nfp*self.Nfaces,self.K)),'F')
        Yimpp = np.reshape(Yimpp,((self.Nfp*self.Nfaces,self.K)),'F')

        fluxE = 1/(Zimpm+Zimpp)*(self.nx*Zimpp*dH-dE)
        fluxH = 1/(Yimpm+Yimpp)*(self.nx*Yimpp*dE-dH)

        #fluxE = np.reshape(fluxE, (self.Np, self.K), 'F'); fluxH = np.reshape(fluxH, (self.Np, self.K), 'F')


        rhsE = (-self.rx*np.dot(self.Dr,H) + np.dot(self.LIFT,self.Fscale*fluxE))/eps
        rhsH = (-self.rx*np.dot(self.Dr,E) + np.dot(self.LIFT,self.Fscale*fluxH))/mu

        return rhsE,rhsH

    def Maxwell1D(self, E,H,eps,mu, FinalTime):

        resE = np.zeros((self.Np,self.K))
        resH = np.zeros((self.Np,self.K))

        time = 0

        mindeltax = np.min(np.abs(self.x[0,:]-self.x[1,:]))
        CFL = 1.
        dt = CFL*mindeltax
        Nsteps = np.ceil(FinalTime/dt).astype(int)
        dt = FinalTime/Nsteps

        solE = [E]
        solH = [H]
        tVec = [time]

        aa = 36.
        s = 10
        sigma = np.exp(-aa*((np.arange(1,self.Np+1)-1)/self.N)**s)
        Lambda = np.diag(sigma)

        V = Vandermonde1D(self.r,0,0,self.N)
        VLambda = np.dot(V,Lambda)
        for tstep in range(0,Nsteps):
            for INTRK in range(0,5):

                rhsE, rhsH = self.MaxwellRHS1D(self,time,E,H,eps,mu)

                resE = self.rk4a[INTRK]*resE + dt*rhsE
                resH = self.rk4a[INTRK]*resH + dt*rhsH

                resE = np.dot(VLambda,np.linalg.solve(V,resE))
                resH = np.dot(VLambda,np.linalg.solve(V,resH))

                E = E+self.rk4b[INTRK]*resE
                H = H+self.rk4b[INTRK]*resH
            solE.append(E)
            solH.append(H)
            time = time+dt
            tVec.append(time)

        return solE,solH,tVec

class BDF2(DG_1D):
    def __init__(self, f, u0, t0, te, Ntime,xmin=0, xmax=1, K=5, N=5):

        DG_1D.__init__(self, xmin=xmin, xmax=xmax, K=K, N=N)

        self.f = f
        self.u0 = u0.astype(float)
        self.t0 = t0
        self.te = te
        self.deltat = (te - t0) / (Ntime + 1)
        self.Ntime = Ntime
        self.m = len(u0)
        self.time = 0
        self.MaxNewtonIter = 50

        self.alpha = np.array([3, -4, 1]) / 2

    def InitialStep(self):

        # Uinit = self.sol[-1] + self.deltat * self.f(self.time, self.sol[-1])

        G = lambda Unew: Unew - self.sol[-1] - self.deltat * self.f(self.time + self.deltat, Unew)
        '''
        def G(Unew):
            g = Unew - self.sol[-1] - self.deltat * self.f(self.time + self.deltat, Unew)

            return g
        '''
        Unew = opt.newton(G, self.sol[-1])

        Unew[0:int(len(Unew) / 3)] = self.SlopeLimitN(
            np.reshape(Unew[0:int(len(Unew) / 3)], (self.Np, self.K), 'F')).flatten('F')
        Unew[int(len(Unew) / 3):int(2 * len(Unew) / 3)] = self.SlopeLimitN(
            np.reshape(Unew[int(len(Unew) / 3):int(2 * len(Unew) / 3)], (self.Np, self.K), 'F')).flatten('F')
        Unew[-int(len(Unew) / 3):] = self.SlopeLimitN(
            np.reshape(Unew[-int(len(Unew) / 3):], (self.Np, self.K), 'F')).flatten('F')



        return Unew

    def UpdateState(self):

        # Uinit = self.sol[-1] + self.deltat * self.f(self.time, self.sol[-1])

        G = lambda Unew: (self.alpha[0] * Unew + self.alpha[1] * self.sol[-1] + self.alpha[2] * self.sol[
            -2]) / self.deltat - self.f(self.time + self.deltat, Unew)
        Unew = opt.newton(G, self.sol[-1])

        Unew[0:int(len(Unew) / 3)] = self.SlopeLimitN(np.reshape(Unew[0:int(len(Unew) / 3)], (self.Np, self.K), 'F')).flatten('F')
        Unew[int(len(Unew) / 3):int(2 * len(Unew) / 3)] = self.SlopeLimitN(np.reshape(Unew[int(len(Unew) / 3):int(2 * len(Unew) / 3)], (self.Np, self.K), 'F')).flatten('F')
        Unew[-int(len(Unew) / 3):] = self.SlopeLimitN(np.reshape(Unew[-int(len(Unew) / 3):], (self.Np, self.K), 'F')).flatten('F')

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

        for i in range(self.Ntime - 1):
            self.Un = self.UpdateState()
            self.time += self.deltat
            tVec.append(self.time)
            self.sol.append(self.Un)

            if i % 1000 == 0:
                print(self.time)

        return tVec, np.asarray(self.sol)

class Euler1D(DG_1D):
    def __init__(self, xmin=0, xmax=1, K=10, N=5):
        DG_1D.__init__(self, xmin=xmin, xmax=xmax, K=K, N=N)



    def EulerRHS1D(self,time,q1,q2,q3):

        q1 = q1.flatten('F')
        q2 = q2.flatten('F')
        q3 = q3.flatten('F')

        gamma = 1.4

        pressure = (gamma-1.)*(q3-0.5*np.divide(q2*q2,q1))
        cvel = np.sqrt(gamma*np.divide(pressure,q1))
        lm = np.abs(np.divide(q2,q1)) + cvel

        q1Flux = q2
        q2Flux = np.divide(np.power(q2,2),q1) + pressure
        q3Flux = np.divide((q3+pressure)*q2,q1)

        dq1 = q1[self.vmapM] - q1[self.vmapP]
        dq2 = q2[self.vmapM] - q2[self.vmapP]
        dq3 = q3[self.vmapM] - q3[self.vmapP]

        dq1Flux = q1Flux[self.vmapM] - q1Flux[self.vmapP]
        dq2Flux = q2Flux[self.vmapM] - q2Flux[self.vmapP]
        dq3Flux = q3Flux[self.vmapM] - q3Flux[self.vmapP]

        LFc = np.maximum((lm[self.vmapM]),(lm[self.vmapP]))

        dq1Flux = self.nx.flatten('F')*dq1Flux/2. - LFc/2. * dq1
        dq2Flux = self.nx.flatten('F')*dq2Flux/2. - LFc/2. * dq2
        dq3Flux = self.nx.flatten('F')*dq3Flux/2. - LFc/2. * dq3


        q1in = 1
        q1out = 0.125
        q2in = 0.0
        q2out = 0.0
        pin = 1.0
        pout = 0.1
        q3in = pin/(gamma-1.)
        q3out = pout/(gamma-1.)

        nx = self.nx.flatten('F')

        q1FluxIn = q2in
        q2FluxIn = np.divide(np.power(q2in,2),q1in) + pin
        q3FluxIn = (pin/(gamma-1) + 0.5*np.divide(q2in*q2in,q1in)+pin)*np.divide(q2in,q1in)

        lmIn = lm[self.vmapI]/2
        nxIn = nx[self.mapI]

        dq1Flux[self.mapI] = np.dot(nxIn,q1Flux[self.vmapI]-q1FluxIn)/2 - np.dot(lmIn,q1[self.vmapI]-q1in)
        dq2Flux[self.mapI] = np.dot(nxIn,q2Flux[self.vmapI]-q2FluxIn)/2 - np.dot(lmIn,q2[self.vmapI]-q2in)
        dq3Flux[self.mapI] = np.dot(nxIn,q3Flux[self.vmapI]-q3FluxIn)/2 - np.dot(lmIn,q3[self.vmapI]-q3in)

        q1FluxOut = q2out
        q2FluxOut = np.divide(np.power(q2out, 2), q1out) + pout
        q3FluxOut = (pout / (gamma - 1) + 0.5 * np.divide(q2out * q2out, q1out) + pout) * np.divide(q2out, q1out)
        lmOut = lm[self.vmapO] / 2
        nxOut = nx[self.mapO]

        dq1Flux[self.mapO] = np.dot(nxOut, q1Flux[self.vmapO] - q1FluxOut) / 2 - np.dot(lmOut, q1[self.vmapO] - q1out)
        dq2Flux[self.mapO] = np.dot(nxOut, q2Flux[self.vmapO] - q2FluxOut) / 2 - np.dot(lmOut, q2[self.vmapO] - q2out)
        dq3Flux[self.mapO] = np.dot(nxOut, q3Flux[self.vmapO] - q3FluxOut) / 2 - np.dot(lmOut, q3[self.vmapO] - q3out)

        q1Flux = np.reshape(q1Flux, (self.Np, self.K), 'F')
        q2Flux = np.reshape(q2Flux, (self.Np, self.K), 'F')
        q3Flux = np.reshape(q3Flux, (self.Np, self.K), 'F')

        dq1Flux = np.reshape(dq1Flux,((self.Nfp*self.Nfaces,self.K)),'F')
        dq2Flux = np.reshape(dq2Flux,((self.Nfp*self.Nfaces,self.K)),'F')
        dq3Flux = np.reshape(dq3Flux,((self.Nfp*self.Nfaces,self.K)),'F')

        rhsq1 = (-self.rx*np.dot(self.Dr,q1Flux) + np.dot(self.LIFT,self.Fscale*dq1Flux))
        rhsq2 = (-self.rx*np.dot(self.Dr,q2Flux) + np.dot(self.LIFT,self.Fscale*dq2Flux))
        rhsq3 = (-self.rx*np.dot(self.Dr,q3Flux) + np.dot(self.LIFT,self.Fscale*dq3Flux))

        return rhsq1,rhsq2,rhsq3

    def EulerRHS1DImplicit(self,time,q):
        q1 = q[0:int(len(q)/3)]
        q2 = q[int(len(q)/3):int(2*len(q)/3)]
        q3 = q[-int(len(q)/3):]

        gamma = 1.4

        pressure = (gamma-1.)*(q3-0.5*np.divide(q2*q2,q1))
        cvel = np.sqrt(gamma*np.divide(pressure,q1))
        lm = np.abs(np.divide(q2,q1)) + cvel

        q1Flux = q2
        q2Flux = np.divide(np.power(q2,2),q1) + pressure
        q3Flux = np.divide((q3+pressure)*q2,q1)

        dq1 = q1[self.vmapM] - q1[self.vmapP]
        dq2 = q2[self.vmapM] - q2[self.vmapP]
        dq3 = q3[self.vmapM] - q3[self.vmapP]

        dq1Flux = q1Flux[self.vmapM] - q1Flux[self.vmapP]
        dq2Flux = q2Flux[self.vmapM] - q2Flux[self.vmapP]
        dq3Flux = q3Flux[self.vmapM] - q3Flux[self.vmapP]

        LFc = np.maximum((lm[self.vmapM]),(lm[self.vmapP]))

        dq1Flux = self.nx.flatten('F')*dq1Flux/2. - LFc/2. * dq1
        dq2Flux = self.nx.flatten('F')*dq2Flux/2. - LFc/2. * dq2
        dq3Flux = self.nx.flatten('F')*dq3Flux/2. - LFc/2. * dq3


        q1in = 1
        q1out = 0.125
        q2in = 0.0
        q2out = 0.0
        pin = 1.0
        pout = 0.1
        q3in = pin/(gamma-1.)
        q3out = pout/(gamma-1.)

        nx = self.nx.flatten('F')

        q1FluxIn = q2in
        q2FluxIn = np.divide(np.power(q2in,2),q1in) + pin
        q3FluxIn = (pin/(gamma-1) + 0.5*np.divide(q2in*q2in,q1in)+pin)*np.divide(q2in,q1in)

        lmIn = lm[self.vmapI]/2
        nxIn = nx[self.mapI]

        dq1Flux[self.mapI] = np.dot(nxIn,q1Flux[self.vmapI]-q1FluxIn)/2 - np.dot(lmIn,q1[self.vmapI]-q1in)
        dq2Flux[self.mapI] = np.dot(nxIn,q2Flux[self.vmapI]-q2FluxIn)/2 - np.dot(lmIn,q2[self.vmapI]-q2in)
        dq3Flux[self.mapI] = np.dot(nxIn,q3Flux[self.vmapI]-q3FluxIn)/2 - np.dot(lmIn,q3[self.vmapI]-q3in)

        q1FluxOut = q2out
        q2FluxOut = np.divide(np.power(q2out, 2), q1out) + pout
        q3FluxOut = (pout / (gamma - 1) + 0.5 * np.divide(q2out * q2out, q1out) + pout) * np.divide(q2out, q1out)
        lmOut = lm[self.vmapO] / 2
        nxOut = nx[self.mapO]

        dq1Flux[self.mapO] = np.dot(nxOut, q1Flux[self.vmapO] - q1FluxOut) / 2 - np.dot(lmOut, q1[self.vmapO] - q1out)
        dq2Flux[self.mapO] = np.dot(nxOut, q2Flux[self.vmapO] - q2FluxOut) / 2 - np.dot(lmOut, q2[self.vmapO] - q2out)
        dq3Flux[self.mapO] = np.dot(nxOut, q3Flux[self.vmapO] - q3FluxOut) / 2 - np.dot(lmOut, q3[self.vmapO] - q3out)

        q1Flux = np.reshape(q1Flux, (self.Np, self.K), 'F')
        q2Flux = np.reshape(q2Flux, (self.Np, self.K), 'F')
        q3Flux = np.reshape(q3Flux, (self.Np, self.K), 'F')

        dq1Flux = np.reshape(dq1Flux,((self.Nfp*self.Nfaces,self.K)),'F')
        dq2Flux = np.reshape(dq2Flux,((self.Nfp*self.Nfaces,self.K)),'F')
        dq3Flux = np.reshape(dq3Flux,((self.Nfp*self.Nfaces,self.K)),'F')

        rhsq1 = (-self.rx*np.dot(self.Dr,q1Flux) + np.dot(self.LIFT,self.Fscale*dq1Flux))
        rhsq2 = (-self.rx*np.dot(self.Dr,q2Flux) + np.dot(self.LIFT,self.Fscale*dq2Flux))
        rhsq3 = (-self.rx*np.dot(self.Dr,q3Flux) + np.dot(self.LIFT,self.Fscale*dq3Flux))

        return np.concatenate((rhsq1.flatten('F'), rhsq2.flatten('F'), rhsq3.flatten('F')), axis=0)

    def ExplicitIntegration(self,q1,q2,q3):

        time = 0

        mindeltax = np.min(np.abs(self.x[0, :] - self.x[1, :]))
        CFL = 1.

        gamma = 1.4

        solq1 = [q1]
        solq2 = [q2]
        solq3 = [q3]

        tVec = [time]

        q1 = DG_1D.SlopeLimitN(self, q1)
        q2 = DG_1D.SlopeLimitN(self, q2)
        q3 = DG_1D.SlopeLimitN(self, q3)

        while time < self.FinalTime:
            temp = np.divide((q3 - 0.5 * np.divide(q2 * q2, q1)), q1)
            cvel = np.sqrt(gamma * (gamma - 1) * temp)
            dt = CFL * np.min(np.min(mindeltax / np.abs(np.divide(q2, q1) + cvel)))

            rhsq1, rhsq2, rhsq3 = self.EulerRHS1D(time, q1, q2, q3)
            q1_1 = q1 + dt * rhsq1
            q2_1 = q2 + dt * rhsq2
            q3_1 = q3 + dt * rhsq3

            q1_1 = DG_1D.SlopeLimitN(self, q1_1)
            q2_1 = DG_1D.SlopeLimitN(self, q2_1)
            q3_1 = DG_1D.SlopeLimitN(self, q3_1)

            rhsq1, rhsq2, rhsq3 = self.EulerRHS1D(time, q1_1, q2_1, q3_1)
            q1_2 = (3 * q1 + q1_1 + dt * rhsq1) / 4
            q2_2 = (3 * q2 + q2_1 + dt * rhsq2) / 4
            q3_2 = (3 * q3 + q3_1 + dt * rhsq3) / 4

            q1_2 = DG_1D.SlopeLimitN(self, q1_2)
            q2_2 = DG_1D.SlopeLimitN(self, q2_2)
            q3_2 = DG_1D.SlopeLimitN(self, q3_2)

            rhsq1, rhsq2, rhsq3 = self.EulerRHS1D(time, q1_2, q2_2, q3_2)
            q1 = (q1 + 2 * q1_2 + 2 * dt * rhsq1) / 3
            q2 = (q2 + 2 * q2_2 + 2 * dt * rhsq2) / 3
            q3 = (q3 + 2 * q3_2 + 2 * dt * rhsq3) / 3

            q1 = DG_1D.SlopeLimitN(self, q1)
            q2 = DG_1D.SlopeLimitN(self, q2)
            q3 = DG_1D.SlopeLimitN(self, q3)

            solq1.append(q1)
            solq2.append(q2)
            solq3.append(q3)

            time = time + dt
            tVec.append(time)

        return solq1, solq2, solq3, tVec

    def ImplicitIntegration(self,q1,q2,q3):

        N_steps = int(self.FinalTime / self.stepsize)

        #q1 = self.SlopeLimitN(q1)
        #q2 = self.SlopeLimitN(q2)
        #q3 = self.SlopeLimitN(q3)

        initCondition = np.concatenate((q1.flatten('F'),q2.flatten('F'),q3.flatten('F')),axis=0)

        system = BDF2(self.EulerRHS1DImplicit,initCondition,t0=0,te=self.FinalTime,Ntime=N_steps,xmin=self.xmin, xmax=self.xmax, K=self.K, N=self.N)
        t_vec,solution = system.solve()

        return solution[:,0:int(solution.shape[1]/3)],solution[:,int(solution.shape[1]/3):2*int(solution.shape[1]/3)], solution[:,-int(solution.shape[1]/3):], t_vec

    def solve(self,q1,q2,q3, FinalTime,implicit=False,stepsize=1e-5):
        self.FinalTime = FinalTime
        self.implicit = implicit
        self.stepsize = stepsize

        t0 = timing.time()
        if implicit:
            solq1,solq2,solq3,  tVec = self.ImplicitIntegration(q1,q2,q3)
        else:
            solq1,solq2,solq3,  tVec = self.ExplicitIntegration(q1,q2,q3)
        t1 = timing.time()

        print('Simulation finished \nTime: {:.2f} seconds'.format(t1 - t0))

        return solq1,solq2,solq3, tVec

class NSWE1D(DG_1D):
    def __init__(self, xmin=0, xmax=1, K=10, N=5):
        DG_1D.__init__(self, xmin=xmin, xmax=xmax, K=K, N=N)
    def NSWERHS1D(self,time,q1,q2):

        g = 9.81

        q1 = q1.flatten('F')
        q2 = q2.flatten('F')

        u = np.divide(q2,q1)

        q1Flux = q2
        q2Flux = q1*np.power(u,2) + 0.5*g * np.power(q1,2)

        lam = np.max(np.abs(np.concatenate((u+np.sqrt(g*q1/2),u-np.sqrt(g*q1/2)))))

        C = np.max(lam)

        nx = self.nx.flatten('F')

        f1star = 1/2*(q1Flux[self.vmapM] + q1Flux[self.vmapP]) + C/2. * nx*(q1[self.vmapM] - q1[self.vmapP])
        f2star = 1/2*(q2Flux[self.vmapM] + q2Flux[self.vmapP]) + C/2. * nx*(q2[self.vmapM] - q2[self.vmapP])
        dq1Flux = nx*(q1Flux[self.vmapM]-f1star)
        dq2Flux = nx*(q2Flux[self.vmapM]-f2star)

        q2in = 0.
        q2out = 0.

        q1FluxIn = q2in
        q2FluxIn = q1[self.vmapI] * np.power(q2in, 2) + 0.5 * g * np.power(q1[self.vmapI], 2)
        q1FluxOut = q2out
        q2FluxOut = q1[self.vmapO] * np.power(q2out, 2) + 0.5 * g * np.power(q1[self.vmapO], 2)

        dq1Flux[self.mapI] = np.dot(nx[self.mapI],q1Flux[self.vmapI]-q1FluxIn)/2
        dq2Flux[self.mapI] = np.dot(nx[self.mapI],q2Flux[self.vmapI]-q2FluxIn)/2 - np.dot(C,q2[self.vmapI]-q2in)

        dq1Flux[self.mapO] = np.dot(nx[self.mapO],q1Flux[self.vmapO]-q1FluxOut)/2
        dq2Flux[self.mapO] = np.dot(nx[self.mapO],q2Flux[self.vmapO]-q2FluxOut)/2 - np.dot(C,q2[self.vmapO]-q2out)



        q1Flux = np.reshape(q1Flux, (self.Np, self.K), 'F')
        q2Flux = np.reshape(q2Flux, (self.Np, self.K), 'F')

        dq1Flux = np.reshape(dq1Flux,((self.Nfp*self.Nfaces,self.K)),'F')
        dq2Flux = np.reshape(dq2Flux,((self.Nfp*self.Nfaces,self.K)),'F')

        rhsq1 = (-self.rx*np.dot(self.Dr,q1Flux) + np.dot(self.LIFT,self.Fscale*dq1Flux))
        rhsq2 = (-self.rx*np.dot(self.Dr,q2Flux) + np.dot(self.LIFT,self.Fscale*dq2Flux))
        return rhsq1,rhsq2

    def NSWE1D(self, q1,q2, FinalTime):

        g = 9.81
        time = 0

        mindeltax = np.min(np.abs(self.x[0,:]-self.x[1,:]))
        CFL = .3

        solq1 = [q1]
        solq2 = [q2]

        tVec = [time]

        q1 = DG_1D.SlopeLimitN(self,q1)
        q2 = DG_1D.SlopeLimitN(self,q2)

        while time < FinalTime:
            u = np.divide(q2, q1)
            lam = np.max(np.abs(np.concatenate((u + np.sqrt(g * q1 / 2), u - np.sqrt(g * q1 / 2)))))
            C = np.max(lam)
            dt = CFL * mindeltax/C

            rhsq1,rhsq2 = DG_1D.NSWERHS1D(self,time,q1,q2)
            q1_1 = q1 + dt*rhsq1
            q2_1 = q2 + dt*rhsq2

            q1_1 = DG_1D.SlopeLimitN(self, q1_1)
            q2_1 = DG_1D.SlopeLimitN(self, q2_1)


            rhsq1, rhsq2 = DG_1D.NSWERHS1D(self, time, q1_1, q2_1)
            q1_2 = (3*q1 + q1_1 + dt * rhsq1)/4
            q2_2 = (3*q2 + q2_1 + dt * rhsq2)/4

            q1_2 = DG_1D.SlopeLimitN(self, q1_2)
            q2_2 = DG_1D.SlopeLimitN(self, q2_2)

            rhsq1, rhsq2 = DG_1D.NSWERHS1D(self, time, q1_2, q2_2)
            q1 = (q1 + 2*q1_2 + 2*dt * rhsq1) / 3
            q2 = (q2 + 2*q2_2 + 2*dt * rhsq2) / 3

            q1 = DG_1D.SlopeLimitN(self, q1)
            q2 = DG_1D.SlopeLimitN(self, q2)

            solq1.append(q1)
            solq2.append(q2)

            time = time+dt
            tVec.append(time)

        return solq1,solq2,tVec




