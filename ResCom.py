# -*- coding: utf-8 -*-
"""

Created on Tue July 06 12:46:32 2021

@author: David Fox
"""

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import splprep, splev
import scipy.integrate as integrate
from scipy import stats

    
    
class ESN(object):
    """
    Implementation of an echo state network (ESN).
    
    Instance Variables:
            seed (int): Random seed, used in generating W_in_orig and M_orig.
            N (int): Dimension of hidden layer.
            d (int): Dimension of input layer.
            p (double): Density of hidden layer connections.
            rho (double): Spectral radius of adjacency matrix M.
            gamma (double): Controls time scale of hidden layer dynamics.
            sigma (double): Scales input signal strength.
            W_in_orig (Nxd sparse csr matrix): Matrix describing connections from input layer to hidden layer.
            W_in (Nxd sparse csr matrix): W_in_orig, scaled by sigma.
            M_orig (NxN sparse csr matrix): Adjacency matrix for hidden layer connections.
            M (NxN sparse csr matrix): M_orig, scaled to have spectral radius rho.
            W_out (dx2N matrix): Output layer connections learned during training.
            beta (double): Regularization parameter used when training W_out.
            r_T (Nd array): Hidden layer state after training.
            
    """
    def __init__(self, N, p, d, rho, sigma, gamma, beta, seed=1):      
        # Create input layer.
        self.N = N
        self.d = d
        self.seed = seed
        self.W_in_orig = self.input_matrix()
        self.sigma = sigma
        
        # Create reservoir adjacency matrix.
        self.p = p
        self.M_orig = self.adj_matrix()
        self.rho = rho
        
        # Output layer initially 'None' before training.
        self.W_out = None
    
        self.gamma = gamma
        self.beta = beta
        self.u = None
        self.r_T = None
        
        
    @property
    def rho(self):
        return self.__rho
    
    
    @rho.setter
    def rho(self, rho):
        """
        Changes adjacency matrix M by rescaling M_orig after resetting rho.

        """
        self.__rho = rho
        self.M = rho * self.M_orig
        
        
    @property
    def sigma(self):
        return self.__sigma

    
    @sigma.setter
    def sigma(self, sigma):
        """
        Changes input matrix W_in by rescaling W_in_orig after resetting sigma.

        """
        self.__sigma = sigma
        self.W_in = sigma * self.W_in_orig
    
    
    def f_PR(self, r, t, *args):
        v = self.W_out.dot(self.q(r))
        return self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v)))
    
    
    def f_LR(self, r, t, *args):
        return self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t))))
        
        
    def adj_matrix(self):
        """
        Generates a random Erdos-Renyi NxN sparse csr matrix and its max eigenvalue, returns 
        """
        
        np.random.seed(seed = self.seed)
        rvs = stats.uniform(loc=-1, scale=2).rvs
        M = sp.random(self.N, self.N, self.p, format='csr', data_rvs=rvs)
        max_eval = np.abs(sp.linalg.eigs(M, 1, which='LM', return_eigenvectors=False)[0])
        
        return (1/max_eval)*M


    def input_matrix(self):
        """
        Generates a random Nxd sparse csr matrix with 1 entry in each 
        row sampled from UNIF(-1,1).
        """
        
        np.random.seed(seed = self.seed)
        # Create sparse matrix in COO form, then cast to CSR form.
        rows = np.arange(0, self.N)
        cols = stats.randint(0, self.d).rvs(self.N)
        values = stats.uniform(loc=-1, scale=2).rvs(self.N)
        W_in = sp.coo_matrix((values, (rows, cols))).tocsr()
        
        return W_in
    
    
    def q(self, r):
        x = np.zeros(2*self.N)
        x[0:self.N] = r
        x[self.N: 2*self.N] = r**2
        
        return x
    

    def spline(self, data, t):
        coords = [data[:,i] for i in range(self.d)]
        tck, u = splprep(coords, u = t, s=0)
        return lambda t: np.asarray(splev(t, tck))
    
    
    def train(self, data, t, t_listen):
        # Integrate Listening reservoir system
        if type(self.u) == type(None):    
            self.u = self.spline(data, t)
        LR_traj = integrate.odeint(self.f_LR, np.zeros(self.N), t)
        
        X = np.zeros((2*self.N, t.size - t_listen))
        Y = np.transpose(data[t_listen:])
        for i in range(t_listen, t.size - 1):
              X[:,i+1 - t_listen] = self.q(LR_traj[i+1])
                    
        # Calculate output matrix.
        X_T = np.transpose(X)
        M_1 = np.matmul(Y, X_T)
        M_2 = np.linalg.inv(np.matmul(X, X_T) + self.beta*np.identity(2*self.N))
        self.W_out = np.matmul(M_1, M_2)
        self.r_T = LR_traj[-1]
        
        return LR_traj
    
    
    def predict(self, t_predict, data=None, t=None):
        if type(data) == type(None):
            PR_traj = integrate.odeint(self.f_PR, self.r_T, t_predict)
        else:
            self.u = self.spline(data, t)
            LR_traj = integrate.odeint(self.f_LR, np.zeros(self.N), t)
            PR_traj = integrate.odeint(self.f_PR, LR_traj[-1], t_predict)
            
        prediction = np.asarray([self.W_out.dot(self.q(p)) for p in PR_traj])
        return prediction
        
        
class AHESN(ESN):
    def __init__(self, N, p, d, rho, sigma, gamma, beta, eta, epochs, seed=1):
        ESN.__init__(self, N, p, d, rho, sigma, gamma, beta, seed)
        self.eta = eta
        self.epochs = epochs
        
        
    def f_PR(self, r, t, *args):
        v = self.W_out.dot(self.q(r))
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v)))
        return np.squeeze(np.asarray(f))
    
    
    def f_LR(self, r, t, *args):
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t))))
        return np.squeeze(np.asarray(f))
        
    
    def hebb_learn(self, data, t, t_listen, rescale=True):
        if rescale:
            self.rho = self.rho 
        self.u = self.spline(data, t)
        x = integrate.odeint(self.f_LR, np.zeros(self.N), t)
        X = np.zeros((self.N,self.N))
        M = self.M_orig.todense().copy()
    
        for e in range(self.epochs):
            for t in range(t_listen, data.shape[0]-1):
                for i in range(self.N):
                    X[i,:] = x[t+1,i] * x[t,:]
                M = M - self.eta*X
        
        self.M = self.rho*sp.csr_matrix(M)
        return None
    
    
    def hebb_learn_vector(self, data, t, t_listen, reset_M=True, scale_M=0):
        if reset_M:
            self.rho = self.rho 
        self.u = self.spline(data, t)
        x = integrate.odeint(self.f_LR, np.zeros(self.N), t)
        M = self.M.copy()
    
        for e in range(self.epochs):
            for t in range(t_listen, data.shape[0]-1):
                M = M - self.eta*(np.asarray([x[t+1]]).T)@np.asarray([x[t]])
                if scale_M == 2:
                    max_eval = np.abs(sp.linalg.eigs(M, 1, which='LM', return_eigenvectors=False)[0])
                    M = (self.rho/max_eval)*M
        if scale_M == 1:
            M = sp.csr_matrix(M)
            max_eval = np.abs(sp.linalg.eigs(M, 1, which='LM', return_eigenvectors=False)[0])
            self.M = (self.rho/max_eval)*M
        elif scale_M == 2:
            self.M = M
        else:
            self.M = sp.csr_matrix(M)
        return None
    
    
    def norm_hebb_learn(self, data, t, t_listen, reset_M=True):
        if reset_M:
            self.rho = self.rho 
        self.u = self.spline(data, t)
        x = integrate.odeint(self.f_LR, np.zeros(self.N), t)
        M = self.M_orig.copy()
    
        for e in range(self.epochs):
            for t in range(t_listen, data.shape[0]-1):
                M_hat = M - self.eta*(np.asarray([x[t+1]]).T)@np.asarray([x[t]])
                M = np.diag(1/(np.linalg.norm(M_hat, axis=0)))@M_hat
        
        self.M = self.rho*sp.csr_matrix(M)
        return None
    
    
class IPESN(AHESN):
    def __init__(self, N, p, d, rho, sigma, gamma, beta, nu, mu, sd, eta=0, epochs=0, seed=1):
        AHESN.__init__(self, N, p, d, rho, sigma, gamma, beta, eta, epochs, seed)
        self.nu = nu
        self.mu=mu
        self.sd=sd
        self.a = np.ones(N)
        self.b = np.zeros(N)
    
    
    def f_PR(self, r, t, *args):
        v = self.W_out.dot(self.q(r))
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(v) + self.b))
        return np.squeeze(np.asarray(f))
    
    
    def f_LR(self, r, t, *args):
        f = self.gamma*(-r + np.tanh(self.M.dot(r) + self.W_in.dot(self.u(t)) + self.b))
        return np.squeeze(np.asarray(f))
    
    
    def H(self, x):
        return -self.mu/self.sd**2 + (x/self.sd**2)*(2*self.sd**2 + 1 - x**2 + self.mu*x)
    
    
    def IP_train(self, data, t_points, epochs, rescale=True):
        if rescale:
            self.rho = self.rho
            self.sigma=self.sigma
        self.u = self.spline(data, t_points)
        x = integrate.odeint(self.f_LR, np.zeros(self.N), t_points)
        z = np.asarray([self.M.dot(x[t]) + self.W_in.dot(self.u(t_points[t])) for t in range(len(t_points))])
        
        for e in range(epochs):
            for t in range(len(t_points)):
                delta_b = -self.nu*self.H(x[t])
                delta_a = self.nu/self.a + np.dot(np.diag(delta_b), z[t])
                self.b += delta_b
                self.a += delta_a
        
        self.M = self.rho*np.matmul(np.diag(self.a), self.M.todense())
        self.W_in = self.sigma*np.matmul(np.diag(self.a), self.W_in.todense())
            
        
        
    
        
        
        
        
        
    
        