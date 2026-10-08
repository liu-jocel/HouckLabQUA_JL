# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:24:45 2022

@author: crgui
"""

import numpy as np
import scipy.optimize as scp
import pickle
import cirq
from scipy.linalg import sqrtm

def project(rho,projs,i,j):
    return np.real(np.trace(projs[i][j]@rho))

def buildRho(ts):
    T=np.array([[ts[0],             0,                  0,              0],
                [ts[4]+1j*ts[5],    ts[1],              0,              0],
                [ts[10]+1j*ts[11],  ts[6]+1j*ts[7],     ts[2],          0],
                [ts[14]+1j*ts[15],  ts[12]+1j*ts[13],   ts[8]+1j*ts[9], ts[3]]],dtype=complex)
    rho=np.conjugate(T.T)@T/np.trace(np.conjugate(T.T)@T)
    
    return rho
    

def likelihood(ts,projs,probs):
    rho=buildRho(ts)
    
    #projs=params[0]
    #probs=params[1]
    
    L=0
    
    for i in range(9):
        for j in range(4):
            L += 0.5 *((project(rho,projs,i,j)-probs[i][j])/(np.sqrt(project(rho,projs,i,j))))**2
            
    return L

def make1QBPaulis():
    i=np.array([[1,0],[0,1]],dtype=complex)
    px=np.array([[0,1],[1,0]],dtype=complex)
    py=np.array([[0,-1j],[1j,0]],dtype=complex)
    pz=np.array([[1,0],[0,-1]],dtype=complex)
    
    p=[i,px,py,pz]
    
    return p

def make1QBProjectors():
    sx0=1/np.sqrt(2)*np.array([[1],[1]],dtype=complex)
    sx1=1/np.sqrt(2)*np.array([[1],[-1]],dtype=complex)
    sy0=1/np.sqrt(2)*np.array([[1],[1j]],dtype=complex)
    sy1=1/np.sqrt(2)*np.array([[1],[-1j]],dtype=complex)
    sz0=np.array([[1],[0]],dtype=complex)
    sz1=np.array([[0],[1]],dtype=complex)
    
    px0=np.outer(sx0,np.conjugate(sx0.T))
    px1=np.outer(sx1,np.conjugate(sx1.T))
    py0=np.outer(sy0,np.conjugate(sy0.T))
    py1=np.outer(sy1,np.conjugate(sy1.T))
    pz0=np.outer(sz0,np.conjugate(sz0.T))
    pz1=np.outer(sz1,np.conjugate(sz1.T))
    
    proj=[px0,px1,py0,py1,pz0,pz1]
    
    return proj

class paulis():
    
    def __init__(self):
        self.paulis=make1QBPaulis()
        self.projectors=make1QBProjectors()
        
    def getPauli(self,char):
        return self.paulis(char)
    
    def getProjector(self,char):
        return self.projectors(char)
    
    def tensorProduct(self,a,b):
        return np.kron(a,b)
    
    def make2QBPaulis(self):
        p=[]
        for i in range(4):
            for j in range(4):
                p.append(self.tensorProduct(self.paulis[i],self.paulis[j]))
        return p
    
    def make2QBProjectors(self):
        ps=[]
        projs=self.projectors
        for i in range(3):
            for j in range(3):
                p=[]                
                p.append(self.tensorProduct(projs[2*i],projs[2*j]))
                p.append(self.tensorProduct(projs[2*i],projs[2*j+1]))
                p.append(self.tensorProduct(projs[2*i+1],projs[2*j]))
                p.append(self.tensorProduct(projs[2*i+1],projs[2*j+1]))            
                ps.append(p)
        return ps

def findRhoMLE(probs):
    p=paulis()
            
    projs=p.make2QBProjectors()
    
    ts=[1]*16
    
    res=scp.minimize(likelihood,ts,args=(projs,probs))
    
    rho=buildRho(res.x)
    
    return rho


if __name__ == '__main__':
    
    probs=[[0.44999984,	0.02,	0.12999998,	0.39999989],
    [0.31999996,	0.13999999,	0.13999999,	0.39999989],
    [0.32999995	,0.22000003,	0.25000003,	0.20000002],
    [0.36999992	,0.18000001	,0.16,	0.28999999],
    [0.029999999,	0.45999983	,0.49999979,	0.0099999998],
    [0.42999986,	0.23000003,	0.11999998,	0.22000003],
    [0.24000004,	0.16,	0.28,	0.31999996],
    [0.22000003,	0.23000003,	0.21000002,	0.33999994],
    [0.49999979,	0,	0.059999995	,0.43999985]]
    
    rho_data=findRhoMLE(probs) # test Bell state from old data
    
    # try to simulate from measured moments
    
    q1,q2 = cirq.LineQubit.range(2)
    prep_circuit = cirq.Circuit()
    prep_circuit.append(cirq.H(q1))
    prep_circuit.append(cirq.CNOT(q1,q2))
    
    prep_circuit_unitary = np.matrix(cirq.unitary(prep_circuit))
    initial_state = np.matrix([[1],
                               [0],
                               [0],
                               [0]],dtype=complex)
    
    prep_rho = prep_circuit_unitary @ np.outer(initial_state,initial_state.H) @ prep_circuit_unitary.H
    depol_prob = 0.0 # depolarize this a bit to help the random sampling out
    prep_rho = (1-depol_prob)*prep_rho + depol_prob / 4 * np.identity(4)
    
    with open('meas_moments.pkl','rb') as f:
        meas_moments = pickle.load(f)
        
    probs = np.zeros([9,4])
    command_to_unitary={'-Y/2':np.matrix(cirq.unitary(cirq.Y(q1)**-0.5)),
                        'X/2':np.matrix(cirq.unitary(cirq.X(q1)**0.5)),
                        'I':np.matrix(cirq.unitary(cirq.Y(q1)**2))}
    
    for m in range(9):
        moment = meas_moments[m]
        q1_U = command_to_unitary[moment['Q1']]
        q2_U = command_to_unitary[moment['Q2']]
        meas_U = np.kron(q1_U,q2_U)
        meas_rho = meas_U @ prep_rho @ meas_U.H
        for o in range(4):
            probs[m][o] = np.real(meas_rho.item((o,o)))
    
    rho_comp = findRhoMLE(probs)
    
    # trace distance between result and intended state
    print('T =',1/2*np.trace(sqrtm((rho_comp-prep_rho).H@(rho_comp-prep_rho))))
        