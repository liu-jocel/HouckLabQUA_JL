# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:45:01 2022

@author: crgui
"""
import numpy as np
from scipy.optimize import curve_fit
import random as rd
from scipy.stats import norm

class RBAnalysis:

    def __init__(self, name, numQubits, gates, mAxis, data):
        
        self.method=2
        
        # identifying data
        self.name = name
        self.data = data
        self.numQubits = numQubits
        self.d = 2**numQubits
        self.gates = gates
        self.numGates = np.shape(data)[0]
        self.mAxis = mAxis
        self.numMs = np.shape(data)[2]
        self.numks = np.shape(data)[1]
        self.numShots = np.shape(data)[3]
        
        # useful structures
        self.traces=self.makeTraces(data)
        self.kAvgs=self.makekAverages(data)
        
        # fitting params
        self.pGuess=0.99
        self.AGuess=1-1/(2**numQubits)
        self.BGuess=1/(2**numQubits)
        self.CGuess = 0.01
        
        # find naive estimates
        self.naivePs,self.naiveResults=self.naiveFit(self.traces)
        self.magesanFit()
        
        # bootstrap parameters
        self.numBoots = 1000
        self.bootSamps = self.numks
        self.bootks = self.numks
        self.kStart = 0
        self.bootShots = self.numShots

    def __RBFitZerothOrder(self,m,p,A,B):
        return A*p**m+B

    def __RBFitFirstOrder(self,m,p,A,B,C):
        return A*p**m + C*(m-1)*p**(m-2) + B

    def __gateFidel(self,pRef,pGate,pRefErr,pGateErr):
        fidel = 1-((self.d-1)/(self.d)*(1-pGate/pRef))
        err = (self.d-1)/(self.d) * pGate/pRef * np.sqrt((pRefErr/pRef)**2+(pGateErr/pGate)**2)
        return fidel,err
    
    def makeTraces(self,data):
        traces = np.zeros([self.numGates,self.numMs])
        
        for g in range(self.numGates):
            for m in range(self.numMs):
                traces[g][m]=np.mean(data[g,:,m,:])
                
        return traces
    
    def makeTraceskAvg(self,data):
        
        traces = np.zeros([self.numGates,self.numMs])
        
        for g in range(self.numGates):
            for m in range(self.numMs):
                traces[g][m]=np.mean(data[g,m,:])
        
        return traces
    
    def makekAverages(self,data):
        kAvgs = np.zeros([self.numGates,self.numMs,self.numks])
        
        for g in range(self.numGates):
            for m in range(self.numMs):
                for k in range(self.numks):
                    kAvgs[g][m][k]=np.mean(data[g,k,m,:])
        
        return kAvgs

    def naiveFit(self,traces):
        
        numGates=self.numGates
        ps=np.zeros((numGates,2))
        
        for g in range(numGates):
            
            optResult=curve_fit(self.__RBFitZerothOrder, self.mAxis, traces[g],p0=[self.pGuess,self.AGuess,self.BGuess])
            ps[g][0]=optResult[0][0]
            ps[g][1]=np.sqrt(optResult[1][0][0])
        
        naiveResults=np.zeros((numGates-1,2))
        
        for g in range(numGates-1):
            naiveResults[g][0],naiveResults[g][1]=self.__gateFidel(ps[0][0],ps[g+1][0],ps[0][1],ps[g+1][1])
        
        return ps,naiveResults
    
    def magesanFit(self):
        
        traces=self.traces
        numGates=self.numGates
        ps=np.zeros(numGates)
        d=self.d
        
        for g in range(numGates):
            
            optResult=curve_fit(self.__RBFitZerothOrder, self.mAxis, traces[g],p0=[self.pGuess,self.AGuess,self.BGuess])
            ps[g]=optResult[0][0]
        
        magesanResults=np.zeros((numGates-1,2))
        
        for t in range(numGates-1):
            magesanResults[t][0]=self.__gateFidel(ps[0],ps[t+1],0,0)[0]
            magesanResults[t][1]=min([(d-1)/d*(np.abs(ps[0]-(ps[t+1]/ps[0]))+1-ps[0]),2*(d**2-1)*(1-ps[0])/(ps[0]*d**2)+4*np.sqrt(1-ps[0])*np.sqrt(d**2-1)/ps[0]])
        
        self.magesanResults=magesanResults
        
    def runBootstrap(self):
        
        numBoots = self.numBoots
        numGates = self.numGates
        bootSamps = self.bootSamps
        numMs = self.numMs
        numShots = self.numShots
        bootks = self.bootks
        kStart = self.kStart
        bootShots = self.bootShots
        
        fidels=np.zeros((numBoots,numGates-1))
        fidelErrs=np.zeros((numBoots,numGates-1))
        
        data=self.data[:,:,:,0:bootShots]
        
        kAvgs=self.makekAverages(data)
        
        if self.method==1:
            for b in range(numBoots):
                
                tempData=np.zeros([numGates,numMs,bootSamps,numShots])
                
                for g in range(numGates):
                    for m in range(numMs):
                        for s in range(bootSamps):
                        
                            kSamp = rd.randint(kStart,kStart+bootks-1)
                            
                            tempData[g][m][s]=data[g][m][kSamp]
                
                tempTraces=self.makeTraces(tempData)
                tempps,tempfs = self.naiveFit(tempTraces)
                fidels[b]=tempfs[:,0]
                fidelErrs[b]=tempfs[:,1]
        elif self.method==2:
            for b in range(numBoots):
                
                tempkAvgs=np.zeros([numGates,numMs,bootSamps])
                
                for g in range(numGates):
                    for m in range(numMs):
                        for s in range(bootSamps):
                        
                            kSamp = rd.randint(kStart,kStart+bootks-1)
                            
                            tempkAvgs[g][m][s]=kAvgs[g][m][kSamp]
                
                tempTraces=self.makeTraceskAvg(tempkAvgs)
                tempps,tempfs = self.naiveFit(tempTraces)
                fidels[b]=tempfs[:,0]
                fidelErrs[b]=tempfs[:,1]
        
        self.fidelsBoot=fidels
        self.fidelErrsBoot=fidelErrs
        self.CI()
        
    def CI(self):
        numGates = self.numGates
        results = np.zeros((numGates-1,2))
        errs = np.zeros((numGates-1,2))
        
        for g in range(numGates-1):
            results[g][0], results[g][1] = norm.fit(self.fidelsBoot[:,g])
            errs[g][0], errs[g][1] = norm.fit(self.fidelErrsBoot[:,g])
        
        self.bootResults=results
        self.bootErrResults=errs
    
    
    def printResults(self,results):
        gates=self.gates
        for g in range(self.numGates-1):
            print(gates[g+1],': F = %.5f +/- %.5f' % (results[g][0], results[g][1]))