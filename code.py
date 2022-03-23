import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from math import exp,log

K= 120
r1= np.array([3, 5, 2, 7, 7, 2, 5, 3, 5, 11, 6, 6, 11, 4, 4, 2, 8, 8, 6, 
5, 15, 4, 9, 9, 4, 12, 8, 8, 6, 8, 12, 4, 7, 16, 12, 9, 4, 7, 
8, 11, 5, 12, 8, 17, 9, 3, 2, 7, 6, 5, 11, 14, 13, 8, 6, 4, 8, 
4, 8, 7, 15, 15, 9, 9, 5, 6, 3, 9, 12, 14, 16, 17, 8, 8, 9, 5, 
9, 11, 6, 14, 21, 16, 6, 9, 8, 9, 8, 4, 11, 11, 6, 9, 4, 4, 9, 
9, 10, 14, 6, 3, 4, 6, 10, 4, 3, 3, 10, 4, 10, 5, 4, 3, 13, 1, 
7, 5, 7, 6, 3, 7])
n1= np.array([28, 21, 32, 35, 35, 38, 30, 43, 49, 53, 31, 35, 46, 53, 61, 
40, 29, 44, 52, 55, 61, 31, 48, 44, 42, 53, 56, 71, 43, 43, 43, 
40, 44, 70, 75, 71, 37, 31, 42, 46, 47, 55, 63, 91, 43, 39, 35, 
32, 53, 49, 75, 64, 69, 64, 49, 29, 40, 27, 48, 43, 61, 77, 55, 
60, 46, 28, 33, 32, 46, 57, 56, 78, 58, 52, 31, 28, 46, 42, 45, 
63, 71, 69, 43, 50, 31, 34, 54, 46, 58, 62, 52, 41, 34, 52, 63, 
59, 88, 62, 47, 53, 57, 74, 68, 61, 45, 45, 62, 73, 53, 39, 45, 
51, 55, 41, 53, 51, 42, 46, 54, 32])
c= np.array([0, 2, 2, 1, 2, 0, 1, 1, 1, 2, 4, 4, 2, 1, 7, 4, 3, 5, 3, 2, 
4, 1, 4, 5, 2, 7, 5, 8, 2, 3, 5, 4, 1, 6, 5, 11, 5, 2, 5, 8, 
5, 6, 6, 10, 7, 5, 5, 2, 8, 1, 13, 9, 11, 9, 4, 4, 8, 6, 8, 6, 
8, 14, 6, 5, 5, 2, 4, 2, 9, 5, 6, 7, 5, 10, 3, 2, 1, 7, 9, 13, 
9, 11, 4, 8, 2, 3, 7, 4, 7, 5, 6, 6, 5, 6, 9, 7, 7, 7, 4, 2, 
3, 4, 10, 3, 4, 2, 10, 5, 4, 5, 4, 6, 5, 3, 2, 2, 4, 6, 4, 1])
n0= np.array([28, 21, 32, 35, 35, 38, 30, 43, 49, 53, 31, 35, 46, 53, 61, 
40, 29, 44, 52, 55, 61, 31, 48, 44, 42, 53, 56, 71, 43, 43, 43, 
40, 44, 70, 75, 71, 37, 31, 42, 46, 47, 55, 63, 91, 43, 39, 35, 
32, 53, 49, 75, 64, 69, 64, 49, 29, 40, 27, 48, 43, 61, 77, 55, 
60, 46, 28, 33, 32, 46, 57, 56, 78, 58, 52, 31, 28, 46, 42, 45, 
63, 71, 69, 43, 50, 31, 34, 54, 46, 58, 62, 52, 41, 34, 52, 63, 
59, 88, 62, 47, 53, 57, 74, 68, 61, 45, 45, 62, 73, 53, 39, 45, 
51, 55, 41, 53, 51, 42, 46, 54, 32])
year= np.array([-10, -9, -9, -8, -8, -8, -7, -7, -7, -7, -6, -6, -6, -6, -6, 
-5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, 
-3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 
3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 
6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10])

def GibbsSampler(nchain, initialisation, data, param) :
    #chain[i,] = [alpha, beta1, beta2, sigma]
    
    #initialisation
    chain = np.zeros((nchain + 1, 244))
    chain[0,:] = initialisation
    
    
    for i in range(nchain):
        
        chain[i+1]=chain[i]
        
     #mise à jour de alpha
        alpha_prop=chain[i+1,0]+np.random.normal()
        
        psi_old=chain[i+1,0]+chain[i+1,1]*year+chain[i+1,2]*(year**2-22)+chain[i+1,4:124]
        psi_prop=alpha_prop+chain[i+1,1]*year+chain[i+1,2]*(year**2-22)+chain[i+1,4:124]
        
        p1_old=exp(chain[i+1,124:244]+log(psi_old))/(1+exp(chain[i+1,124:244]+log(psi_old)))
        p1_prop=exp(chain[i+1,124:244]+log(psi_prop))/(1+exp(chain[i+1,124:244]+log(psi_prop)))
        
        top=-alpha_prop**2/(2*10**-6)+np.sum(r1*p1_prop+(n1-r1)*(1-p1_prop))
        bottom=-chain[i+1,0]**2/(2*10**-6)+np.sum(r1*p1_old+(n1-r1)*(1-p1_old))
        
        if np.random.uniform()<exp(top-bottom):
            chain[i+1,0]=alpha_prop
    
    

        #mise à jour de Beta1
        beta_prop=chain[i+1,0]+np.random.normal()

        psi_old=chain[i+1,0]+chain[i+1,1]*year+chain[i+1,2]*(year**2-22)+chain[i+1,4:124]
        psi_prop=chain[i+1,0]+beta_prop*year+chain[i+1,2]*(year**2-22)+chain[i+1,4:124]

        p1_old=exp(chain[i+1,124:244]+log(psi_old))/(1+exp(chain[i+1,124:244]+log(psi_old)))
        p1_prop=exp(chain[i+1,124:244]+log(psi_prop))/(1+exp(chain[i+1,124:244]+log(psi_prop)))

        top=-beta_prop**2/(2*10**-6)+np.sum(r1*p1_prop+(n1-r1)*(1-p1_prop))
        bottom=-chain[i+1,1]**2/(2*10**-6)+np.sum(r1*p1_old+(n1-r1)*(1-p1_old))

        if np.random.uniform()<exp(top-bottom):
            chain[i+1,1]=beta_prop


      
        #mise à jour de Beta2
        beta2_prop=chain[i+1,0]+np.random.normal()

        psi_old=chain[i+1,0]+chain[i+1,1]*year+chain[i+1,2]*(year**2-22)+chain[i+1,4:124]
        psi_prop=chain[i+1,0]+chain[i+1,1]*year+beta2_prop*(year**2-22)+chain[i+1,4:124]

        p1_old=exp(chain[i+1,124:244]+log(psi_old))/(1+exp(chain[i+1,124:244]+log(psi_old)))
        p1_prop=exp(chain[i+1,124:244]+log(psi_prop))/(1+exp(chain[i+1,124:244]+log(psi_prop)))

        top=-beta2_prop**2/(2*10**-6)+np.sum(r1*p1_prop+(n1-r1)*(1-p1_prop))
        bottom=-chain[i+1,2]**2/(2*10**-6)+np.sum(r1*p1_old+(n1-r1)*(1-p1_old))

        if np.random.uniform()<exp(top-bottom):
            chain[i+1,2]=beta2_prop
    
    #mise à jour de Tau
        chain[i+1,3]=gamma.rvs(a=10**-3+60,scale=1/(10**-3+np.sum(chain[i+1,4:124])/2))
        
    
     #mise à jour des b_i
        for j in range(4,124):
            
            bj_prop=chain[i+1,j]+np.random.normal()
            
            psi_old=exp(chain[i+1,0]+chain[i+1,1]*year[j]+chain[i+1,2]*(year[j]**2-22)+chain[i+1,j])
            p1j_old=exp(chain[i+1,j+120]+log(psi_old))/(1+exp(chain[i+1,j+120]+log(psi_old)))
            
            psi_prop=exp(chain[i+1,0]+chain[i+1,1]*year[j]+chain[i+1,2]*(year[j]**2-22)+bj_prop)
            p1j_prop=exp(chain[i+1,j+120]+log(psi_prop))/(1+exp(chain[i+1,j+120]+log(psi_prop)))
            
            top=-chain[i+1,3]*bj_prop**2/2+r1[j]*p1j_prop+(n1[j]-r1[j])*(1-p1j_prop)
            bottom=-chain[i+1,3]*chain[i+1,j]**2/2+r1[j]*p1j_old+(n1[j]-r1[j])*(1-p1j_old)
            
            if np.random.uniform()<exp(top-bottom):
                chain[i+1,j]=bj_prop
    
    
     #mise à jour de sigma
  
  return(chain)



