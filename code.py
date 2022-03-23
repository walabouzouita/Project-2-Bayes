import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def GibbsSampler(nchain, initialisation, data, param) :
    #chain[i,] = [alpha, beta1, beta2, sigma]
    
    #initialisation
    chain = np.zeros((nchain + 1, 4))
    chain[0,:] = initialisation
    
    
    for i in range(nchain):
      
     #mise à jour de alpha
    
    
    
     #mise à jour de Beta1
      
      
      
     #mise à jour de Beta2
    
    
    
    
     #mise à jour de sigma
  
  return(chain)



