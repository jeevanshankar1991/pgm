import sys
from numpy import *
import numpy as np
import math
from PIL import Image
import random 
import png
import os, os.path

global WP, WB, WC, D, K


''' Save the image given 1D binary vector and a name into output directory '''
def save_img(x, name):
	        if type(name) == type(2):
			name = str(name)
	        Y = x.reshape(28, 28)
	        img = Image.fromarray((Y*255).astype(np.uint8))
	        # make a directory if output doesnot exist 
		if not os.path.isdir('output'):
			os.mkdir('output')
	        f = 'output/' + name + ".png" 
	        img.save(f)

''' load weights '''
def load_weights():
	global WP, WB, WC 
	WP = np.loadtxt('Models/MNISTWP.txt')
	WB = np.loadtxt('Models/MNISTWB.txt')
	WC = np.loadtxt('Models/MNISTWC.txt')
        print WP.shape, WC.shape, WB.shape

''' load_train_data '''
def load_train_data():
        #train_data = np.loadtxt('Data/MNISTXtrain.txt')
	train_data = [ [int(val) for val in line.strip().split(' ')] for line in open('Data/MNISTXtrain.txt', 'r') ] 
	return train_data

''' load test data '''
def load_test_data():
	test_data = np.loadtxt('Data/MNISTXtest.txt')
	return test_data

''' logit function '''
def logit_fn(x): 
        y = math.exp(x)
	return y / float(1 + y)

''' calculate energy '''
def energy(x, h):
	energy = dot(dot(WP, h), x)
	energy += dot(WB, h)
	energy += dot(WC, x)
	return energy

''' conditional prob(Hk = 1 | X) '''
def get_cond_prob_hk(k, x):
	prob = logit_fn(WB[k] + dot(WP[:, k], x))
	return prob

''' conditional prob(Xd = 1 | H) '''
def get_cond_prob_xd(d, h):
	prob = logit_fn(WC[d] + dot(WP[d], h))
	return prob 

''' sample x[d]  give h '''
def sample_xd(d, h): 
	prob = logit_fn(WC[d] + dot(WP[d], h))
	alpha = np.random.uniform()
	return 1 if prob > alpha else 0

''' sample h[k] given x'''
def sample_hk(k, x): 
	prob = logit_fn(WB[k] + dot(WP[:, k], x))
	alpha = np.random.uniform()
	return 1 if prob > alpha else 0

''' sample x given h '''
def sample_x(h):
	x = array(map(lambda d : sample_xd(d, h), range(D)))
	return x
''' sample h given x '''
def sample_h(x):
	h = array(map(lambda k : sample_hk(k, x), range(K)))
	return h

''' Inference : Block Gibbs Sampling '''
def block_gibbs_sampler(S, check = False):
	h = zeros([S+1, K])
	x = zeros([S+1, D])
	for k in range(K): h[0][k] = random.randint(0,1)
	for s in range(1, S+1):
		x[s] = sample_x(h[s-1])
		h[s] = sample_h(x[s])
		if s%20 == 0 and check:
		    save_img(x[s], s)
	return (x[1:,:], h[1:,:])

''' LEARNING : Learn the weights for Restricted Boltzman Machines (RBM) '''
def RBMLearn(data_x = load_train_data(), T = 50, B = 100, NB = 600, alpha = 0.1, Lambda = 1e-4):
   global WP, WC, WB
   # Initialize weights 
   WC = zeros(D); WB = zeros(K); WP = zeros([D, K]);
   
   #Initialize the parameters with Gaussian Noise
   for d in range(D): WC[d] = np.random.normal(0, 0.1)
   for k in range(K): WB[k] = np.random.normal(0, 0.1)
   for k in range(K):
	   for d in range(D):
		   WP[d][k] = np.random.normal(0, 0.1)
   ## Run the mini-batch SGD 
   print 'started mini-batch SGD ''' 
   for t in range(T):
	   print t
	   for b in range(1,B+1):
                   gpWC = zeros(D); gpWB = zeros(K); gpWP = zeros([D, K]);
                   gnWC = zeros(D); gnWB = zeros(K); gnWP = zeros([D, K]);
                   ## compute the +ve gradient contribution using the data seen
		   for n in range(1 + (b-1)*NB, b*NB):
			  gpWC = gpWC + data_x[n-1]
			  cond_prob = [ get_cond_prob_hk(k, data_x[n-1]) for k in range(K) ]
			  gpWB = gpWB + cond_prob
			  gpWP = gpWP + matrix(data_x[n-1]).T * cond_prob
		    
		    ## compute the -ve gradient contribution using data sampled using Block Gibbs Sampling
		   sampled_x, samplex_y = block_gibbs_sampler(C)
		   for c in range(C):
			   gnWC = gnWC + sampled_x[c]
			   cond_prob = [ get_cond_prob_hk(k, sampled_x[c]) for k in range(K) ]
			   gnWB = gnWB + cond_prob
			   gnWP = gnWP + matrix(sampled_x[c]).T * cond_prob

	            ## Take a gradient step for the mini-batch
		   WC = WC + alpha * (gpWC/float(NB) - gnWC/float(C) - Lambda * WC)
		   WB = WB + alpha * (gpWB/float(NB) - gnWB/float(C) - Lambda * WB)
		   WP = WP + alpha * (gpWP/float(NB) - gnWP/float(C) - Lambda * WP)
   return (WP, WB, WC)

   		
if __name__ == '__main__':
     global D, K
     D = 784 ; K = 100 ;
     print 'hi'
     RBMLearn()
     #writeImage(S, xsamples)
