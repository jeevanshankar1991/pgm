import sys
from numpy import *
import numpy as np
import math
from PIL import Image
import random 
import png
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import os, os.path

global WP, WB, WC, D, K

''' Image helper functions '''
'''def dict_to_numpy(e):
	a = zeros( (28, 28) )
	for w in range(len(e)):
		j=w%28
		i=w/28
		a[i, j]=e[w]
		return a

def save_to_png(pixels, file_path):
		(height, width) = pixels.shape
		grid = (pixels * 255).astype(int).tolist()
		#print grid
		#print numpy.amin(grid)
		writer = png.Writer(width, height, greyscale=True)
		with open(file_path, 'wb') as png_file:
		    		writer.write(png_file, grid)

def writeImage(T, xsamples):
	t = 5
        fig = plt.figure()
        gs = gridspec.GridSpec(10,10)
	ax = [plt.subplot(gs[i]) for i in range(100)]
	i= 0
	while(t <= T):
           #print t, len(self.xsamples[t])
           #print i
           save_to_png(dict_to_numpy(xsamples[t]),"output/part1_"+str(t))
           ax[i].imshow(dict_to_numpy(xsamples[t]), cmap = cm.Greys_r)
	   i += 1	
	   t += 5
        plt.show()

def finalWrite(ax, start, xsamples):
	ax[start].imshow(dict_to_numpy(xsamples[500]), cmap = cm.Greys_r)
'''
''' Image helper close  '''


''' logit function '''
def logit_fn(x): 
        y = math.exp(x)
	return y / (1 + y)

''' load weights '''
def load_weights():
	global WP, WB, WC 
	WP = np.loadtxt('Models/MNISTWP.txt')
	WB = np.loadtxt('Models/MNISTWB.txt')
	WC = np.loadtxt('Models/MNISTWC.txt')
        print WP.shape, WC.shape, WB.shape

''' calculate energy '''
def energy(x, h):
	energy = dot(dot(WP, h), x)
	energy += dot(WB, h)
	energy += dot(WC, x)
	return energy

''' conditional prob(Hk = 1 | X) '''
def get_cond_prob_h(k, x):
	prob = logit_fn(WB[k] + dot(WP[:, k], x))
	return prob
''' conditional prob(Xd = 1 | H) '''
def get_cond_prob_x(d, h):
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

''' Block Gibbs Sampling '''
def block_gibbs_sampler(S):
	h = zeros([S+1, K])
	x = zeros([S+1, D])
	for k in range(K): h[0][k] = random.randint(0,1)
	for s in range(1, S+1):
		x[s] = sample_x(h[s-1])
		h[s] = sample_h(x[s])
                #print s
	return (x[1:,:], h[1:,:])

def save_img(x, name):
	        if type(name) == type(2):
			name = str(name)
	        Y = x.reshape(28, 28)
	        img = Image.fromarray((Y*255).astype(np.uint8))
		if not os.path.isdir('output'):
			os.mkdir('output')
	        f = 'output/' + name + ".png" 
	        img.save(f)
def RBMLearn():
   #Initialize gibbs chain
   for c in range(C):
       for k in range(K):
                h[c][k] = random.randint(0, 1)
   #Initialize the parameters with Gaussian Noise
   for d in range(D): WC[d] = np.random.gauss(0, 0.1)
   for k in range(K): WB[k] = np.random.gauss(0, 0.1)
   for k in range(K):
	   for d in range(D):
		   WP[d][k] = np.random.gauss(0, 0.1)
   ## Run the mini-batch SGD 
   for t in range(T):
	   for b in range(1,B+1):
                   gpWC = zeros(D); gpWB = zeros(K); gpWP = zeros([D, K]);
                   gnWC = zeros(D); gnWB = zeros(K); gnWP = zeros([D, K]);
                   ## compute the +ve gradient contribution using the data seen
		   for n in range((b-1)*NB, b*NB):
			  gpWC = gpWC + data_x[n]
			  cond_probs = map(lambda k : get_cond_prob_h(k, data_x[n]), range(K))
			  gpWB = gpWB + cond_probs
			  gpWP = gpWP + matrix(cond_probs).T * data_x[n]
		    
		    ## compute the -ve gradient contribution using data sampled using Block Gibbs Sampling
		    sampled_x, samplex_y = block_gibbs_sampler(C)
		    for c in range(C):
			   gpWC = gpWC + sampled_x[c]
			   cond_probs = map(lambda k : get_cond_prob_h(k, sampled_x[n]), range(K))
			   gpWB = gpWB + cond_probs
			   gpWP = gpWP + matrix(cond_probs).T * sampled_x[n]

	            ## Take a gradient step for the mini-batch
		    WC = WC + alpha * (gpWC/float(NB) - gnWC/float(C) - Lambda * WC)
		    WB = WB + alpha * (gpWB/float(NB) - gnWB/float(C) - Lambda * WB)
		    WP = WP + alpha * (gpWP/float(NB) - gnWP/float(C) - Lambda * WP)
   return (WP, WB, WC)

   		

if __name__ == '__main__':
     global D, K
     D = 784
     K = 100
     load_weights()
     print WP
     S = int(sys.argv[1])
     T = int(sys.argv[2])
     for t in range(T):
	    print t
            xsamples, hsamples = block_gibbs_sampler(S)
	    save_img(xsamples[-1], t)
     #writeImage(S, xsamples)
