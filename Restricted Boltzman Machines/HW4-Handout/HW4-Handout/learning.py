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

''' sample x[d]  give h '''
def sample_x(d, h): 
	prob = logit_fn(WC[d] + dot(WP[d], h))
	alpha = np.random.uniform()
	return 1 if prob > alpha else 0

''' sample h[k] given x'''
def sample_h(k, x): 
	prob = logit_fn(WB[k] + dot(WP[:, k], x))
	alpha = np.random.uniform()
	return 1 if prob > alpha else 0

''' Block Gibbs Sampling '''
def block_gibbs_sampler(S):
	h = zeros([S+1, K])
	x = zeros([S+1, D])
	for k in range(K): h[0][k] = random.randint(0,1)
	print h.shape, x.shape
	print h[0]
	for s in range(1, S+1):
		for d in range(D):
			x[s][d] = sample_x(d, h[s-1])
		for k in range(K):
			h[s][k] = sample_h(k, x[s])
                #print s
	return (x, h)

def save_img(x, name):
	        if type(name) == type(2):
			name = str(name)
	        Y = x.reshape(28, 28)
	        img = Image.fromarray((Y*255).astype(np.uint8))
		if not os.path.isdir('output'):
			os.mkdir('output')
	        f = 'output/' + name + ".png" 
	        img.save(f)
	        

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
