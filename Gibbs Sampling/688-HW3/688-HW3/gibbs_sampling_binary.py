import sys
from numpy import *
import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt

H = 100
W = 100
n = 100
Y = zeros([n, n])
X = zeros([n, n])
I = zeros([n, n])
T = 0
WL = 0
WP = 0
mae = []

def load_image(f):
#return  np.loadtxt(f, dtype='float32' )
      return [map(lambda(x) : int(float(x)), line.strip().split(' ')) for line in open(f, 'r')]

def get_prob(i, j, k):
	energy = [0, 0]
	for s in range(2):
	  energy[s] += (X[i][j] == s) * WL
	  energy[s] += ((Y[i-1][j] == s) * WP if i-1 >=0 else 0 ) + ((Y[i+1][j] == s) * WP if i+1 < n else 0)
	  energy[s] += ((Y[i][j-1] == s) * WP if j-1 >=0 else 0 ) + ((Y[i][j+1] == s) * WP if j+1 < n else 0)
        energy = map(lambda x : math.exp(x), energy)
	return float(energy[k])/float(sum(energy))

def run_gibbs_sampling():
	global Y
	global mae
	run_sum_y= zeros([H, W])
	mae = []
	for t in range(T):
		error = 0
		for i in range(n):
			for j in range(n):
				prob = get_prob(i, j, 1)
				alpha = np.random.uniform()
				Y[i][j] = 1 if alpha < prob else 0
				run_sum_y[i][j] += Y[i][j] ## stores the previous history. just the sum 
				error += np.absolute(run_sum_y[i][j]/float(t+1) - I[i][j]) ## abs the error 
		mae.append(0)
		mae[t] = error/float(H*W)
#if t != 0 and np.absolute(mae[t] - mae[t-1]) <= 1e-6 :
#		     break
                print "Error at", t, error, error/float(H * W)

def save_image():
	img = Image.fromarray((Y*255).astype(np.uint8))
	img.show()
	f = str(T) + "_" + str(WP) + "_" + str(WL) + "_stripes_img.png"
	img.save(f)

def plot(y):
	x = range(len(y))
	plt.plot(x, y, '-bo')
	plt.axis([0, len(y) + 10, 0.01, 0.05])
	plt.xlabel('Iterations')
	plt.ylabel('MAE')
	plt.title('MAE vs Iterations')
	wl = sys.argv[3]
	wp = sys.argv[2]
	f = str(T) + "_" + str(wp) + "_" + str(wl) + "_stripes_graph.png"
	plt.savefig(f)
	plt.show()

if __name__ == '__main__':
    global WP
    global WL
    global T 
    global X
    global Y
    global I
    I = array(load_image('data/stripes.txt'))
    X = array(load_image('data/stripes-noise.txt'))
    Y = copy(X) ## initialization
    print I.shape, X.shape, Y.shape
    T = int(sys.argv[1])
    WP = int(sys.argv[2])
    WL = int(sys.argv[3])
    run_gibbs_sampling()
    save_image()
    plot(mae)
