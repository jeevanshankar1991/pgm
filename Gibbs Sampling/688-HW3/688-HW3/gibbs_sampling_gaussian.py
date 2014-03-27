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
I = zeros([n, n])
Y = zeros([n, n])
X = zeros([n, n])
mae = []
T = 0
WL = ones([n, n])
WPR = ones([n-1, n]) ## have n cols, but n-1 rows
WPC = ones([n, n-1]) ## have n rows, but n-1 colns

def load_image(f):
      return [map(lambda(x) : float(x), line.strip().split(' ')) for line in open(f, 'r')]

def get_mu_sigma(i, j):
	N = 0
	mu = 0
	sigma = 0
	# N (weight neighbour count )
	N = WL[i][j]
	N += WPR[i][j] if i+1 < n else 0
	N += WPC[i][j] if j+1 < n else 0
	N += WPR[i-1][j] if i-1 >=0 else 0
	N += WPC[i][j-1] if j-1 >=0 else 0
        # sigma
	sigma = 1 / float(2 * N)
        # mu
	mu = WL[i][j] * X[i][j]
	mu += WPR[i-1][j] * Y[i-1][j] if i-1 >=0 else 0
	mu += WPR[i][j] * Y[i+1][j] if i+1 < n else 0
	mu += WPC[i][j-1] * Y[i][j-1] if j-1 >=0 else 0
	mu += WPC[i][j] * Y[i][j+1] if j+1 < n else 0
	mu = mu / float(N)
	return (mu, sigma)

def run_gibbs_sampling():
	global Y
	global mae
	run_sum_y= zeros([H, W])
	for t in range(T):
		error = 0
		for i in range(n):
			for j in range(n):
				mu, sigma = get_mu_sigma(i, j)
				z = np.random.normal(0, 1)
				Y[i][j] =  mu + z * math.sqrt(sigma)
				run_sum_y[i][j] += Y[i][j] ## stores the previous history. just the sum 
				error += np.absolute(run_sum_y[i][j]/float(t+1) - I[i][j]) ## abs the error 
		mae.append(0)
		mae[t] = error/float(H*W)
		if t != 0 and np.absolute(mae[t] - mae[t-1]) <= 1e-6 :
		     break
                print "Error at", t, error, error/float(H * W)

def save_image():
	img = Image.fromarray((Y*255).astype(np.uint8))
	img.show()
	wl = sys.argv[3]
	wp = sys.argv[2]
	f = str(T) + "_" + str(wp) + "_" + str(wl) + "_swirl_img.png"
	img.save(f)

def plot(y):
	x = range(len(y))
	plt.plot(x, y, '-bo')
	plt.axis([0, len(y) + 10, 0.01, 0.04])
	plt.xlabel('Iterations')
	plt.ylabel('MAE')
	plt.title('MAE vs Iterations')
	wl = sys.argv[3]
	wp = sys.argv[2]
	f = str(T) + "_" + str(wp) + "_" + str(wl) + "_swirl_graph.png"
	plt.savefig(f)
	plt.show()

def make_wp_wc():
	global WPR
	global WPC
	for j in range(n):
		for i in range(n-1):
			WPR[i][j] = WPR[i][j]/float(0.01 + (X[i][j] - X[i+1][j])**2)
	for i in range(n):
		for j in range(n-1):
			WPC[i][j] = WPC[i][j]/float(0.01 + (X[i][j] - X[i][j+1])**2)

if __name__ == '__main__':
    global T 
    global X
    global Y
    global I
    I = array(load_image('data/swirl.txt'))
    X = array(load_image('data/swirl-noise.txt'))
    Y = copy(X) ## initialization
    print I.shape, X.shape, Y.shape
    T = int(sys.argv[1])
    WPR = float(sys.argv[2]) * WPR
    WPC = float(sys.argv[2]) * WPC
    WL = float(sys.argv[3]) * WL 
    make_wp_wc()	
    run_gibbs_sampling()
    save_image()
    plot(mae)
