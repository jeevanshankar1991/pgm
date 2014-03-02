import sys
from dicts import DefaultDict
from numpy import *
import math
import itertools

labels = {'e':0,'t':1,'a':2,'i':3,'n':4,'o':5,'s':6,'h':7,'r':8,'d':9}

size = 321
n = 10
WCC = zeros( (n, n) )
WCF = zeros( (n, size) ) 
clique = None
log_msg = None

''' logsumexp trick '''
def logsumexp(lst):
	c = max(lst)
	lst = map(lambda x : math.exp(x-c), lst)
	return c + sum(lst)

def load_feature_params(f):
        i = 0
	for line in open(f, 'r'):
		wgts = map(lambda x : float(x), line.strip().split(' '))
		WCF[i] = wgts
		i += 1

def load_transition_params(f):
	i = 0
	for line in open(f, 'r'):
		wgts = map(lambda x : float(x), line.strip().split(' '))
		WCC[i] = wgts
		i += 1

def get_node_potential(f):
	i = 0
	node_potential = []
	for line in open(f, 'r'):
		features = map(lambda x : float(x), line.strip().split(' '))
		potential = zeros(n)
		for c in range(n):
			potential[c] = dot(WCF[c], features)
		node_potential.append( potential )
	return node_potential
	
### Question - 1 ####
def create_clique_tree(f, word):
	global clique
	l = len(word)
	node_potential = get_node_potential(f)
	clique = zeros( (l-1, n, n) )
	for i in range(l-1):
		clique[i] = node_potential[i] * WCC
		if (i == l-1):
			clique[i] *= node_potential[i+1]
	# report the clique potentials
	for i in range(l-1):
		print clique[i]

### Question - 2 ###
def logspace_sumproduct(f, word):
	''' msg(3,2) = sum over 4 '''
	global log_msg
	log_clique = log(clique) ## converts very variable to log 
	l = len(word)
	log_msg = zeros( (l-1, l-1, n) )  # for eg, msg[3,2], msg[2,1], msg[1,2], msg[2,3] 
	
        ## backward pass
	log_msg[l-2][l-3] = logsumexp( sum(log_clique[i], axis=1) )
	for i in range(l-3, 0, -1):
		 log_msg[i][i-1] = logsumexp(log_clique[i] + log_msg[i+1][i])
	## forward pass
	msg[0][1] = logsumexp( sum(log_clique[0], axis = 0) )
	for i in range(1, l-2):
		log_msg[i][i-1] = logsumexp(log_clique[i] + log_msg[i+1][i])
	print log_msg

if __name__ == "__main__":

     load_feature_params('model/feature-params.txt')
     load_transition_params('model/transition-params.txt')
     ### Question - 1 ### 
     create_clique_tree('data/test_img1.txt', 'tree')
     ### Question - 2 ###
     logspace_sumproduct('data/test_img1.txt', 'tree')
     ### Question - 3 ###


