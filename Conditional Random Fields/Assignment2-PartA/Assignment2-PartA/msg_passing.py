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
fmsg = None
bmsg = None
belief = None

''' logsumexp trick '''
def logsumexp(lst):
	c = max(lst)
	lst = map(lambda x : math.exp(x-c), lst)
	return c + math.log( sum(lst) )

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
		if (i == l-2):
			clique[i] = matrix(node_potential[i]).T + matrix(node_potential[i+1]) + WCC
		else :
			clique[i] = matrix(node_potential[i]).T +  WCC

	# report the clique potentials
	for i in range(l-1):
		print "clique ", i
		for c1 in ['e', 't', 'r']:
		   for c2 in ['e', 't', 'r']:
		         print c1, c2, clique[i][ labels[c1] ][ labels[c2] ]

### Question - 2 : log space messages ###
def logspace_sumproduct(f, word):
	''' msg(3,2) = sum over 4 '''
	global bmsg
	global fmsg
	l = len(word)
	nclique = l-1
	nmsg = nclique-1
	bmsg = zeros( (nmsg, n) )
	fmsg = zeros( (nmsg, n) )
        ## backward pass
	k = nmsg-1
	for i in range(n):
	    bmsg[k][i]  = logsumexp( clique[nclique-1,i,:] )
	print bmsg[k]
	k -= 1
	for i in range(1, nclique-1):
		 for j in range(n):
		     bmsg[k][j] = logsumexp(clique[nclique-1-i, j , :] + bmsg[k+1])
		 print bmsg[k]
		 k -= 1
	
        ## forward pass 
	for i in range(n):
		fmsg[0][i] = logsumexp( clique[0, :, i] )
	print fmsg[0]
	for i in range(1,nclique-1):
		for j in range(n):
		    fmsg[i][j] = logsumexp( clique[i, :, j] + fmsg[i-1] )
		print fmsg[i]

### Question - 3 : compute the log beliefs ###
def bp(f, word):
	global belief 
	l = len(word)
	nclique = l-1
	belief = zeros( (nclique, n, n) )
        ## compute the beliefs
	belief[0] = clique[0] + matrix(bmsg[0])
	for i in range(1, nclique-1):
		belief[i] = clique[i] + matrix(fmsg[i-1]).T + matrix(bmsg[i])
	belief[nclique-1] = clique[nclique-1] + matrix(fmsg[nclique-2]).T

	for i in range(nclique):
	  print "clique #: ", i 
	  for y1 in ['e', 't']:
	       for y2 in ['e', 't']:
	         print belief[i][labels[y1]][labels[y2]],
	       print

## Question - 5 ###
def marginals(f, word):
      l = len(word)
      nclique = l-1
      for i in range(nclique):
	     total = 0
             for c1 in range(n):
                for c2 in range(n):
                         total += math.exp(belief[i][c1][c2])
	     for c1 in ['e', 't', 'r']:
		 for c2 in ['e', 't', 'r']:
			print math.exp(belief[i][labels[c1]][ labels[c2] ])/total,
	         print
	     print 
             			 


if __name__ == "__main__":

     load_feature_params('model/feature-params.txt')
     load_transition_params('model/transition-params.txt')
     ### Question - 1 ### 
     create_clique_tree('data/test_img1.txt', 'tree')
     ### Question - 2 ###
     logspace_sumproduct('data/test_img1.txt', 'tree')
     ### Question - 3 ###
     bp('data/test_img1.txt', 'tree')
     ### Question - 4 ####
     marginals('data/test_img1.txt', 'tree')


