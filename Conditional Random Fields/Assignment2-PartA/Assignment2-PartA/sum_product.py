import sys
from dicts import DefaultDict
from numpy import *
import math
import itertools

labels = {'e':0,'t':1,'a':2,'i':3,'n':4,'o':5,'s':6,'h':7,'r':8,'d':9}
rev_labels = {0:'e', 1:'t', 2:'a', 3:'i', 4:'n', 5:'o', 6:'s', 7:'h', 8:'r', 9:'d'}

size = 321
n = 10

WCC = zeros( (n, n) )
WCF = zeros( (n, size) ) 
clique = None
fmsg = None
bmsg = None
belief = None
pairwise_marginal = None
singlevar_marginal = None 
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
		print "potentials for clique #", i
		print "\te", "\tt", "\tr"
		for c1 in ['e', 't', 'r']:
		   print c1, 
		   for c2 in ['e', 't', 'r']:
		         print clique[i][ labels[c1] ][ labels[c2] ],
		   print
		print

### Question - 2 : log space messages ###
def logspace_messages():
	''' msg(3,2) = sum over 4 '''
	global bmsg
	global fmsg
	l = len(word)
	nclique = l-1
	nmsg = nclique-1
	bmsg = zeros( (nmsg, n) )
	fmsg = zeros( (nmsg, n) )
        ## backward pass
	for i in range(n):
	    bmsg[0][i]  = logsumexp( clique[nclique-1,i,:] )
	for i in range(1, nclique-1):
		 for j in range(n):
		     bmsg[i][j] = logsumexp(clique[nclique-1-i, j , :] + bmsg[i-1])
	
        ## forward pass 
	for i in range(n):
		fmsg[0][i] = logsumexp( clique[0, :, i] )
	for i in range(1,nclique-1):
		for j in range(n):
		    fmsg[i][j] = logsumexp( clique[i, :, j] + fmsg[i-1] )
	## print messages
	print "LOG SPACE MESSAGES"
	print "forward messages" 
	for i in range(nmsg):
		print i+1, "->", i+2, fmsg[i]
	print "backward messages"
	for i in range(nmsg):
		print nclique - i, "->", nclique-i-1, bmsg[i]

### Question - 3 : compute the log beliefs ###
def logspace_beliefs():
	global belief 
	l = len(word)
	nclique = l-1
	nmsg = nclique - 1
	belief = zeros( (nclique, n, n) )
        ## compute the beliefs
	belief[0] = clique[0] + matrix(bmsg[nmsg-1])
	for i in range(1, nclique-1):
		belief[i] = clique[i] + matrix(fmsg[i-1]).T + matrix(bmsg[nclique-i-2])
	belief[nclique-1] = clique[nclique-1] + matrix(fmsg[nclique-2]).T

	### print the beliefs
	for i in range(nclique):
	  print "belief #: ", i 
	  print "\te", "\tt"
	  for y1 in ['e', 't']:
	       print y1,
	       for y2 in ['e', 't']:
	         print belief[i][labels[y1]][labels[y2]],
	       print

## Question - 5 ###
def marginals():
      global pairwise_marginal
      global singlevar_marginal
      l = len(word)
      nclique = l-1
      pairwise_marginal = zeros( (nclique, n, n) )
      singlevar_marginal = zeros( (l, n) )
      for i in range(nclique):
	     total = 0
             for c1 in range(n):
                for c2 in range(n):
                         total += math.exp(belief[i][c1][c2])
	     print "clique # ", i, total
	     for c1 in range(n):
		     for c2 in range(n):
			     pairwise_marginal[i][c1][c2] = math.exp(belief[i][c1][c2])/total
	     for c1 in ['e', 't', 'r']:
		 for c2 in ['e', 't', 'r']:
			print pairwise_marginal[i][ labels[c1]][ labels[c2] ], 
	         print
	     print
	     for c1 in range(n):
		     singlevar_marginal[i][c1] = sum(pairwise_marginal[i,c1,:])
		     singlevar_marginal[i+1][c1] = sum(pairwise_marginal[i, :, c1])
      for i in range(l):
	      print singlevar_marginal[i]

             			 
def predict(word):
	best_seq = []
	l = len(word)
	for i in range(l):
		idx = argmax(singlevar_marginal[i])
		best_seq.append( rev_labels[idx] )
	best_word = ''.join(best_seq)
	return best_word

def logZ():
	Z = float(0.0)
	for c1 in range(n):
		for c2 in range(n):
			Z += math.exp( belief[1][c1][c2] )
	return math.log(Z)

def energy(word):
	l = len(word)
	nclique = l-1
	energy = float(0.0)
	for i in range(nclique):
		y1 = labels[word[i]]
		y2 = labels[word[i+1]]
		energy += clique[i][y1][y2]
	return energy

def log_likelihood(word):
         return energy(word) - logZ()

if __name__ == "__main__":

     load_feature_params('model/feature-params.txt')
     load_transition_params('model/transition-params.txt')
     cnt = 1
     correct = 0
     total = 0
     for line in open('data/test_words.txt', 'r'):
        f = "data/test_img" + str(cnt) + ".txt"
        word = line.strip()

     	### Sum Product Algo ###
    	create_clique_tree(f, word)
     	logspace_messages()
     	logspace_beliefs()
     	marginals()

        ### Question - 5 ###
     	pred_word = predict(word)
	for i in range(len(word)):
		if word[i] == pred_word[i]:
		    correct += 1
		total += 1
	cnt += 1
     print float(correct)/float(total)
     i = 1
     total_log_likelihood = 0
     for line in open('data/train_words.txt', 'r'):
	     f = "data/train_img" + str(i) + ".txt"
	     word = line.strip()
    	     
	     create_clique_tree(f, word)
     	     logspace_messages()
     	     logspace_beliefs()
     	     marginals()
	     
	     total_log_likelihood += log_likelihood(word)
	     i += 1
	     if i == 51 : 
	        break
     print total_log_likelihood/50.00
	 
     


