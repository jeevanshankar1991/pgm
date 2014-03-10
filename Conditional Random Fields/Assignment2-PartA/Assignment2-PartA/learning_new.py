import sys
from dicts import DefaultDict
from numpy import *
import math
import itertools
import scipy
import scipy.optimize
labels = {'e':0,'t':1,'a':2,'i':3,'n':4,'o':5,'s':6,'h':7,'r':8,'d':9}
rev_labels = {0:'e', 1:'t', 2:'a', 3:'i', 4:'n', 5:'o', 6:'s', 7:'h', 8:'r', 9:'d'}

size = 321
n = 10

WCC = zeros( (n, n) )
WCF = zeros( (n, size) ) 
l = 0
report = False
node_potential = []
features = []
clique = None
fmsg = None
bmsg = None
belief = None
pairwise_marginal = None
singlevar_marginal = None 
calls = 0

def logsumexp(lst):
	c = max(lst)
	lst = map(lambda x : math.exp(x-c), lst)
	return c + math.log( sum(lst) )

def load_feature_params(f):
	global WCF
        i = 0
	for line in open(f, 'r'):
		wgts = map(lambda x : float(x), line.strip().split(' '))
		WCF[i] = wgts
		i += 1

def load_transition_params(f):
	global WCC
	i = 0
	for line in open(f, 'r'):
		wgts = map(lambda x : float(x), line.strip().split(' '))
		WCC[i] = wgts
		i += 1


def calc_node_potential(f):
	global node_potential
	global features
	for line in open(f, 'r'):
		feature_vec = map(lambda x : float(x), line.strip().split(' '))
		potential = zeros(n)
		for c in range(n):
			potential[c] = dot(WCF[c], feature_vec)
		node_potential.append( potential )
	        features.append( feature_vec )
	
def create_clique_tree():
	global clique
	clique = zeros( (l-1, n, n) )
	for i in range(l-1):
		if (i == l-2):
			clique[i] = matrix(node_potential[i]).T + matrix(node_potential[i+1]) + WCC
		else :
			clique[i] = matrix(node_potential[i]).T +  WCC

	# report the clique potentials
	if report :
	   for i in range(l-1):
		print "clique potentials #", i
		print "\te", "\tt", "\tr"
		for c1 in ['e', 't', 'r']:
		   print c1, 
		   for c2 in ['e', 't', 'r']:
		         print clique[i][ labels[c1] ][ labels[c2] ],
		   print
		print

def pass_messages():
	global bmsg
	global fmsg
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

	## report messages
	if report  :
	  print '-'*75
	  print "log space messages\n"
          print "forward messages\n" 
	  for i in range(nmsg):
		print "delta", i+1, "->", i+2, fmsg[i]
	  print "\nbackward messages\n"
	  for i in range(nmsg):
		print "delta", nclique - i, "->", nclique-i-1, bmsg[i]

def calc_beliefs():
	global belief 
	nclique = l-1
	nmsg = nclique - 1
	belief = zeros( (nclique, n, n) )
        ## compute the beliefs
	belief[0] = clique[0] + matrix(bmsg[nmsg-1])
	for i in range(1, nclique-1):
		belief[i] = clique[i] + matrix(fmsg[i-1]).T + matrix(bmsg[nclique-i-2])
	belief[nclique-1] = clique[nclique-1] + matrix(fmsg[nclique-2]).T

	### print the beliefs
	if report :
	 print '-'*75
	 print "beliefs "  
	 for i in range(nclique):
	   print "Beta #: ", i 
	   print "\te", "\tt"
	   for y1 in ['e', 't']:
	       print y1,
	       for y2 in ['e', 't']:
	         print belief[i][labels[y1]][labels[y2]],
	       print 
	   print 

## Question - 5 ###
def calc_marginals():
      global pairwise_marginal
      global singlevar_marginal
      nclique = l-1
      pairwise_marginal = zeros( (nclique, n, n) )
      singlevar_marginal = zeros( (l, n) )
      Z = math.exp( logZ() )
      for i in range(nclique):
	     for c1 in range(n):
		     for c2 in range(n):
			     pairwise_marginal[i][c1][c2] = math.exp(belief[i][c1][c2])/Z
	     for c1 in range(n):
		     singlevar_marginal[i][c1] = sum(pairwise_marginal[i,c1,:])
		     singlevar_marginal[i+1][c1] = sum(pairwise_marginal[i, :, c1])
      if report :
	  print '-'*75
	  print "pair-wise marginals"
	  for i in range(nclique):
	     print 'Y' + str(i+1), 'Y' + str(i+2)
	     print "\te", "\t\tt\t", "\tr"
	     for y1 in ['e', 't', 'r']:
	         print y1,
		 for y2 in ['e', 't', 'r']:
			print pairwise_marginal[i][ labels[y1]][ labels[y2] ], 
	         print
	     print
          print "single var marginals : "
          for i in range(l):
	      print "Y" + str(i+1), ":", singlevar_marginal[i]

             			 
def predict():
	best_seq = []
	for i in range(l):
		idx = argmax(singlevar_marginal[i])
		best_seq.append( rev_labels[idx] )
	best_word = ''.join(best_seq)
	return best_word

def logZ():
	Z = float(0.0)
	lst1 = []
	for c1 in range(n):
		lst = []
		for c2 in range(n):
			lst.append( belief[1][c1][c2] )
		lst1.append( logsumexp(lst) )
	return logsumexp(lst1)

def energy(word):
	nclique = l-1
	energy = float(0.0)
	for i in range(nclique):
		y1 = labels[word[i]]
		y2 = labels[word[i+1]]
		energy += clique[i][y1][y2]
	return energy

def log_prob(word):
         return energy(word) - logZ()

def do_sumproduct(f, word, r = False):
	global l
	global report
	global node_potential
	global features
	global clique
	global fmsg
	global bmsg
	global belief
	global pairwise_marginal
	global singlevar_marginal

	l = len(word)
	report = r
	node_potential = []
	features = []
	clique = None
	fmsg = None
	bmsg = None
	belief = None
	pairwise_marginal = None
	singlevar_marginal = None 
	
	calc_node_potential(f) ## factor reduction
	create_clique_tree() ## calc clique potentials
	pass_messages()  ## pass messages backward and forward
	calc_beliefs()  ## calc beliefs 
	calc_marginals()  ## calc marginals . will used for learning weights 

	
def calc_objective(wgts):
     	global WCC
        global WCF
        WCC = wgts[0:n*n].reshape([n, n])
        WCF = wgts[n*n:].reshape([n, size])
	words = [ line.strip() for line in open('data/train_words.txt') ]
	
        ## initialize the objective
	obj = float(0.0)
	
	for itr in range(N):
		word = words[itr]
		f = 'data/train_img' + str(itr+1) + '.txt'
		do_sumproduct(f, word)
		obj += log_prob(word)
	avg_objective = -obj/float(N)
	return avg_objective

def calc_gradient(wgts):

     words = [ line.strip() for line in open('data/train_words.txt') ]
     ## initialize the counts and gradient
     global WCC
     global WCF
     WCC = wgts[0:n*n].reshape([n, n])
     WCF = wgts[n*n:].reshape([n, size])
     featgrad = zeros([n, size])
     transgrad = zeros([n, n])
     for itr in range(N):
	     word = words[itr]
	     f = 'data/train_img' + str(itr+1) + '.txt' 
             do_sumproduct(f, word)
	     ## do for transition counts
	     for j in range(l-1):  ### j is the word position and l is len(word)
		      c = labels[word[j]]
		      cprime = labels[word[j+1]]
		      transgrad[c][cprime] += 1
		      for c in range(n):
			      for cprime in range(n):
				      transgrad[c][cprime] -= pairwise_marginal[j][c][cprime] ## edge marginal computed as result of sum-product
	     ## do for emission counts
	     for j in range(len(word)):
		     c = labels[word[j]]
		     for f in range(size):
		          featgrad[c][f] += features[j][f]
		          for y in range(n):
				  featgrad[y][f] -= features[j][f] * singlevar_marginal[j][y] ## node marginal computed as result of sum-product

     featgrad = concatenate(featgrad, axis=1)
     transgrad = concatenate(transgrad,axis=1)
     print -concatenate([transgrad,featgrad],axis=1)/float(N)
     return -concatenate([transgrad,featgrad],axis=1)/float(N)
     
	



if __name__ == "__main__":
      global N 
      N = int(sys.argv[1])
      ini_wgts = zeros( [1, n*n + n*size] )
      print 'started'
      load_feature_params('model/feature-params.txt')
      load_transition_params('model/transition-params.txt')
      do_sumproduct('data/test_img1.txt', 'tree', True)
#gradient(ini_wgts)
#      objective(ini_wgts)
      result = scipy.optimize.fmin_bfgs(f = calc_objective, x0 = ini_wgts, fprime = calc_gradient)
      print result	
     
