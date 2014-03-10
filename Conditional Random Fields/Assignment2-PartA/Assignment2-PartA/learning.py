import sys
from dicts import DefaultDict
from numpy import *
import math
import itertools
from scipy.optimize import fmin_bfgs
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
	
#print 'node potential'
	calc_node_potential(f) ## factor reduction
#	print 'create clique'
	create_clique_tree() ## calc clique potentials
#	print 'pass msg'
	pass_messages()  ## pass messages backward and forward
#	print 'calc belief'
	calc_beliefs()  ## calc beliefs 
#	print 'calc marginals'
	calc_marginals()  ## calc marginals . will used for learning weights 

#### added f() and fprime() required for BGFS optimization method ####
def load_wgts(wgts):
	global WCC
	global WCF
	i = 0
	for c in range(n):
		for cprime in range(n):
			WCC[c][cprime] = wgts[i]
			i += 1
	for c in range(n):
		for f in range(size):
			WCF[c][f] = wgts[i]
			i += 1
	
def objective(wgts):
        ## load wgts and words
	load_wgts(wgts)
	words = [ line.strip() for line in open('data/train_words.txt') ]
	
        ## initialize the objective
	objective = float(0.0)
        global calls
        print calls
        calls += 1
	
	for itr in range(N):
		word = words[itr]
		f = 'data/train_img' + str(itr+1) + '.txt'
		do_sumproduct(f, word)
		objective += log_prob(word)
	avg_objective = -1 * objective/float(N)
	return avg_objective

def gradient(wgts):

     ## load wgts and words
     load_wgts(wgts)
     words = [ line.strip() for line in open('data/train_words.txt') ]
     ## initialize the counts and gradient
     trans_data_count = zeros( (n, n) )
     trans_expected_count = zeros( (n, n) )
     emis_data_count = zeros( (n, size) )
     emis_expected_count = zeros( (n, size ) )
     grad = zeros ( (n*n + n*size) ) ## first n*n parameters is transition parameters and n*size is emission parameters
     for itr in range(N):
	     word = words[itr]
	     f = 'data/train_img' + str(itr+1) + '.txt' 
             do_sumproduct(f, word)
	     ## do for transition counts
	     for j in range(len(word)-1):
		      c = labels[word[j]]
		      cprime = labels[word[j+1]]
		      trans_data_count[c][cprime] += 1
		      for c in range(n):
			      for cprime in range(n):
				      trans_expected_count[c][cprime] += pairwise_marginal[j][c][cprime] ## edge marginal computed as result of sum-product
	     ## do for emission counts
	     for j in range(len(word)):
		     c = labels[word[j]]
		     for f in range(size):
		          emis_data_count[c][f] += features[j][f]
		          for y in range(n):
				  emis_expected_count[y][f] += features[j][f] * singlevar_marginal[j][y] ## node marginal computed as result of sum-product
     i = 0
     for c in range(n):
	     for cprime in range(n):
		     grad[i] = trans_data_count[c][cprime] - trans_expected_count[c][cprime]
		     i += 1
     for c in range(n):
	     for f in range(size):
		     grad[i] = emis_data_count[c][f] - emis_expected_count[c][f]
		     i += 1
     grad = -1 * grad/float(N)
     return grad
	



if __name__ == "__main__":
      global N 
      N = int(sys.argv[1])
      ini_wgts = zeros( (n*n + n*size) )
      print 'started'
      result = fmin_bfgs(objective, ini_wgts, fprime = gradient, gtol = 1e-3, epsilon = 1.5, maxiter=100)
      print result	
     



