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


''' Question -1 ''' 
def node_potential_conditional(f):
	node_potential = []
	for line in open(f, 'r'):
 	        potential = zeros(n)
		features = map(lambda x : int(x), line.strip().split(' '))
		for c in range(n):
			potential[c] = dot(WCF[c], features)
		node_potential.append( potential )
	return node_potential	 


''' Question - 2 '''
def negative_energy(f, word):
	energy = 0
	node_potential = node_potential_conditional(f)
	for i in range(len(word)-1):
		j = labels[ word[i] ]
		k = labels[ word[i+1] ]
		energy += node_potential[i][j] + WCC[j][k]
	j = labels[ word[-1] ]
	energy += node_potential[len(word)-1][j]
	return energy

''' Question - 3 '''
def log_partition(f, word):
	partition = float(0.0)
	chars = []
	for i in range(len(word)):
		chars.append( labels.keys() )
	for word_comb in itertools.product(*chars):
		word = ''.join(word_comb)
		energy = negative_energy(f, word)
		partition += math.exp(energy)
	print math.log(partition)




if __name__ == "__main__":

     load_feature_params('model/feature-params.txt')
     load_transition_params('model/transition-params.txt')
     ### question-1 ###
     print 'node potential'
     print node_potential_conditional('data/test_img1.txt')
     ### question-2 ####
     print 'energy'
     print negative_energy('data/test_img1.txt', 'tree')
     print negative_energy('data/test_img2.txt', 'net')
     print negative_energy('data/test_img3.txt', 'trend')
     ### question-3 ####
     print 'log paritition'
     print log_partition('data/test_img1.txt', 'tree')
     print log_partition('data/test_img2.txt', 'net')
     print log_partition('data/test_img3.txt', 'trend')


