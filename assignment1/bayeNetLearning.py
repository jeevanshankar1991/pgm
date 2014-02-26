import sys
import itertools
from math import ceil


""" vals range of each discrete nodes """
vals = {'A' : (1,2,3), 'G' : (1,2), 'CP' : (1,2,3,4), 'BP' : (1, 2),'CH' : (1,2), 'ECG' : (1,2),
	'HR' : (1,2), 'EIA' : (1,2), 'HD' : (1,2) }

""" order in which each line is to be interupted in the file """
order = ['A', 'G', 'CP', 'BP', 'CH', 'ECG', 'HR','EIA', 'HD']

""" Our graph is a DAG and dictionary is used to represent to represent this """
graph = {}

""" Our counts """
counts = {}

def createGraph():
	""" Write the adjency where nodes points to allits parent """
	global graph
	graph['A'] = []
	graph['G'] = []
	graph['CP'] = ['HD']
	graph['BP'] = ['G']
	graph['CH'] = ['A', 'G']
	graph['ECG'] = ['HD']
	graph['HR'] = ['HD', 'BP', 'A']
	graph['EIA'] = ['HD']
	graph['HD'] = ['CH', 'BP']

def initCount():
	global counts
	for node in graph.keys():
		counts[node] = {}
	for node, parents in graph.items():
		if len(parents) == 0 :
		    for nodeval in vals[node]:
		         counts[node][nodeval] = 0
		else :
		     parentVals = [ vals[parent] for parent in parents ] 
		     for product in itertools.product(*parentVals):
			 counts[node][product] = 0
			 nodevals = vals[node]
			 for nodeval in nodevals:
			     tmp = (nodeval, ) + product
			     counts[node][tmp] = 0
def updateCounts(instance):
	"""" instance is a dictionary . for eg, {'A' : 1, 'G' : 2, 'CP' : 3, ..... } """
	global counts
	for node, parents in graph.items():
		if len(parents) == 0 :
		      counts[node][ instance[node] ] += 1
		else :
		      parentval = tuple( [ instance[parent] for parent in parents ] )
		      nodeval = (instance[node], ) + parentval
		      counts[node][parentval] += 1
		      counts[node][nodeval] +=1

def getCPT(node):
	""" get the CPT of the node . Code figures its parents using graph data structure """
	parents = graph[node]
	if len(parents) == 0 :
	      total_cnt = 0
	      for nodeval in vals[node]:
	          total_cnt += counts[node][nodeval]
	      for nodeval in vals[node]:
	          print node, nodeval, counts[node][nodeval], total_cnt, float(counts[node][nodeval])/float(total_cnt)
	else :
		parentVals = [ vals[parent] for parent in graph[node] ]
		for product in itertools.product(*parentVals):
			countParent = counts[node][product]
			if countParent == 0 : 
			      continue
			for nodeval in vals[node]:
			      tmp = (nodeval, ) + product
			      print node, nodeval, product, counts[node][tmp], countParent, float(counts[node][tmp])/float(countParent)

def getJointProb(instance):
	""" get the joint probability given the instance """ 
	ans = 1
	for node, parents in graph.items():
		if len(parents) == 0 :
		    total_cnt = 0
		    for val in vals[node]:
		           total_cnt += counts[node][val]
		    ans *= float(counts[node][instance[node]])/float(total_cnt)
		else :
		    parentval = tuple( [instance[parent] for parent in parents] )
		    nodeval = (instance[node], ) + parentval
		    ans *= float(counts[node][nodeval])/float(counts[node][parentval])
	return ans

def learnBayesNet(filename):
	""" paramters for this graph are learned using MLE"""
	initCount()
	for line in open(filename, 'r'):
		 instance = {}
		 values = map(lambda x : int(x), line.strip().split(',') )
		 for i in range(len(values)):
			 instance[ order[i] ] = values[i]
		 updateCounts(instance)

def classify(instance):
	""" Classify the instance (patient) if he/she has heart-disease or not """
	instance['HD'] = 1
	prob1 = getJointProb(instance)
	instance['HD'] = 2
	prob2 = getJointProb(instance)
	if (prob1 >= prob2):
		return 1
	else :
	        return 2

def testBayesNet(filename):
	""" Test the bayesnet  """
	total = 0
	correct = 0
	for line in open(filename, 'r'):
		instance = {}
		values = map(lambda x : int(x), line.strip().split(',') )
		for i in range(len(values)):
			instance[ order[i] ] = int(values[i])
		if (classify(instance) == values[-1]) : ## heart-disease
			correct +=1
		total +=1
	print "Total : ", total, "# Correct : ", correct

def probQuery():
   ###  Question - 1
   '''prob = {}
   for ch in vals['CH']:
          prob[ch] = float(counts['CH'][(ch,2,2)])/float(counts['CH'][(2,2)]) * float(counts['HD'][(1,ch,1)])/float(counts['HD'][(ch,1)]) 	
   print prob[1]/(prob[1] + prob[2]) '''
   instance = {'A': 2, 'G':2, 'CH':1, 'CP':4, 'BP' : 1, 'ECG' : 1, 'HR' : 1, 'EIA' : 1, 'HD' : 1} 
   num = getJointProb(instance)
   dem = num
   instance['CH'] = 2
   dem += getJointProb(instance)
   print float(num)/float(dem)
   ### Question -  2
   instance = {'A' :2, 'G' : 1, 'CH':2, 'BP':1, 'ECG':1, 'HR':2, 'CP':1, 'EIA':2, 'HD':1}
   num = 0
   dem = 0
   for g in vals['G']:
        instance['G'] = g
        num += getJointProb(instance)
   for g in vals['G']:
         for bp in vals['BP']:
	     instance['G'] = g
	     instance['BP'] = bp
	     dem += getJointProb(instance)
   print float(num)/float(dem)


if __name__ == '__main__':
    createGraph()
    learnBayesNet(sys.argv[1])

    """ question - 4 """
#     print "Question 4 : CPT"
#    getCPT('A')
#    getCPT('BP')    
#    getCPT('HD')
#    getCPT('HR')

    """ question - 5 """
    print "Question-5 : Prob Queries "
    probQuery()

    """ question - 6 """
    print "Question-6 : Classfication "
    testBayesNet(sys.argv[2])
    
        


