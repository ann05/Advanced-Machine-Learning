
import numpy as np
import scipy
import pandas
import pgmpy
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling


#--------------Reading the data file using pandas package-------------------------

data = pandas.read_csv("Bank_Defaults.csv")

#-------------Converting discrete variables' values to numeric codes--------------

data['default'] = data['default'].astype('category')
data['default'] = data['default'].cat.codes

data['marital'] = data['marital'].astype('category')
data['marital'] = data['marital'].cat.codes

data['education'] = data['education'].astype('category')
data['education'] = data['education'].cat.codes
data.default.unique()
data.marital.unique()
data.education.unique()

data['housing'] = data['housing'].astype('category')
data['housing'] = data['housing'].cat.codes
data.housing.unique()

data['loan'] = data['loan'].astype('category')
data['loan'] = data['loan'].cat.codes

data.contact.unique()
data['contact'] = data['contact'].astype('category')
data['contact'] = data['contact'].cat.codes
data.contact.unique()

data.month.unique()
data['month'] = data['month'].astype('category')
data['month'] = data['month'].cat.codes
data.month.unique()

data.poutcome.unique()
data['poutcome'] = data['poutcome'].astype('category')
data['poutcome'] = data['poutcome'].cat.codes
data.poutcome.unique()

data.y.unique()
data['y'] = data['y'].astype('category')
data['y'] = data['y'].cat.codes
data.y.unique()

data.groupby(["housing","loan"]).size()
data.previous.unique()
data['previous'] = data['previous'].astype('category')
data.previous.unique()
data.groupby(["education","loan"]).size()

#---------------Construct Markov Model based on intuition---------------------

mark = MarkovModel([('education','loan'),('education','housing'),('loan','y'),('housing','y'),('marital','y'),('default','y'),('contact','month'),('month','y'),('poutcome','y')])

#----------------Generate factors in the Markov model and add them------------
data.groupby(["education","loan"]).size()
f1 = DiscreteFactor(["education","loan"],cardinality=[4,2],values=(584,94,1890,416,1176,174,180,7))
data.groupby(["education","housing"]).size()
f2 = DiscreteFactor(["education","housing"],cardinality=[4,2],values=(295,383,876,1430,687,663,104,83))
data.groupby(["loan","y"]).size()
f3 = DiscreteFactor(["loan","y"],cardinality=[2,2],values=(3352,478,648,43))
data.groupby(["housing","y"]).size()
f4 = DiscreteFactor(["housing","y"],cardinality=[2,2],values=(1661,301,2339,220))
data.groupby(["marital","y"]).size()
f5 = DiscreteFactor(["marital","y"],cardinality=[3,2],values=(451,77,2520,277,1029,167))
data.groupby(["default","y"]).size()
f6 = DiscreteFactor(["default","y"],cardinality=[2,2],values=(3933,512,67,9))
data.groupby(["contact","previous"]).size()
data.groupby(["contact","month"]).size()
f7 = DiscreteFactor(["contact","month"],cardinality=[3,12],values=(276,607,18,199,129,576,73,38,529,345,64,42,17,20,2,23,18,102,8,10,49,38,11,3,0,6,0,0,1,28,450,1,820,6,5,7))
data.groupby(["month","y"]).size()
f8 = DiscreteFactor(["month","y"],cardinality=[12,2],values=(237,56,554,79,11,9,184,38,132,16,645,61,476,55,28,21,1305,93,350,39,43,37,35,17))
data.groupby(["poutcome","y"]).size()
f9 = DiscreteFactor(["poutcome","y"],cardinality=[4,2],values=(427,63,159,38,46,83,3368,337))
mark.add_factors(f1)
mark.add_factors(f2)
mark.add_factors(f3)
mark.add_factors(f3)
mark.add_factors(f4)
mark.add_factors(f5)
mark.add_factors(f6)
mark.add_factors(f7)
mark.add_factors(f8)
mark.add_factors(f9)


mark.get_local_independencies()

#------------------Calculate inference using Mplp algorithm--------------------

mplp = Mplp(mark)
mplp.find_triangles()
mplp.map_query()

#infer1 = BayesianModelSampling(mark)
#evidence1 = [State('y',1)]
#sample1 = infer1.forward_sample(5)
#sample1

#---------Calculate inference using Belief Propagation and Variable Elimination and answer corresponding queries-------------

belief_prop = BeliefPropagation(mark)
bp1 = belief_prop.query(variables=['y'],evidence={'marital' : 2,'default' : 0})
print(bp1['y'])

infer = VariableElimination(mark)
phi_query = infer.query(variables=['y'],evidence={'marital':0,'default':0})
print(bp1['y'])

bp2 = belief_prop.query(variables=['y'],evidence={'education' : 2,'housing' : 1})
print(bp2['y'])

infer = VariableElimination(mark)
phi_query = infer.query(variables=['y'],evidence={'education':2,'housing':1})
print(bp1['y'])

infer = VariableElimination(mark)
phi_query = infer.query(variables=['y'],evidence={'month':11,'poutcome':2})
print(bp1['y'])

bp3 = belief_prop.query(variables=['y'],evidence={'month' : 11,'poutcome' : 2})
print(bp3['y'])

bp4 = belief_prop.query(variables=['y'],evidence={'contact' : 2})
print(bp4['y'])

bp5 = belief_prop.query(variables=['y'],evidence={'poutcome' : 2, 'loan' : 1})
print(bp5['y'])

bp6 = belief_prop.query(variables=['y'],evidence={'marital' : 1, 'loan' : 1, 'contact' : 1, 'month' : 5})
print(bp6['y'])

#-----------------Sampling using GibbsSampling--------------------------

gibbs_chain = GibbsSampling(mark)
gen = gibbs_chain.generate_sample(size=5)
[sample for sample in gen]

gibbs_chain.sample(size=4)

for fact in mark.get_factors():
    print(fact)

data1 = data[['marital','education','default','housing','loan','contact','month','poutcome','y']].copy()

df = data1[0:5]

#------------------Calculate mean and entropy using the samples generated above---------------------------
np.mean(df)
scipy.stats.entropy(df)

arr = pandas.DataFrame.as_matrix(df)

for i in arr:
    for j in i:
        print i        

arr1 = arr.sum(axis=0)
s1 = arr.astype(float)

for j in range(0,9):
    for i in s1:
        #print i[0]
        #print arr1[1]
        i[j] = i[j]/arr1[j]

scipy.stats.entropy(s1)
