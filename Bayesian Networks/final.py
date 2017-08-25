
# coding: utf-8

# In[1]:

import pgmpy
import pandas
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
import numpy as np
import scipy


Mental_health_model= BayesianModel([('Age','treatment'),('Gender','treatment'),('Country','treatment'),('family_history','treatment'),
                                    ('self_employed','treatment'),('care_options','treatment'),
                                    ('anonymity','treatment'),('treatment','leave'),('work_interfere','leave'),
                                    ('coworkers','obs_consequence'),
                                    ('obs_consequence','supervisor'),('mental_vs_physical','supervisor'),('obs_consequence','mental_health_consequence'),
                                    ('mental_health_consequence','mental_health_interview'),('anonymity','wellness_program'),
                                    ('no_employees','wellness_program'),('seek_help','wellness_program'),
                                    ('tech_company','wellness_program'),('tech_company','benefits'),('seek_help','anonymity'),('mental_health_consequence','leave'),('wellness_program','treatment')])



data = pandas.read_csv('C:\UB Spring Edu\AML\survey.csv')


train = data[1:600]

x = Mental_health_model.fit(train, estimator = MaximumLikelihoodEstimator)

#for cpd in Mental_health_model.get_cpds():
 #   print(cpd)


# In[2]:



# In[3]:

belief_prop = BeliefPropagation(Mental_health_model)


# In[5]:

bp1 = belief_prop.query(variables=['leave','wellness_program'],evidence={'tech_company' : 0})
print(bp1['leave'])
print(bp1['wellness_program'])


# In[6]:

bp2 = belief_prop.query(variables=['treatment'],evidence={'Age' : 1, 'Gender' : 0, 'family_history' : 1})
print(bp2['treatment'])


# In[7]:

bp3 = belief_prop.query(variables=['benefits','treatment'],evidence={'tech_company' : 1})
print(bp3['benefits'])
print(bp3['treatment'])


# In[8]:

bp4= belief_prop.query(variables=['leave'],evidence={'mental_health_consequence' : 1})
print(bp4['leave'])


# In[9]:

bp5 = belief_prop.query(variables=['mental_health_interview','supervisor'],evidence={'obs_consequence' : 0})
print(bp5['mental_health_interview'])
print(bp5['supervisor'])


# In[10]:

bp6 = belief_prop.query(variables=['leave'],evidence={'seek_help' : 0,'wellness_program' : 1,'mental_health_consequence' : 1})
print(bp6['leave'])


# In[11]:

bp7 = belief_prop.query(variables=['treatment','leave'],evidence={'seek_help' : 0,'care_options' : 0})
print(bp7['treatment'])
print(bp7['leave'])


# In[12]:

bp8 = belief_prop.query(variables=['treatment'],evidence={'anonymity' : 0})
print(bp8['treatment'])


# In[13]:

x = belief_prop.query(variables=['treatment'])
print(x['treatment'])


# In[14]:

x1 = belief_prop.query(variables=['leave'])
x2 = belief_prop.query(variables=['tech_company'])
x3 = belief_prop.query(variables=['wellness_program'])
x4 = belief_prop.query(variables=['treatment'])
x5 = belief_prop.query(variables=['Age'])
x6 = belief_prop.query(variables=['Gender'])
x7 = belief_prop.query(variables=['family_history'])
x8 = belief_prop.query(variables=['benefits'])
x9 = belief_prop.query(variables=['mental_health_consequence'])
x10 = belief_prop.query(variables=['mental_health_interview'])
x11 = belief_prop.query(variables=['obs_consequence'])
x12 = belief_prop.query(variables=['supervisor'])
x13 = belief_prop.query(variables=['seek_help'])
x14 = belief_prop.query(variables=['care_options'])
x15 = belief_prop.query(variables=['anonymity'])
print(x1['leave'])
print(x2['tech_company'])
print(x3['wellness_program'])
print(x4['treatment'])
print(x5['Age'])
print(x6['Gender'])
print(x7['family_history'])
print(x8['benefits'])
print(x9['mental_health_consequence'])
print(x10['mental_health_interview'])
print(x11['obs_consequence'])
print(x12['supervisor'])
print(x13['seek_help'])
print(x14['care_options'])
print(x15['anonymity'])


# In[15]:




# In[19]:

#infer1 = BayesianModelSampling(Mental_health_model)
#evidence2 = [State('treatment',1)]
#np.mean(infer1.likelihood_weighted_sample(evidence2,5))


# In[20]:



# In[30]:

infer1 = BayesianModelSampling(Mental_health_model)
evidence1 = [State('treatment',1)]
sample1 = infer1.forward_sample(5)
sample1


# In[31]:

m = np.mean(sample1)
print("Mean: ",m)

# In[32]:



# In[33]:

scipy.stats.entropy(sample1)


# In[71]:

arr = pandas.DataFrame.as_matrix(sample1)


# In[72]:


# In[73]:

arr1 = arr.sum(axis=0)


# In[74]:

# In[76]:

s1 = arr.astype(float)


# In[77]:

for j in range(0,20):
    for i in s1:
        #print i[0]
        #print arr1[1]
        i[j] = i[j]/arr1[j]


# In[78]:


# In[79]:
print("Entropy:")
print(scipy.stats.entropy(s1))


# In[ ]:



