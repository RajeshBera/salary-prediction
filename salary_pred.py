
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


# In[2]:


df = pd.read_csv('Salary_Data.csv')


# In[3]:


reg = linear_model.LinearRegression()
reg.fit(df[['YearsExperience']],df.Salary)


# In[4]:


plt.scatter(df.YearsExperience,df.Salary,color='red')
plt.plot(df[['YearsExperience']], reg.predict(df[['YearsExperience']]),color='green')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary predection with experience")


# In[5]:


reg.predict([[1]])

