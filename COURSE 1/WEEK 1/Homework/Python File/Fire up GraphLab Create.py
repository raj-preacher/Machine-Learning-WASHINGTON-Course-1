
# coding: utf-8

# # Fire up GraphLab Create

# In[1]:

import graphlab


# # Load a Tabular DataSet

# In[3]:

sf=graphlab.Sframe('C:\Users\ROHIT\Desktop\MACHINE LEARNING\COURSE 1\WEEK 1\DATA\Week 1\people-example.csv')


# In[4]:

sf=graphlab.Sframe('C:\Users\ROHIT\Desktop\MACHINE LEARNING\COURSE 1\WEEK 1\DATA\Week 1\people-example.csv')


# In[5]:

sf=graphlab.SFrame('C:\Users\ROHIT\Desktop\MACHINE LEARNING\COURSE 1\WEEK 1\DATA\Week 1\people-example.csv')


# # SFrame Basics

# In[6]:

sf #we can view first few lines of the Table


# In[7]:

sf.head #to view first few lines


# In[8]:

sf.tail #first few lines from bottom


# # GraphLab Canvas

# 

# In[9]:

#Take any Data Structure in GraphLab Create
sf.show()


# In[10]:

graphlab.canvas.set_target('ipynb') #To Visualize Canvas in the Notebook


# In[11]:

sf['age'].show(view='Categorical')


# # Inspect columns of DataSet

# In[12]:

sf['Country']


# In[13]:

sf['age']


# In[14]:

sf['age'].mean()


# In[15]:

sf['age'].max()


# # Create new columns in SFrame

# In[16]:

sf


# In[19]:

sf['Full Name']=sf['First Name'] + ' '+sf['Last Name']  #New column of Full Name is generated


# In[18]:

sf


# In[20]:

sf['age']+2


# # Use the apply Function to do a Advance Transformation of our Data

# In[21]:

sf['Country'].show()


# In[22]:

def transform_country(country):
    if country == 'USA':
        return 'United States'
    else:
        return country


# In[23]:

transform_country('Brazil')


# In[24]:

transform_country('USA')


# In[25]:

sf['Country'].apply(transform_country)  # Change the 'USA' to 'United states'  row by row using apply function


# In[26]:

sf['Country']=sf['Country'].apply(transform_country)  # fix the change that applied above row by row


# In[27]:

sf


# In[ ]:



