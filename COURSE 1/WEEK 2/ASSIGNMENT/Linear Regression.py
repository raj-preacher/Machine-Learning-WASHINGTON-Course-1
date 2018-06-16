
# coding: utf-8

# # Fire up Graphlab Create

# In[1]:

import graphlab


# # Load some House sales data

# In[2]:

sales=graphlab.SFrame('C:\Users\ROHIT\Desktop\MACHINE LEARNING\COURSE 1\WEEK 2\DATA\Week 2\home_data.gl')


# In[3]:

sales


# # Exploring the data for Housing Sales

# In[7]:

graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living",y="price")


# 
# # Create a simple regression model of sqft_living to price

# In[8]:

train_data,test_data = sales.random_split(0.8,seed=0)


# ## Build the Regression Model

# In[10]:

sqft_model=graphlab.linear_regression.create(train_data,target='price', features=['sqft_living']) 
#target specify what we want to predict  and feature is the input


# # Evaluate the Simple Model

# In[11]:

print test_data['price'].mean()


# In[13]:

print sqft_model.evaluate(test_data)


# # Let's show what our predictions look like

# In[14]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[19]:

plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')   
#blue dots will be the test_data for houses and green line is the predicted house values


# In[23]:

sqft_model.get('coefficients')  #gives w


# # Explore other Features in the data

# In[25]:

my_features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# In[26]:

sales[my_features].show()


# In[29]:

sales.show(view='BoxWhisker Plot',x='zipcode',y='price')


# # Build a regression model with more features

# In[30]:

my_features_model=graphlab.linear_regression.create(train_data,target='price',features=my_features)


# In[31]:

print sqft_model.evaluate(test_data)
my_features_model.evaluate(test_data)


# # Apply learned model to predict price of 3 houses

# In[32]:

house1=sales[sales['id']=='5309101200']


# In[33]:

house1


# <img src="house.jpg">

# <img src="C:\Users\ROHIT\Desktop\MACHINE LEARNING\COURSE 1\WEEK 2\DATA\Week 2\house.jpg">

# In[36]:

print house1['price']


# In[37]:

print sqft_model.predict(house1)


# In[38]:

print my_features_model.predict(house1)


# In[39]:

house2=sales[sales['id']=='1925069082']


# In[40]:

house2


# In[41]:

print sqft_model.predict(house2)


# In[42]:

print my_features_model.predict(house2)


# # last house, Super Fancy

# In[43]:

bill_gates={'bedrooms':[8],
            'bathrooms':[25],
            'sqft_living':[50000],
            'sqft_lot':[225000],
            'floors':[4],
            'zipcode':['98039'],
            'condition':[10],
            'grade':[10],
            'waterfront':[1],
            'view':[4],
            'sqft_above':[37500],
            'sqft_basement':[12500],
            'yr_built':[1994],
            'yr_renovated':[2010],
            'lat':[47.627606],
            'long':[-122.242054],
            'sqft_living15':[5000],
            'sqft_lot15':[40000]}
    


# In[45]:

print my_features_model.predict(graphlab.SFrame(bill_gates))


# # ASSIGNMENT

# In[56]:

high= sales[sales['zipcode']=='98039']


# In[63]:

filter=sales[(sales['sqft_living']>=2000) & (sales['sqft_living']<=4000)]


# In[53]:

filter


# In[57]:

high['price'].mean()


# In[64]:

filter.num_rows()


# In[65]:

sales.num_rows()


# In[74]:

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# In[75]:

my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features)


# In[76]:

print my_features_model.evaluate(test_data)


# In[77]:

advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode','condition', 'grade','waterfront','view','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15'] 


# In[78]:

adv_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features)


# In[79]:

print adv_features_model.evaluate(test_data)


# In[ ]:



