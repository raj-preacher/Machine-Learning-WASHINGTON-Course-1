
# coding: utf-8

# In[20]:

import graphlab


# # Read some Product Review Data

# In[21]:

products=graphlab.SFrame("amazon_baby.gl")


# # Let's explore this data together
# 

# In[3]:

products.head()


# # Build the word count vector for each Reviews

# In[4]:

products['word_count']=graphlab.text_analytics.count_words(products['review'])


# In[5]:

products.head(
)


# In[6]:

graphlab.canvas.set_target('ipynb')


# In[7]:

products['name'].show()


# In[15]:

products['rating'].show()


# In[8]:

graphlab.canvas.set_target('ipynb')


# In[9]:

products.show()


# # Explore Vullie Sophie

# In[10]:

giraffe_reviews=products[products['name']=='Vulli Sophie the Giraffe Teether']


# In[11]:

len(giraffe_reviews)


# In[12]:


giraffe_reviews['rating'].show(view='Categorical')


# In[24]:

len(giraffe_reviews)


# # Build a sentiment Classifier

# In[13]:

products['rating'].show(view='Categorical')


# ## Define whats a positive or Negative Sentiment

# In[14]:

#ignore all 3* Reviews
products=products[products['rating']!=3]


# In[15]:

#positive sentiment =4* or 5* Reviews
products['sentiment']=products['rating']>=4


# In[29]:

products.head()


# ## Let's Train the Sentiment Classifier

# In[16]:

train_data,test_data=products.random_split(.8,seed=0)


# In[17]:

sentiment_model=graphlab.logistic_classifier.create(train_data,target='sentiment',
                                                   features=['word_count'],
                                                   validation_set=test_data)


# # Evaluate the Sentiment Model

# In[18]:

sentiment_model.evaluate(test_data,metric='roc_curve')


# In[19]:

sentiment_model.show(view='Evaluation')


# # Apply the learned model to understand the sentiment of Giraffe

# In[22]:

giraffe_reviews['predicted_sentiment']=sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[23]:

giraffe_reviews.head()


# # Sort the Review based on the predicted sentiment and Explore

# In[25]:

giraffe_reviews=giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[26]:

giraffe_reviews.head()


# In[27]:

giraffe_reviews[0]['review']


# ## Show most negative ReviewS

# In[28]:

giraffe_reviews[-1]['review']


# In[ ]:



