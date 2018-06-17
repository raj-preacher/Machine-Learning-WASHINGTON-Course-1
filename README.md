# Machine-Learning-WASHINGTON-Course-1
## COURSERA -  Machine Learning Foundations: A Case Study Approach (by University of Washington)

## Machine Learning Foundations: A Case Study Approach
## University of Washington
About this Course
Do you have data and wonder what it can tell you?  Do you need a deeper understanding of the core ways in which machine learning can improve your business?  Do you want to be able to converse with specialists about anything from regression and classification to deep learning and recommender systems?

In this course, you will get hands-on experience with machine learning from a series of practical case-studies.  At the end of the first course you will have studied how to predict house prices based on house-level features, analyze sentiment from user reviews, retrieve documents of interest, recommend products, and search for images.  Through hands-on practice with these use cases, you will be able to apply machine learning methods in a wide range of domains.

This first course treats the machine learning method as a black box.  Using this abstraction, you will focus on understanding tasks of interest, matching these tasks to machine learning tools, and assessing the quality of the output. In subsequent courses, you will delve into the components of this black box by examining models and algorithms.  Together, these pieces form the machine learning pipeline, which you will use in developing intelligent applications.

### Learning Outcomes:  By the end of this course, you will be able to:
   -Identify potential applications of machine learning in practice.  
   -Describe the core differences in analyses enabled by regression, classification, and clustering.
   -Select the appropriate machine learning task for a potential application.  
   -Apply regression, classification, clustering, retrieval, recommender systems, and deep learning.
   -Represent your data as features to serve as input to machine learning models. 
   -Assess the model quality in terms of relevant error metrics for each task.
   -Utilize a dataset to fit a model to analyze new data.
   -Build an end-to-end application that uses machine learning at its core.  
   -Implement these techniques in Python.
   
   # ASSIGNMENTS
   # WEEK-2
                                                                   
                
## Predicting house prices assignment
## Predicting house prices
In this module, we focused on using regression to predict a continuous value (house prices) from features of the house (square feet of living space, number of bedrooms,...). We also built an iPython notebook for predicting house prices, using data from King County, USA, the region where the city of Seattle is located.

In this assignment, we are going to build a more accurate regression model for predicting house prices by including more features of the house. In the process, we will also become more familiar with how the Python language can be used for data exploration, data transformations and machine learning. These techniques will be key to building intelligent applications.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

Learning outcomes
Execute programs with the iPython notebookLoad and transform real, tabular dataCompute summaries and statistics of the dataBuild a regression model using features of the data

What to do for this assignment:
Watch the video and explore the iPython notebook on predicting house prices
If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebook on predicting house prices from this module. You can then open up the iPython notebook and familiarize yourself with the steps we covered in this example.

Next steps
Now you are ready! Open the Predicting House Prices notebook, located in the Week 2 workspace to follow along.

We are going to do three tasks in this assignment. There are 3 results you need to gather along the way to enter into the quiz after this reading.

1. Selection and summary statistics: In the notebook we covered in the module, we discovered which neighborhood (zip code) of Seattle had the highest average house sale price. Now, take the sales data, select only the houses with this zip code, and compute the average price. Save this result to answer the quiz at the end.

2. Filtering data: One of the key features we used in our model was the number of square feet of living space (‘sqft_living’) in the house. For this part, we are going to use the idea of filtering (selecting) data.

In particular, we are going to use logical filters to select rows of an SFrame. You can find more info in the Logical Filter section of this documentation. Using such filters, first select the houses that have ‘sqft_living’ higher than 2000 sqft but no larger than 4000 sqft.What fraction of the all houses have ‘sqft_living’ in this range? Save this result to answer the quiz at the end.

3. Building a regression model with several more features: In the sample notebook, we built two regression models to predict house prices, one using just ‘sqft_living’ and the other one using a few more features, we called this set



Now, going back to the original dataset, you will build a model using the following features:



Note that using copy and paste from this webpage to the IPython Notebook sometimes does not work perfectly in some operating systems, especially on Windows. For example, the quotes defining strings may not paste correctly. Please check carefully is you use copy & paste.

Compute the RMSE (root mean squared error) on the test_data for the model using just my_features, and for the one using advanced_features.

Note 1: both models must be trained on the original sales dataset, not the filtered one.

Note 2: when doing the train-test split, make sure you use seed=0, so you get the same training and test sets, and thus results, as we do.

Note 3: in the module we discussed residual sum of squares (RSS) as an error metric for regression, but GraphLab Create uses root mean squared error (RMSE). These are two common measures of error regression, and RMSE is simply the square root of the mean RSS:


where N is the number of data points. RMSE can be more intuitive than RSS, since its units are the same as that of the target column in the data, in our case the unit is dollars ($), and doesn't grow with the number of data points, like the RSS does.

(Important note: when answering the question below using GraphLab Create, when you call the linear_regression.create() function, make sure you use the parameter validation_set=None, as done in the notebook you download above. When you use regression GraphLab Create, it sets aside a small random subset of the data to validate some parameters. This process can cause fluctuations in the final RMSE, so we will avoid it to make sure everyone gets the same answer.)

What is the difference in RMSE between the model trained with my_features and the one trained with advanced_features? Save this result to answer the quiz at the end.


Note: If you would rather use other ML tools...
You are welcome to use any ML tool for this course, such as scikit-learn. Though, as discussed in the intro module, we strongly recommend you use iPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in open-source under a permissive BSD license. So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use Pandas or other options out there: home_data.csv

Show offline instructions



# Week-3                                                     
                         
 ## Analyzing product sentiment assignment
## Analyzing product sentiment
In this module, we focused on classifiers, applying them to analyzing product sentiment, and understanding the types of errors a classifier makes. We also built an exciting iPython notebook for analyzing the sentiment of real product reviews.

In this assignment, we are going to explore this application further, training a sentiment analysis model using a set of key polarizing words, verify the weights learned to each of these words, and compare the results of this simpler classifier with those of the one using all of the words. These techniques will be a core component in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

Learning outcomes
Execute sentiment analysis code with the IPython notebookLoad and transform real, text dataUsing the .apply() function to create new columns (features) for our modelCompare results of two models, one using all words and the other using a subset of the wordsCompare learned models with majority class predictionExamine the predictions of a sentiment model Build a sentiment analysis model using a classifier

Open the Amazon Product Sentiment notebook in Week 3 to get started!

Note: If you would rather use other ML tools...
You are welcome to use any ML tool for this course, such as scikit-learn. Though, as discussed in the intro module, we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in open-source under a permissive BSD license. So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use Pandas or other options out there: amazon_baby.csv

Watch the video and explore the IPython notebook on analyzing sentiment
If you haven’t done so yet, before you start, we recommend you watch the video where we go over the IPython notebook on analyzing product sentiment using classifiers from this module. You can then open up the IPython notebook we used and familiarize yourself with the steps we covered in this example.

What you will do
Now you are ready! We are going do four tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

In the IPython notebook above, we used the word counts for all words in the reviews to train the sentiment classifier model. Now, we are going to follow a similar path, but only use this subset of the words:



Often, ML practitioners will throw out words they consider “unimportant” before training their model. This procedure can often be helpful in terms of accuracy. Here, we are going to throw out all words except for the very few above. Using so few words in our model will hurt our accuracy, but help us interpret what our classifier is doing.

Use .apply() to build a new feature with the counts for each of the selected_words: In the notebook above, we created a column ‘word_count’ with the word counts for each review. Our first task is to create a new column in the products SFrame with the counts for each selected_word above, and, in the process, we will see how the method .apply() can be used to create new columns in our data (our features) and how to use a Python function, which is an extremely useful concept to grasp!

Our first goal is to create a column products[‘awesome’] where each row contains the number of times the word ‘awesome’ showed up in the review for the corresponding product, and 0 if the review didn’t show up. One way to do this is to look at the each row ‘word_count’ column and follow this logic:

If ‘awesome’ shows up in the word counts for a particular product (row of the products SFrame), then we know how often ‘awesome’ appeared in the review, if ‘awesome’ doesn’t appear in the word counts, then it didn’t appear in the review, and we should set the count for ‘awesome’ to 0 in this review.

We could use a for loop to iterate this logic for each row of the products SFrame, but this approach would be really slow, because the SFrame is not optimized for this being accessed with a for loop. Instead, we will use the .apply() method to iterate the the logic above for each row of the products[‘word_count’] column (which, since it’s a single column, has type SArray). Read about using the .apply() method on an SArray here.

We are now ready to create our new columns:

First, you will use a Python function to define the logic above. You will write a function called awesome_count which takes in the word counts and returns the number of times ‘awesome’ appears in the reviews.

A few tips:

i. Each entry of the ‘word_count’ column is of Python type dictionary.

ii. If you have a dictionary called dict, you can access a field in the dictionary using:



but only if ‘awesome’ is one of the fields in the dictionary, otherwise you will get a nasty error.

iii. In Python, to test if a dictionary has a particular field, you can simply write:



In our case, if this condition doesn’t hold, the count of ‘awesome’ should be 0.

Using these tips, you can now write the awesome_count function.

Next, you will use .apply() to iterate awesome_count for each row of products[‘word_count’] and create a new column called ‘awesome’ with the resulting counts. Here is what that looks like:



And you are done! Check the products SFrame and you should see the new column you just create.

Repeat this process for the other 11 words in selected_words. (Here, we described a simple procedure to obtain the counts for each selected_word. There are other more efficient ways of doing this, and we encourage you to explore this further.)Using the .sum() method on each of the new columns you created, answer the following questions: Out of the selected_words, which one is most used in the dataset? Which one is least used? Save these results to answer the quiz at the end.

2. Create a new sentiment analysis model using only the selected_words as features: In the IPython Notebook above, we used word counts for all words as features for our sentiment classifier. Now, you are just going to use the selected_words:

Use the same train/test split as in the IPython Notebook from lecture:



Train a logistic regression classifier (use graphlab.logistic_classifier.create) using just the selected_words. Hint: you can use this parameter in the .create() call to specify the features used to be exactly the new columns you just created:



Call your new model: selected_words_model.

You will now examine the weights the learned classifier assigned to each of the 11 words in selected_words and gain intuition as to what the ML algorithm did for your data using these features. In GraphLab Create, a learned model, such as the selected_words_model, has a field 'coefficients', which lets you look at the learned coefficients. You can access it by using:



The result has a column called ‘value’, which contains the weight learned for each feature.

Using this approach, sort the learned coefficients according to the ‘value’ column using .sort(). Out of the 11 words in selected_words, which one got the most positive weight? Which one got the most negative weight? Do these values make sense for you? Save these results to answer the quiz at the end.

3. Comparing the accuracy of different sentiment analysis model: Using the method



What is the accuracy of the selected_words_model on the test_data? What was the accuracy of the sentiment_model that we learned using all the word counts in the IPython Notebook above from the lectures? What is the accuracy majority class classifier on this task? How do you compare the different learned models with the baseline approach where we are just predicting the majority class? Save these results to answer the quiz at the end.

Hint: we discussed the majority class classifier in lecture, which simply predicts that every data point is from the most common class. This is baseline is something we definitely want to beat with models we learn from data.

4. Interpreting the difference in performance between the models: To understand why the model with all word counts performs better than the one with only the selected_words, we will now examine the reviews for a particular product.

We will investigate a product named ‘Baby Trend Diaper Champ’. (This is a trash can for soiled baby diapers, which keeps the smell contained.)Just like we did for the reviews for the giraffe toy in the IPython Notebook in the lecture video, before we start our analysis you should select all reviews where the product name is ‘Baby Trend Diaper Champ’. Let’s call this table diaper_champ_reviews.Again, just as in the video, use the sentiment_model to predict the sentiment of each review in diaper_champ_reviews and sort the results according to their ‘predicted_sentiment’.What is the ‘predicted_sentiment’ for the most positive review for ‘Baby Trend Diaper Champ’ according to the sentiment_model from the IPython Notebook from lecture? Save this result to answer the quiz at the end.Now use the selected_words_model you learned using just the selected_words to predict the sentiment most positive review you found above. Hint: if you sorted the diaper_champ_reviews in descending order (from most positive to most negative), this command will be helpful to make the prediction you need:



Save this result to answer the quiz at the end.

Why is the predicted_sentiment for the most positive review found using the model with all word counts (sentiment_model) much more positive than the one using only the selected_words (selected_words_model)? Hint: examine the text of this review, the extracted word counts for all words, and the word counts for each of the selected_words, and you will see what each model used to make its prediction. Save this result to answer the quiz at the end.                       


 #  WEEK-4                                                       
                                                         
## Retrieving Wikipedia articles assignment
## Retrieving Wikipedia articles
In this module, we focused on using nearest neighbors and clustering to retrieve documents that interest users, by analyzing their text. We explored two document representations: word counts and TF-IDF. We also built an iPython notebook for retrieving articles from Wikipedia about famous people.

In this assignment, we are going to dig deeper into this application, explore the retrieval results for various famous people, and familiarize ourselves with the code needed to build a retrieval system. These techniques will be key to building the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

Learning outcomes
Execute document retrieval code with the iPython notebookLoad and transform real, text dataCompare results with word counts and TF-IDFSet the distance function in the retrievalBuild a document retrieval model using nearest neighbor search

Open the Document Retrieval notebook in Week 4 to get started!

Note: If you would rather use other ML tools...
You are welcome to use any ML tool for this course, such as scikit-learn. Though, as discussed in the intro module, we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in open-source under a permissive BSD license. So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use Pandas or other options out there: people_wiki.csv

Watch the video and explore the iPython notebook on retrieving wikipedia articles
If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebook on retrieving documents from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in this example.

What you will do
Now you are ready! We are going do three tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

Compare top words according to word counts to TF-IDF: In the notebook we covered in the module, we explored two document representations: word counts and TF-IDF. Now, take a particular famous person, 'Elton John'. What are the 3 words in his articles with highest word counts? What are the 3 words in his articles with highest TF-IDF? These results illustrate why TF-IDF is useful for finding important words. Save these results to answer the quiz at the end.Measuring distance: Elton John is a famous singer; let’s compute the distance between his article and those of two other famous singers. In this assignment, you will use the cosine distance, which one measure of similarity between vectors, similar to the one discussed in the lectures. You can compute this distance using the graphlab.distances.cosine function. What’s the cosine distance between the articles on ‘Elton John’ and ‘Victoria Beckham’? What’s the cosine distance between the articles on ‘Elton John’ and Paul McCartney’? Which one of the two is closest to Elton John? Does this result make sense to you? Save these results to answer the quiz at the end.Building nearest neighbors models with different input features and setting the distance metric: In the sample notebook, we built a nearest neighbors model for retrieving articles using TF-IDF as features and using the default setting in the construction of the nearest neighbors model. Now, you will build two nearest neighbors models:Using word counts as featuresUsing TF-IDF as features

In both of these models, we are going to set the distance function to cosine similarity. Here is how: when you call the function



add the parameter:



Now we are ready to use our model to retrieve documents. Use these two models to collect the following results:

What’s the most similar article, other than itself, to the one on ‘Elton John’ using word count features?What’s the most similar article, other than itself, to the one on ‘Elton John’ using TF-IDF features?What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using word count features?What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using TF-IDF features?

Save these results to answer the quiz at the end.                                                        
            , 
            
  #  WEEK-5          
            
                                                         
                                                          
##    Recommending songs assignment
## Recommending songs
In this module, we focused on building recommender systems to find products, music and movies that interest users. We also built an exciting iPython notebook for recommending songs, which compared the simple popularity-based recommendation with a personalized model, and showed the significant improvement provided by personalization.

In this assignment, we are going to explore the song data and the recommendations made by our model. In the process, you are going to learn how to use one of the most important data manipulation primitives, groupby. These techniques will be important to building the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

Learning outcomes
Execute song recommendation code with the iPython notebookLoad and transform real, song dataBuild a song recommender modelUse the model to recommend songs to individual usersUse groupby to compute aggregate statistics of the data

Open the Song Recommender notebook in Week 5 to get started!

Note: If you would rather use other ML tools...
You are welcome to use any ML tool for this course, such as scikit-learn. Though, as discussed in the intro module, we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in open-source under a permissive BSD license. So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use Pandas or other options out there: song_data.csv

Watch the video and explore the iPython notebook on recommending songs
If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebook on song recommendation from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in this example.

What you will do
Now you are ready! We are going do three tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

Counting unique users: The method .unique() can be used to select the unique elements in a column of data. In this question, you will compute the number of unique users who have listened to songs by various artists. For example, to find out the number of unique users who listened to songs by 'Kanye West', all you need to do is select the rows of the song data where the artist is 'Kanye West', and then count the number of unique entries in the ‘user_id’ column. Compute the number of unique users for each of these artists: 'Kanye West', 'Foo Fighters', 'Taylor Swift' and 'Lady GaGa'. Save these results to answer the quiz at the end. Using groupby-aggregate to find the most popular and least popular artist: each row of song_data contains the number of times a user listened to particular song by a particular artist. If we would like to know how many times any song by 'Kanye West' was listened to, we need to select all the rows where ‘artist’=='Kanye West' and sum the ‘listen_count’ column. If we would like to find the most popular artist, we would need to follow this procedure for each artist, which would be very slow. Instead, you will learn about a very important method:



You can read the documentation about groupby here. The .groupby method computes an aggregate (in our case, the sum of the ‘listen_count’) for each distinct value in a column (in our case, the ‘artist’ column).

Follow these steps to find the most popular artist in the dataset:

The .groupby method has two important parameters:

i. key_columns, which takes the column we want to group, in our case, ‘artist’

ii. operations, where we define the aggregation operation we using, in our case, we want to sum over the ‘listen_count’.

With this in mind, the following command will compute the sum listen_count for each artist and return an SFrame with the results:



the total number of listens for each artist will be stored in ‘total_count’.

Sort the resulting SFrame according to the ‘total_count’, and find the artist with the most popular and least popular artist in the dataset. Save these results to answer the quiz at the end.

3. Using groupby-aggregate to find the most recommended songs: Now that we learned how to use .groupby() to compute aggregates for each value in a column, let’s use to find the song that is most recommended by the personalized_model model we learned in the iPython notebook above. Follow these steps to find the most recommended song:

Split the data into 80% training, 20% testing, using seed=0, as was done in the iPython notebook above.Train an item_similarity_recommender, as done in the iPython notebook, using the training data.Next, we are going to make recommendations for the users in the test data, but there are over 200,000 users (58,628 unique users) in the test set. Computing recommendations for these many users can be slow in some computers. Thus, we will use only the first 10,000 users only in this question. Using this command to select this subset of users:



Let’s compute one recommended song for each of these test users. Use this command to compute these recommendations:



Finally, we can use .groupby() to find the most recommended song! :) When we used .groupby() in the previous question, we summed up the total ‘listen_count’ for each artist, by setting the parameter SUM in the aggregator:



For this question, we simply want to count how often each song is recommended, so we will use the COUNT aggregator instead of SUM, and store the results in a column we will call ‘count’ by using:



And, since we want to use the song titles as the key to the aggregator instead of of the ‘artist’, we use:



By sorting the results, you will find out the most recommended song to the first 10,000 users in the test data! Save these results to answer the quiz at the end.


   # WEEK-6                                                                    
                                                                       
##  Deep features for image classification & retrieval assignment
## Deep features for image classification & retrieval
In this module, we focused on using deep learning to create non-linear features to improve the performance of machine learning. We also saw how transfer learning techniques can be applied to use deep features learned with one dataset to get great performance on a different dataset. We also built an iPython notebooks for both image retrieval and image classification tasks on real datasets.

In this assignment, we are going to build new image retrieval models and explore their results on different parts of our image dataset. These techniques will be used at the core of the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

Learning outcomes
Execute image retrieval code with the iPython notebookUse the .sketch_summary() method to view statistics of dataLoad and transform real, image dataBuild image retrieval models using nearest neighbor search and deep featuresCompare the results of various image retrieval modelsUse the .apply() and .sum() methods on SFrames to compute functions of the data.

Open the following notebooks in the Week 6 folder to get started:

Deep Features for Image ClassificationDeep Features for Image Retrieval

Note: If you would rather use other ML tools...
You are welcome to use any ML tool for this course, such as scikit-learn. Though, as discussed in the intro module, we strongly recommend you use IPython Notebook and GraphLab Create. (GraphLab Create is free for academic purposes.)

If you are choosing to use other packages, we still recommend you use SFrame, which will allow you to scale to much larger datasets than Pandas. (Though, it's possible to use Pandas in this course, if your machine has sufficient memory.) The SFrame package is available in open-source under a permissive BSD license. So, you will always be able to use SFrames for free.

If you are not using SFrame, here is the dataset for this assignment in CSV format, so you can use Pandas or other options out there: image_train_data.csv and image_test_data.csv

Watch the videos and explore the iPython notebooks on using deep features for image classification and retrieval
If you haven’t done so yet, before you start, we recommend you watch the video where we go over the iPython notebooks from this module. You can then open up the iPython notebook we used and familiarize yourself with the steps we covered in these examples.

What you will do
Now you are ready! We are going do four tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

Computing summary statistics of the data: Sketch summaries are techniques for computing summary statistics of data very quickly. In GraphLab Create, SFrames and SArrays include a method:



which computes such summary statistics. Using the training data, compute the sketch summary of the ‘label’ column and interpret the results. What’s the least common category in the training data? Save this result to answer the quiz at the end.

2. Creating category-specific image retrieval models: In most retrieval tasks, the data we have is unlabeled, thus we call these unsupervised learning problems. However, we have labels in this image dataset, and will use these to create one model for each of the 4 image categories, {‘dog’,’cat’,’automobile’,bird’}. To start, follow these steps:

Split the SFrame with the training data into 4 different SFrames. Each of these will contain data for 1 of the 4 categories above. Hint: if you use a logical filter to select the rows where the ‘label’ column equals ‘dog’, you can create an SFrame with only the data for images labeled ‘dog’.Similarly to the image retrieval notebook you downloaded, you are going to create a nearest neighbor model using the 'deep_features' as the features, but this time create one such model for each category, using the corresponding subset of the training_data. You can call the model with the ‘dog’ data the dog_model, the one with the ‘cat’ data the cat_model, as so on.

You now have a nearest neighbors model that can find the nearest ‘dog’ to any image you give it, the dog_model; one that can find the nearest ‘cat’, the cat_model; and so on.

Using these models, answer the following questions. The cat image below is the first in the test data:


You can access this image, similarly to what we did in the iPython notebooks above, with this command:



What is the nearest ‘cat’ labeled image in the training data to the cat image above (the first image in the test data)? Save this result.

Hint: When you query your nearest neighbors model, it will return a SFrame that looks something like this:

query_label	reference_label	distance	rank
0	34	42.9886641167	1
0	45	43.8444904098	2
0	251	44.2634660468	3
0	141	44.377719559	4
To understand each column in this table, see this documentation. For this question, the ‘reference_label’ column will be important, since it provides the index of the nearest neighbors in the dataset used to train it. (In this case, the subset of the training data labeled ‘cat’.)

What is the nearest ‘dog’ labeled image in the training data to the cat image above (the first image in the test data)? Save this result.

3. A simple example of nearest-neighbors classification: When we queried a nearest neighbors model, the ‘distance’ column in the table above shows the computed distance between the input and each of the retrieved neighbors. In this question, you will use these distances to perform a classification task, using the idea of a nearest-neighbors classifier.

For the first image in the test data (image_test[0:1]), which we used above, compute the mean distance between this image at its 5 nearest neighbors that were labeled ‘cat’ in the training data (similarly to what you did in the previous question). Save this result.Similarly, for the first image in the test data (image_test[0:1]), which we used above, compute the mean distance between this image at its 5 nearest neighbors that were labeled ‘dog’ in the training data (similarly to what you did in the previous question). Save this result.On average, is the first image in the test data closer to its 5 nearest neighbors in the ‘cat’ data or in the ‘dog’ data? (In a later course, we will see that this is an example of what is called a k-nearest neighbors classifier, where we use the label of neighboring points to predict the label of a test point.)

4. [Challenging Question] Computing nearest neighbors accuracy using SFrame operations: A nearest neighbor classifier predicts the label of a point as the most common label of its nearest neighbors. In this question, we will measure the accuracy of a 1-nearest-neighbor classifier, i.e., predict the output as the label of the nearest neighbor in the training data. Although there are simpler ways of computing this result, we will go step-by-step here to introduce you to more concepts in nearest neighbors and SFrames, which will be useful later in this Specialization.

Training models: For this question, you will need the nearest neighbors models you learned above on the training data, i.e., the dog_model, cat_model, automobile_model and bird_model.Spliting test data by label: Above, you split the train data SFrame into one SFrame for images labeled ‘dog’, another for those labeled ‘cat’, etc. Now, do the same for the test data. You can call the resulting SFrames



Finding nearest neighbors in the training set for each part of the test set: Thus far, we have queried, e.g.,



our nearest neighbors models with a single image as the input, but you can actually query with a whole set of data, and it will find the nearest neighbors for each data point. Note that the input index will be stored in the ‘query_label’ column of the output SFrame.

Using this knowledge find the closest neighbor in to the dog test data using each of the trained models, e.g.,



finds 1 neighbor (that’s what k=1 does) to the dog test images (image_test_dog) in the cat portion of the training data (used to train the cat_model).

Now, do this for every combination of the labels in the training and test data.

Create an SFrame with the distances from ‘dog’ test examples to the respective nearest neighbors in each class in the training data: The ‘distance’ column in dog_cat_neighbors above contains the distance between each ‘dog’ image in the test set and its nearest ‘cat’ image in the training set. The question we want to answer is how many of the test set ‘dog’ images are closer to a ‘dog’ in the training set than to a ‘cat’, ‘automobile’ or ‘bird’. So, next we will create an SFrame containing just these distances per data point. The goal is to create an SFrame called dog_distances with 4 columns:

i. dog_distances[‘dog-dog’] ---- storing dog_dog_neighbors[‘distance’]

ii. dog_distances[‘dog-cat’] ---- storing dog_cat_neighbors[‘distance’]

iii. dog_distances[‘dog-automobile’] ---- storing dog_automobile_neighbors[‘distance’]

iv. dog_distances[‘dog-bird’] ---- storing dog_bird_neighbors[‘distance’]

Hint: You can create a new SFrame from the columns of other SFrames by creating a dictionary with the new columns, as shown in this example:



The resulting SFrame will look something like this:

dog-automobile	dog-bird	dog-cat	dog-dog
41.9579761457	41.7538647304	36.4196077068	33.4773590373
46.0021331807	41.3382958925	38.8353268874	32.8458495684
42.9462290692	38.6157590853	36.9763410854	35.0397073189
Computing the number of correct predictions using 1-nearest neighbors for the dog class: Now that you have created the SFrame dog_distances, you will learn to use the method



on this SFrame to iterate line by line and compute the number of ‘dog’ test examples where the distance to the nearest ‘dog’ was lower than that to the other classes. You will do this in three steps:

i. Consider one row of the SFrame dog_distances. Let’s call this variable row. You can access each distance by calling, for example,



which, in example table above, will have value equal to 36.4196077068 for the first row.

Create a function starting with



which returns 1 if the value for row[‘dog-dog’] is lower than that of the other columns, and 0 otherwise. That is, returns 1 if this row is correctly classified by 1-nearest neighbors, and 0 otherwise.

ii. Using the function is_dog_correct(row), you can check if 1 row is correctly classified. Now, you want to count how many rows are correctly classified. You could do a for loop iterating through each row and applying the function is_dog_correct(row). This method will be really slow, because the SFrame is not optimized for this type of operation.

Instead, we will use the .apply() method to iterate the function is_dog_correct for each row of the SFrame. Read about using the .apply() method here.

iii. Computing the number of correct predictions for ‘dog’: You can now call:



which will return an SArray (a column of data) with a 1 for every correct row and a 0 for every incorrect one. You can call:



on the result to get the total number of correctly classified ‘dog’ images in the test set!

Hint: To make sure your code is working correctly, if you were to do the two steps above in this question to count the number of correctly classified ‘cat’ images in the test data, instead of ‘dog’, the result would be 548.

Accuracy of predicting dog in the test data: Using the work you did in this question, what is the accuracy of the 1-nearest neighbor classifier at classifying ‘dog’ images from the test set? Save this result to answer the quiz at th end.                                                                  
