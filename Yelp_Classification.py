#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load necessary libraries, etc.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


# Set Path and Load Dataset
path = "/Users/ryantomiyama/Desktop/Data/yelp.csv"
yelp = pd.read_csv(path)
yelp.info()


# In[3]:


# Add new column giving the length of text for each review
yelp['text_length'] = yelp['text'].apply(len)


# Exploratory Data Analysis

# In[4]:


# Histograms of text_length for given star ratings
star_text = sns.FacetGrid(data=yelp, col='stars')
star_text.map(plt.hist, 'text_length', bins=30)


# From the above histograms there is more frequency of four and five star reviews.

# In[5]:


# Correlation matrix between several of the features
star = yelp.groupby('stars').mean()
sns.heatmap(data=star.corr(), annot=True)


# From the correlation matrix, useful and funny are strongly correlated as well as useful and text_length. As you can imagine, typically the longer the text it will often lead to a more helpful and thoughtful review.

# # The goal is to predict if reviews are good or bad based off of the text in the review.

# Consider reviews with one or two stars as bad and a reviews of four or five stars are good.

# In[6]:


yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==2) | (yelp['stars']==4) | (yelp['stars']==5)]


# In[7]:


yelp_class.shape
# 8,539 reviews are used out of the 10,000.


# In[8]:


# Creating the X and y for classification.
X = yelp_class['text']
y = yelp_class['stars']


# Now the text pre-processing.

# In[9]:


X[0]
# As you can see changing the words of text into a feature vector will be the task.


# In[10]:


# Use the following for tokenization.
def preprocess(text):
    nopunctuation = [char for char in text if char not in string.punctuation]
    
    nopunctuation = ''.join(nopunctuation)
    
    return [word for word in nopunctuation.split() if word.lower() not in stopwords.words('english')]


# In[13]:


sample = "Hello, how is the weather today? It's going to rain!"
print(preprocess(sample))


# Now convert to a vector. With the use of scikit-learn and more specifically CountVectorizer we can convert the text to a matrix with numbers. The columns of the matrix are the separate reviews and the rows are unique words.

# In[14]:


bag_of_words_transform = CountVectorizer(analyzer=preprocess).fit(X)
# The function CountVectorizer turns the words into a matrix.


# In[15]:


# All possible words from the reviews.
len(bag_of_words_transform.vocabulary_)


# In[16]:


review = X[3]
review


# In[17]:


bag_of_words_review = bag_of_words_transform.transform([review])
print(bag_of_words_review)


# In[18]:


print(bag_of_words_transform.get_feature_names()[9549])


# In[19]:


# Make new X.
X = bag_of_words_transform.transform(X)


# In[20]:


X.shape
# 8,539 reviews and 40,526 possible words.


# In[21]:


# Splitting the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[22]:


# Using a naive bayes model.
NB = MultinomialNB()
NB.fit(X_train, y_train)


# In[23]:


pred = NB.predict(X_test)


# In[24]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[25]:


# Negative/Bad Review
review_1 = yelp_class['text'][65]
review_1


# In[26]:


review_1_transformed = bag_of_words_transform.transform([review_1])


# In[27]:


NB.predict(review_1_transformed)[0]
# From the model, a one star prediction is made.


# In[28]:


# Positive/Good Review
review_2 = yelp_class['text'][22]
review_2


# In[29]:


review_2_transformed = bag_of_words_transform.transform([review_2])


# In[30]:


NB.predict(review_2_transformed)[0]
# From the model, a five star prediction is made.


# In[31]:


# Negative/Bad Review but predicts poorly.
review_3 = yelp_class['text'][140]
review_3


# In[32]:


review_3_transformed = bag_of_words_transform.transform([review_3])


# In[33]:


NB.predict(review_3_transformed)[0]
# From the model, a four star prediction is made.


# In[34]:


yelp_class['stars'][140]


# This particular reviewer gave one star, but the model predicted four stars as seen above. The model may have predicted incorrectly because of the style of writing and word choice in the review. You can see the text includes words like 'great', 'happy', and 'inexpensive' which could be convincing enough to predict a high star review. This review also has a format in which a positive is followed by a negative or vice versa. It may make it harder than predicting a strictly positive or negative review throughout.

# Perhaps the model could improve on prediction abilities with more data. In particular data that includes more negative/bad reviews resulting in one or two stars. This dataset mostly has four or five stars so it could be stronger in predicting positive/good reviews.
