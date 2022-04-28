#!/usr/bin/env python
# coding: utf-8

# # **IN Vehickle Coupon Recommendation**

# # **Importing the Libraries**
# This data was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver.

# In[1]:


# Importing pandas as pd
import pandas as pd
# Importing matplotlib as mt
import matplotlib as mt
# Importing LabelEncoder from sklearn
from sklearn.preprocessing import LabelEncoder
# Importing StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler
# Importing OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder
# Importing make_imbalance from imblearn
from imblearn.datasets import make_imbalance
# Importing matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Importing seaborn as sns
import seaborn as sns
# Importing counter from collection
from collections import Counter
# Importing matplotlib
import matplotlib


# # **Loading the dataset**

# In[2]:


# loading the dataset in csv from to data variable
data = pd.read_csv('vehicle-coupon.csv')
# to print the loaded dataset
data


# # **Pre-Processing and Cleaning the data**

# In[3]:


# Here we are dropping the two column by using drop function
data.drop(['car','toCoupon_GEQ5min'], axis=1, inplace=True)


# In[4]:


# to print first 5 rows of dataset we use head command
data.head()


# In[5]:


# the command to check the types of data is
data.dtypes


# In[6]:


# cleanup is used here to convert the datapytes
cleanup = {"gender": {"Female":0, "Male":1},
           "destination": {"No Urgent Place":0, "Home":1, "Work":2},
           "passanger": {"Alone":0, "Friend(s)":1, "Kid(s)":2, "Partner":3},
           "weather": {"Sunny":0, "Rainy":1, "Snowy":2},
           "time": {"2PM":0, "10AM":1, "10PM":2, "6PM":3, "7AM":4},
           "expiration": {"1d":0, "2h":1},
           "maritalStatus": {"Unmarried partner":0, "Single":1, "Divorced":2, "Widowed":3, "Married partner":4},
           "coupon": {"Restaurant(<20)":0, "Coffee House":1, "Carry out & Take away":2, "Restaurant(20-50)":3, "Bar":4},
           #"education": {"Some college - no degree":0, "Bachelors degree":1, "Associates degree":2, "High School Graduate":3, "Graduate degree (Masters or Doctorate)":4},
           "income": {"$37500 - $49999":0, "$62500 - $74999":1, "$12500 - $24999":2, "$75000 - $87499":3, "$50000 - $62499":4, "$25000 - $37499":5, "$100000 or More":6, "$87500 - $99999":7, "Less than $12500":8},
            "Bar": {"never": 0, "less1":1, "1~3":2, "gt8":3, "4~8":4},
           "CoffeeHouse": {"never":0, "less1":1, "4~8":2, "1~3":3, "gt8":4, "nan":5},
           "age": {"21":0, "46":1, "26":2, "31":3, "41":4, "50plus":5, "36":6, "below21":7},
           "CarryAway": {"4~8":0, "1~3":1, "gt8":2, "less1":3, "never":4},
           "RestaurantLessThan20": {"4~8":0, "1~3":1, "less1":2, "gt8":3, "never":4},
           "Restaurant20To50": {"4~8":0, "1~3":1, "less1":2, "gt8":3, "never":4, "nan":5}

          }
data.replace(cleanup, inplace=True)
# to print first 5 rows of dataset we use head command
data.head(20)


# In[7]:


# get_dummies is used as one-hot encode for categorical data
data=pd.get_dummies(data)
# to print first 5 rows of dataset we use head command
data.head()


# In[8]:


#filling the missinf values with mod
mod = data.mode()['CarryAway']
print(mod[0])
data['CarryAway'].fillna(mod[0], inplace=True)


# In[9]:


#filling the missinf values with mod
mod = data.mode()['Bar']
print(mod[0])
data['Bar'].fillna(mod[0], inplace=True)


# In[10]:


# #filling the missinf values with mod
# mod = data.mode()['CoffeeHouse']
# print(mod[0])
# data['CoffeeHouse'].fillna(mod[0], inplace=True)


# In[11]:


# #filling the missinf values with mod
# mod = data.mode()['Restaurant20To50']
# print(mod[0])
# data['Restaurant20To50'].fillna(mod[0], inplace=True)


# In[12]:


#filling the missinf values with mod
mod = data.mode()['RestaurantLessThan20']
print(mod[0])
data['RestaurantLessThan20'].fillna(mod[0], inplace=True)


# In[13]:


# here we used isnull for checking any null value is present or not
data.isnull().sum(axis = 0)


# In[14]:


# the command to check the types of data is
data.dtypes


# In[15]:


# to print first 5 rows of dataset we use head command
data.head()


# In[ ]:





# In[ ]:





# In[16]:


# this is the code for plotting the heat map
color = sns.color_palette()
sns.set_style('darkgrid')
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[17]:


# to count the number of each class in attribute we use value_counts
data.Y.value_counts()


# In[18]:


# Plotting histograms for all the attributes of dataset
data.hist(figsize = (20,20))
# To show the plotted histogram
plt.show()


# In[19]:


# copying the data of heart disease to y variable
y = data['Y'].copy()
# dropping the heart disease attribute from dataset and storing it in x variable
X = data.drop(['Y'], axis = 1)
# printing the number of class present in y with their time of occurance
print(f'Distribution Before Imblancing : {Counter(y)}')


# In[20]:


#Calculating hte imbalance percentage in the dataset
imbalance = len(data[data['Y'] == 0])/len(data)*100
imbalance


# In[21]:


# plotting the histogram for low imbalancing
hist = data['Y'].hist()
# tile of histogram
plt.title('Balance Data Set')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[22]:


# saving the figure in pdf format of hist
fig= hist.get_figure()
fig.savefig('Hist_Main.pdf')


# # **Low imblance**
# **The dataset over here is 65% Imbalanced**

# In[24]:


# making the data imbalance by 65% and storing it in XA and YA
XA, YA = make_imbalance(X, y, 
                          sampling_strategy={1: 7210, 0: 3882},
                        random_state = 42)
# printing the count after low imbalancing
print(f'Distribution After Imblancing 65%: {Counter(YA)}')


# In[25]:


# plotting the histogram for low imbalancing
hist1 = YA.hist()
# tile of histogram
plt.title('65% Imblance Dataset')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[26]:


# saving the figure in pdf format of hist1
fig= hist1.get_figure()
fig.savefig('Hist_65%.pdf')


# # **Medium Imblance**
# **The dataset over here is 75% Imbalanced**

# In[27]:


# making the data imbalance by 75% and storing it in XB and YB
XB, YB = make_imbalance(X, y, 
                          sampling_strategy={1: 7210, 0: 2403},
                        random_state = 42)
# printing the count after medium imbalancing
print(f'Distribution After Imblancing 75% : {Counter(YB)}')


# In[28]:


# plotting the histogram for low imbalancing
hist2 = YB.hist()
# tile of histogram
plt.title('75% Imblance Dataset')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[29]:


# saving the figure in pdf format of hist1
fig= hist2.get_figure()
fig.savefig('Hist_75%.pdf')


# # **High Imbalance**
# **The dataset over here is 90% Imbalanced**

# In[30]:


# making the data imbalance by 90% and storing it in XC and YC
XC, YC = make_imbalance(X, y, 
                          sampling_strategy={1: 7210, 0: 801},
                        random_state = 42)
# printing the count after High imbalancing
print(f'Distribution After Imblancing 90% : {Counter(YC)}')


# In[31]:


# plotting the histogram for low imbalancing
hist3 = YC.hist()
# tile of histogram
plt.title('90% Imblance Dataset')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[32]:


# saving the figure in pdf format of hist1
fig= hist3.get_figure()
fig.savefig('Hist_90%.pdf')


# # **Assignment-2**

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


# Returns the score of the selected measure for methods and performs 
# cross-validation internally, it receives a list of ground data and 
# predictions and displays the average and std deviation of each measure
def scoring_baseline(X, y, clf, model, measure='f1'):
    scores_a = cross_val_score(clf, X, y, cv=10, scoring = 'accuracy')
    scores_p = cross_val_score(clf, X, y, cv=10, scoring = 'precision')
    scores_r = cross_val_score(clf, X, y, cv=10, scoring = 'recall')
    scores_f = cross_val_score(clf, X, y, cv=10, scoring = 'f1')
    print("SCORES FOR MODEL ", model.upper(), ":", sep='')
    print("ACCURACY: %0.4f +/- %0.4f" % (scores_a.mean(), scores_a.std()))
    print("PRECISION: %0.4f +/- %0.4f" % (scores_p.mean(), scores_p.std()))
    print("RECALL: %0.4f +/- %0.4f" % (scores_r.mean(), scores_r.std()))
    print("F1: %0.4f +/- %0.4f" % (scores_f.mean(), scores_f.std()))
    if measure == 'accuracy':
        return scores_a
    if measure == 'precision':
        return scores_p
    if measure == 'recall':
        return scores_r
    if measure == 'f1':
        return scores_f

# Displays the Silhouette and elbow graphs, receives user input 
# and returns the optimal k and trained kMeans clusterer
def silhouette_elbow_cluster_selection(X, y):
    Sum_of_squared_distances = []
    silhouette = []
    K = range(2,15)
    max_score = -1
    max_k = -1
    # Kmeans trained with 14 possible k
    for k in K:
        km = KMeans(n_clusters=k).fit(X)
        Sum_of_squared_distances.append(km.inertia_)
        silhouette.append(silhouette_score(X, km.labels_, metric='euclidean'))
        
    # Silhouette graph
    plt.plot(K, silhouette, 'bx-')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()

    # User input
    input_silhouette = int(input("Please enter the selected number of clusters for the silhouette graph:\n"))

    # Elbow graph
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
    # User input
    input_elbow = int(input("Please enter the selected number of clusters for the elbow graph:\n"))
    
    # Upper and Lower bounds to test
    min_test_k = input_elbow if input_elbow < input_silhouette else input_silhouette
    max_test_k = input_elbow if input_elbow > input_silhouette else input_silhouette
    
    final_k = 0
    final_score = 0
    final_clustering = KMeans(n_clusters=min_test_k).fit(X)
    
    # Selecting optimal k according to f1 score
    for i in range(min_test_k, max_test_k + 1):
        km_test = KMeans(n_clusters=i).fit(X)
        score = f1_score(km_test.labels_, y, average='weighted')
        if score > final_score:
            final_k = i
            final_clustering = km_test
            final_score = score
            
    print('Best performing clustering f1-score is:', final_score,  "for n_clusters =", final_k)

    return final_k, final_clustering

# Returns a list of trained random forest classifiers/predictions per cluster
def cluster_training(X, y, k, chosen_km):
    true = np.zeros(k)
    false = np.zeros(k)
    X_cluster_list = [[] for i in range(k)]
    y_cluster_list = [[] for i in range(k)]
    result_list = [None for i in range(k)]

    for i in range(len(chosen_km.labels_)):
        true[chosen_km.labels_[i]] += 1 if y[i] == 1 else 0 # Count 1 labels per cluster
        false[chosen_km.labels_[i]] += 1 if y[i] == 0 else 0 # Count 0 labels per cluster
        X_cluster_list[chosen_km.labels_[i]].append(X[i]) # Stores feature arrays per cluster
        y_cluster_list[chosen_km.labels_[i]].append(y[i]) # Stores labels per cluster

    # Stores selected prediction for clusters with only one class
    for i in range(k):
        if true[i] == 0:
            result_list[i] = 1
        if false[i] == 0:
            result_list[i] = 0

    # Stores trained random forest classifier for clusters with more than one class
    for i in range(k):
        if result_list[i] == None:
            #train a random forest classifier
            clf = RandomForestClassifier(random_state=0, max_depth=2)
            #save it to result_list
            result_list[i] = clf.fit(X_cluster_list[i], y_cluster_list[i])
    
    return result_list

# Returns a list of predictions generated by the classifier of the predicted cluster
def cluster_testing(X, chosen_km, clfs):
    cluster_labels = chosen_km.predict(X) # Predict cluster of each example
    y_predicted = []

    # PRedict label according to each cluster
    for index, label in enumerate(cluster_labels):
        if clfs[label] == 0:
            y_predicted.append(0)
        elif clfs[label] == 1:
            y_predicted.append(1)
        else:
            y_predicted.append(clfs[label].predict(X[index].reshape(1, -1))[0])

    return y_predicted

# Returns the score of the selected measure for methods with external 
# cross-validation, it receives a list of ground data and predictions 
# and displays the average and std deviation of each measure
def scoring(y_true, y_pred, measure='f1'):
    scores_a = np.array([])
    scores_p = np.array([])
    scores_r = np.array([])
    scores_f = np.array([])

    for i in range(len(y_true)):
        scores_a = np.append(scores_a, accuracy_score(np.array(y_pred[i]), y_true[i]))
        scores_p = np.append(scores_p, precision_score(np.array(y_pred[i]), y_true[i], average='weighted'))
        scores_r = np.append(scores_r, recall_score(np.array(y_pred[i]), y_true[i], average='weighted'))
        scores_f = np.append(scores_f, f1_score(np.array(y_pred[i]), y_true[i], average='weighted'))
    
    print("SCORES FOR MODEL:")
    print("ACCURACY: %0.4f +/- %0.4f" % (scores_a.mean(), scores_a.std()))
    print("PRECISION: %0.4f +/- %0.4f" % (scores_p.mean(), scores_p.std()))
    print("RECALL: %0.4f +/- %0.4f" % (scores_r.mean(), scores_r.std()))
    print("F1: %0.4f +/- %0.4f" % (scores_f.mean(), scores_f.std()))
    
    if measure == 'accuracy':
        return scores_a
    if measure == 'precision':
        return scores_p
    if measure == 'recall':
        return scores_r
    if measure == 'f1':
        return scores_f

# Displays boxplots
def boxplot(data, title):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_xticklabels(['F1'])
    ax1.boxplot(data)
    plt.show()

# Displays boxplot without outliers
def noOutliers_boxplot(data, title):
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_xticklabels(['F1'])
    ax1.boxplot(data, showfliers=False)
    plt.show()


# In[34]:


rf_clf = RandomForestClassifier(random_state=0, max_depth=2)


# In[36]:


rf_baseline = scoring_baseline(X, y, rf_clf, 'Random Forest Tree', 'f1')


# In[37]:


rf_baseline = scoring_baseline(XA, YA, rf_clf, 'Random Forest Tree', 'f1')


# In[38]:


rf_baseline = scoring_baseline(XB, YB, rf_clf, 'Random Forest Tree', 'f1')


# In[39]:


rf_baseline = scoring_baseline(XC, YC, rf_clf, 'Random Forest Tree', 'f1')


# In[42]:


n = 10
fold = 1
splitter = StratifiedKFold(n_splits=n, random_state=None)
y_predicted_list = []
y_true_list = []

for train_index, test_index in splitter.split(X, y):
    print('\nFOLD:', fold)
    fold += 1
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    k, chosen_km = silhouette_elbow_cluster_selection(X_train, y_train)
    
    clfs = cluster_training(X, y, k, chosen_km)
    
    y_predicted_list.append(cluster_testing(X_test, chosen_km, clfs))
    
    y_true_list.append(y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, f1_score
# from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
# from sklearn.dummy import DummyClassifier

# from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# def baseline_result(data, output_class):
#   x = data.drop(labels= output_class, axis=1)
#   y = data[output_class]

#   baseline_clf = RandomForestClassifier()
#   baseline_scores = cross_val_score(baseline_clf, x, y, cv=10, scoring = 'f1')
#   print("Baseline f1 score: %0.4f +/- %0.4f" % (baseline_scores.mean(), baseline_scores.std()))


# In[ ]:


# class imbalance_dataset():
#   def __init__(self, dataset):
#     self.data = dataset
  
#   # Plot histogram
#   def hist_plot(self, width=20, height=15):
#     _ = self.data.hist(bins=50, figsize=(width,height))
#   # Plot heat map with linear correlation
#   def heatmap(self, width=20, height=7):
#     correlations = self.data.corr()
#     f, ax = plt.subplots(figsize=(width, height))
#     ax = sns.heatmap(correlations, annot=True, center=0,  cmap="YlGnBu")
#   # Plot box plot
#   def box_plot(self, width=20, height=7):
#     f, ax = plt.subplots(figsize=(width, height))
#     column = list(self.data.columns)
#     ax = self.data.boxplot(column= column)

#   def replace(self, label):
#     self.data.replace(label, inplace=True)


# In[ ]:


# # Function for assignment2
# def classification_assignment2(data,output_label):
#   # split data to stratified 10 fold
#   skf = StratifiedKFold(n_splits = 10)
#   x = data.drop(labels=output_label, axis=1).values
#   y = data[output_label].values

#   f1score = []
#   for train_index, test_index in skf.split(x, y):
#     # assign parameter with 9 folds to training data and the rest is test data
#     x_train = x[train_index]
#     x_test = x[test_index]
#     y_train = y[train_index]
#     y_test = y[test_index]

#     # run all trainning data to find the best number of clusters using maximun silhouette score
#     inertias, sil = [], []
#     for k in range(1, 12):
#         kmeans = KMeans(n_clusters=k)
#         y_pred = kmeans.fit_predict(x_train)
#         inertias.append(kmeans.inertia_)
#         if k > 1:
#             sil.append(silhouette_score(x_train, y_pred))
#     k = sil.index(max(sil))+2
    
#     # run kmean with the best number of clusters
#     kmeans = KMeans(n_clusters=k, random_state=1)
#     y_kmean = kmeans.fit_predict(x_train)
    
#     answer = np.zeros(len(y_test))
#     for ck in range(k):                         # create the random forest classifier for each cluster from kmean
#       index = np.where(y_kmean == ck)     
#       y_kmean_cluster = y_train[index]          # defind real data for a cluster
#       x_train_cluster = x_train[index]          # defind real data for a cluster
#       random_cluster = RandomForestClassifier()
#       random_cluster.fit(x_train_cluster, y_kmean_cluster)

#       y_pred = kmeans.predict(x_test)           # predict the cluster from test data
      
#       for i in range(len(y_pred)):              # run the random forest to identify the class in each cluster
#         if y_pred[i] == ck:
#           answer[i] = random_cluster.predict(np.reshape(x_test[i], (1,len(data.columns)-1)))

#     f1score.append(f1_score(y_test,answer))     # save the f1 score in the list

#   return f1score, k


# In[ ]:





# In[ ]:





# In[ ]:




