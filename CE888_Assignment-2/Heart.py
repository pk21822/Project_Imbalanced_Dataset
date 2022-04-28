#!/usr/bin/env python
# coding: utf-8

# # **Heart Disease**
# Cardiovascular disease (CVD) is a general term for conditions affecting the heart or blood vessels. CVD includes all heart and circulatory diseases, including coronary heart disease, angina, heart attack, congenital heart disease, hypertension, stroke and vascular dementia.
# 
#  Cardiovascular disease (CVD) is the main reason of the death if considered globally which has been resulted 17.9 million people. The main targeted people for this are basically the adults

# In[1]:


get_ipython().system('pip3 install imblearn')
get_ipython().system('pip3 install seaborn')


# # **Importing the Libraries**

# In[2]:


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


# # **Loading the dataset**

# In[3]:


# loading the dataset in csv from to data variable
data = pd.read_csv('heart.csv')
# to print the loaded dataset
data


# # **Pre-Processing and Cleaning the data**

# In[4]:


# cleanup is used here to convert the datapytes
cleanup = {"Sex": {"F":0, "M":1},
          "ExerciseAngina": {"N":0, "Y":1},
           "ST_Slope": {"Up":0, "Flat":1}
          }
data.replace(cleanup, inplace=True)
# to print first 5 rows of dataset we use head command
data.head()


# In[5]:


# the command to check the types of data is
data.dtypes


# In[6]:


# get_dummies is used as one-hot encode for categorical data
data=pd.get_dummies(data)
# to print first 5 rows of dataset we use head command
data.head


# In[7]:


# the command to check the types of data is
data.dtypes


# In[ ]:





# In[ ]:





# In[8]:


# this is the code for plotting the heat map
color = sns.color_palette()
sns.set_style('darkgrid')
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[ ]:





# In[9]:


# to count the number of each class in attribute we use value_counts
data.HeartDisease.value_counts()


# In[10]:


# here we used isnull for checking any null value is present or not
data.isnull().sum(axis = 0)


# In[11]:


# Plotting histograms for all the attributes of dataset
data.hist(figsize = (20,20))
plt.show()


# In[12]:


# import numpy as np
# import hvplot.pandas 
# # Calculate the correlations between each pair of variables and plot on the heatmap
# data.heatmap(20,7)


# In[ ]:


# from numpy.core.fromnumeric import sort

# # function for make the dataset to imbalanced dataset
# def make_imbalance(df,output_class,percentage):
#   df = df.sort_values(output_class, axis=0,ascending=True)
#   x, y = df[output_class].value_counts(ascending=True)
#   row = x * 100 / percentage
#   return df.head(math.floor(row))


# In[ ]:


# import math
# # make imbalanced dataset 65 %
# data_65 = make_imbalance(data,'HeartDisease',65)
# data_65['HeartDisease'].value_counts()


# In[13]:


# copying the data of heart disease to y variable
y = data['HeartDisease'].copy()
# dropping the heart disease attribute from dataset and storing it in x variable
x = data.drop(['HeartDisease'], axis = 1)
# printing the number of class present in y with their time of occurance
print(f'Distribution Before Imblancing : {Counter(y)}')


# In[14]:


#Calculating the imbalance percentage in the dataset
imbalance = len(data[data['HeartDisease'] == 0])/len(data)*100
imbalance


# In[15]:


# plotting the histogram for low imbalancing
hist = data['HeartDisease'].hist()
# tile of histogram
plt.title('Balance Data Set')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[16]:


# saving the figure in jpg format of hist
fig= hist.get_figure()
fig.savefig('Hist_Main%.pdf')


# # **Low imblance**
# **The dataset over here is 65% Imbalanced**

# In[17]:


# making the data imbalance by 65% and storing it in XA and YA
XA, YA = make_imbalance(x, y, 
                          sampling_strategy={1: 508, 0: 281},
                        random_state = 42)
# printing the count after low imbalancing
print(f'Distribution After Imblancing 65%: {Counter(YA)}')


# In[18]:


# plotting the histogram for low imbalancing
hist2 = YA.hist()
# tile of histogram
plt.title('65% Imblance Dataset')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[19]:


# saving the figure in jpg format of hist
fig= hist.get_figure()
fig.savefig('Hist_65%.pdf')


# # **Medium Imblance**
# **The dataset over here is 75% Imbalanced**

# In[20]:


# making the data imbalance by 75% and storing it in XB and YB
XB, YB = make_imbalance(x, y, 
                          sampling_strategy={1: 508, 0: 177},
                        random_state = 42)
# printing the count after medium imbalancing
print(f'Distribution After Imblancing 75% : {Counter(YB)}')


# In[21]:


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


# In[22]:


# saving the figure in jpg format of hist
fig= hist.get_figure()
fig.savefig('Hist_75%.pdf')


# # **High Imbalance**
# **The dataset over here is 90% Imbalanced**

# In[23]:


# making the data imbalance by 90% and storing it in XC and YC
XC, YC = make_imbalance(x, y, 
                          sampling_strategy={1: 508, 0: 64},
                        random_state = 42)
# printing the count after High imbalancing
print(f'Distribution After Imblancing 90% : {Counter(YC)}')


# In[24]:


# plotting the histogram for low imbalancing
hist2 = YC.hist()
# tile of histogram
plt.title('90% Imblance Dataset')
# label for x-axis
plt.xlabel('Number of Classes')
# label for y-axis
plt.ylabel('Total Number of Sample')
# to display histogram
plt.show


# In[25]:


# saving the figure in jpg format of hist
fig= hist.get_figure()
fig.savefig('Hist_75%.pdf')


# # Assignment 2

# In[26]:


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


# In[27]:


rf_clf = RandomForestClassifier(random_state=0, max_depth=2)


# In[28]:


rf_baseline = scoring_baseline(x, y, rf_clf, 'Random Forest Tree', 'f1')


# In[29]:


rf_baseline = scoring_baseline(XA, YA, rf_clf, 'Random Forest Tree', 'f1')


# In[30]:


rf_baseline = scoring_baseline(XB, YB, rf_clf, 'Random Forest Tree', 'f1')


# In[31]:


rf_baseline = scoring_baseline(XC, YC, rf_clf, 'Random Forest Tree', 'f1')


# In[32]:


n = 10
fold = 1
splitter = StratifiedKFold(n_splits=n, random_state=None)
y_predicted_list = []
y_true_list = []

for train_index, test_index in splitter.split(x, y):
    print('\nFOLD:', fold)
    fold += 1
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    k, chosen_km = silhouette_elbow_cluster_selection(X_train, y_train)
    
    clfs = cluster_training(X_train, y_train, k, chosen_km)
    
    y_predicted_list.append(cluster_testing(X_test, chosen_km, clfs))
    
    y_true_list.append(y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
count = 0
for train_index, test_index in skf.split(x, y):
    count +=1
    print('The number of counts\n',count)
    print("Train:", train_index, "Test:", test_index)
    X_train = x.iloc[train_index, :]
    y_train = y[train_index]
    X_test = x.iloc[test_index, :]
    y_test = y[test_index]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def cv_rf(x, y):
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, x, y, cv=10)
    return scores
scores = cv_rf(x, y)

scores.mean()


# In[ ]:


scores = cv_rf(XA, YA)
scores.mean()


# In[ ]:


scores = cv_rf(XB, YB)
scores.mean()


# In[ ]:


scores = cv_rf(XC, YC)
scores.mean()


# In[ ]:





# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(X_train)
data_transformed = mms.transform(X_train)


# In[ ]:


data_transformed


# In[ ]:


get_ipython().system('pip3 install yellobrick')
get_ipython().system('pip install -U yellowbrick')
#conda install -c districtdatalabs yellowbrick


# In[ ]:


# Elbow Method for K means
# Import ElbowVisualizer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(X_train)        # Fit data to visualizer
visualizer.show() 


# In[ ]:


# Silhouette Score for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)


# In[ ]:



from sklearn.datasets import make_blobs

features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)


# In[ ]:


features[:5]







true_labels[:5]


# In[ ]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler
kmeans.fit(scaled_features)


# In[ ]:


kmeans.inertia_


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


kmeans.n_iter_


# In[ ]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 14,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)


# In[ ]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score
# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[ ]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
# kmeans.labels_


# In[ ]:


kmeans.labels_


# In[ ]:


kmeans


# In[ ]:


cluster_list = []
cluster_list = kmeans.cluster_centers_


# In[ ]:


cluster_list


# In[ ]:


import numpy as np
b=np.mean(cluster_list)  
b 


# In[ ]:





# In[ ]:





# In[ ]:




