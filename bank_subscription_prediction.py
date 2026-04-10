# CODE FOR DECISION TREE
#step 1 import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split #splits data into training n testing parts
from sklearn.metrics import accuracy_score, precision_score # meausres how good the model is
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_curve, auc

#step 2 reads the data file and puts it into a variable named df1
df1 = pd.read_csv('bank dataset for assignment.csv')

#step 3 extract features into a dataframe to give them numerical values
dict_features= {'job':{'housemaid':0, 'services':1, 'admin.':2, 'blue-collar':3, 'technician':4, 'retired':5, 'management':6, 'unemployed':7, 'unknown':8, 'self-employed':9, 'entrepreneur':10},
                'marital':{'married':0, 'single':1, 'divorced':2, 'unknown':3},
                'education':{'basic.4y':0, 'high.school':1, 'basic.6y':2, 'basic.9y':3, 'professional.course':4, 'unknown':5, 'university.degree':6, 'illiterate':7 },
                'default':{'no':0, 'unknown':1},
                'housing':{'no':0, 'yes':1},
                'loan': {'no':0, 'yes':1},
                'poutcome':{'failure':0, 'nonexistent':1, 'success':2}}

#step 4 extract the label into another dataframe to give it numerical values
dict_labels= {'y':{'no':0, 'yes':1}}


#step 5 extract the features from the full dataset.
#dfx contains all the columns the model will use to learn patterns (age, job, marital, etc)
dfx= df1 [['age', 'job', 'marital', 'education', 'default', 'housing',
           'loan','duration','campaign', 'pdays','previous','poutcome',
           'emp.var.rate','cons.price.idx',
           'cons.conf.idx','euribor3m','nr.employed']]

#step 6 extract the label from the full dataset
#dfy contains the target column "y", which the model will try to predict
dfy= df1 [['y']]

# step 7 make separate copies of the dataframes so changes can be made safely
# this prevents pandas warnings when updating columns (when assigning numerical values to non-numeric stuff)
dfx = dfx.copy()
dfy = dfy.copy()


#step 8 map it column by column
#map only categorical columns from dfx
for col in dict_features: #Goes through each column in the dictionary(job,marital, education) etc.
    if col in dfx.columns: #checks if the column exists in the dfx dataframe
        dfx[col] = dfx[col].map(dict_features[col])# maps only the  column which is there in the dictionary and dataframe


#step 9 map the label column
dfy ['y'] = dfy['y'].map (dict_labels['y'])

# handling missing values
dfx = dfx.dropna()
dfy = dfy.loc[dfx.index]

# step 10 create a decision tree model
DecisionTree = tree.DecisionTreeClassifier(max_depth = 5, criterion = 'gini', random_state = 42)


#step 11 split the dataset into training and testing data
#20% of data is kept for testing
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=42)

#step 12 train the decision tree model (using features n labels from training data)
DecisionTree.fit(x_train, y_train)

# training accuracy
train_acc_dt = DecisionTree.score(x_train, y_train)
print(f"Decision Tree Training Accuracy: {train_acc_dt}")

#step 13 makes predictions using the test data (x_test), n stores it in y_pred
y_pred =DecisionTree.predict(x_test)

#step 14 calculate the accuracy of the decision tree model
accuracy = accuracy_score(y_test, y_pred)
print ("Decision Tree Accuracy: ", accuracy)

#step 15 calculate the precision of the decision tree model
precision = precision_score(y_test, y_pred)
print ("Decision Tree Precision: ", precision)

# confusion Matrix
cm=confusion_matrix(y_test, y_pred)
print ("Decision tree Confusion Matrix: ")
print(cm)

#recall score
recall = recall_score(y_test, y_pred)
print ("Decision tree Recall: ", recall)

#F1 score
F1= f1_score (y_test, y_pred)
print ("Decision tree F1 Score: ", F1)


#step 16 visualization for Decision Tree
plt.figure(figsize=(25, 15))  # Bigger figure for clarity
tree.plot_tree(DecisionTree,
               feature_names=dfx.columns.tolist(),  # Use feature names
               class_names=['No Subscribe', 'Subscribe'],  # Clear class names
               filled=True,     # Color boxes
               rounded=True,    # Rounded corners
               fontsize=10,     # Readable text size
               max_depth=3)     # Show only first 3 levels 
plt.title('Decision Tree Structure - Bank Subscription Prediction', fontsize=14)
plt.show()

 
# CODE FOR KNN
#step 1 import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split #splits data into training n testing parts
from sklearn.metrics import accuracy_score, precision_score  # meausres how good the model is
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

#step 2 reads the data file and puts it into a variable named df1
df1 = pd.read_csv('bank dataset for assignment.csv')


#step 3 extract features into a dataframe to give them numerical values
dict_features= {'job':{'housemaid':0, 'services':1, 'admin.':2, 'blue-collar':3, 'technician':4, 'retired':5, 'management':6, 'unemployed':7, 'unknown':8, 'self-employed':9, 'entrepreneur':10},
                'marital':{'married':0, 'single':1, 'divorced':2, 'unknown':3},
                'education':{'basic.4y':0, 'high.school':1, 'basic.6y':2, 'basic.9y':3, 'professional.course':4, 'unknown':5, 'university.degree':6, 'illiterate':7 },
                'default':{'no':0, 'unknown':1},
                'housing':{'no':0, 'yes':1},
                'loan': {'no':0, 'yes':1},
                'poutcome':{'failure':0, 'nonexistent':1, 'success':2}}

#step 4 extract the label into another dataframe to give it numerical values
dict_labels= {'y':{'no':0, 'yes':1}}


#step 5 extract the features from the full dataset.
#dfx contains all the columns the model will use to learn patterns (age, job, marital, etc)
dfx= df1 [['age', 'job', 'marital', 'education', 'default', 'housing',
           'loan','duration','campaign', 'pdays','previous','poutcome',
           'emp.var.rate','cons.price.idx',
           'cons.conf.idx','euribor3m','nr.employed']]

#step 6 extract the label from the full dataset
#dfy contains the target column "y", which the model will try to predict
dfy= df1 [['y']]

# step 7 make separate copies of the dataframes so changes can be made safely
# this prevents pandas warnings when updating columns (when assigning numerical values to non-numeric stuff)
dfx = dfx.copy()
dfy = dfy.copy()


#step 8 map it column by column
#map only categorical columns from dfx
for col in dict_features: #Goes through each column in the dictionary(job,marital, education) etc.
    if col in dfx.columns: #checks if the column exists in the dfx dataframe
        dfx[col] = dfx[col].map(dict_features[col])# maps only the  column which is there in the dictionary and dataframe

#step 9 map the label column
dfy ['y'] = dfy['y'].map (dict_labels['y'])

#Quick Missing values Check
#print ("Missing in features:", dfx.isnull().sum().sum())
#print ("Missing in labels:", dfy.isnull().sum().sum())

#step 10 handling missing values (this method removes the rows that has missing values)
dfx = dfx.dropna() #looks at the features dataframe, and if it finds any row that has at least one missing value,
# it removes the entire row from dfx n keeps only rows with complete data
dfy = dfy.loc[dfx.index]


#step 11 create the KNN model (n_neighbors=5, but can change it)
knn=KNeighborsClassifier(n_neighbors=5)

#step 12 split the dataset into x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=42)


#step 13 train the model
knn.fit(x_train, y_train.values.ravel())

# training accuracy
train_acc_knn = knn.score(x_train, y_train)
print(f"KNN Training Accuracy: {train_acc_knn}")

#step 14 make predictions
y_pred = knn.predict(x_test)

#step 15 calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy:",accuracy)

#step 16 calculate the precision of the KNN model
precision = precision_score(y_test, y_pred)
print ("KNN Precision: ", precision)

# confusion matrix
cm= confusion_matrix(y_test, y_pred)
print("KNN Confusion Matrix")
print(cm)

# Recall score
recall= recall_score(y_test, y_pred)
print("KNN Recall: ", recall)

#F1 score
f1= f1_score(y_test, y_pred)
print("KNN F1 score: ", f1)


#step 17 visuaization for KNN
# KNN Visualization - Clear and Simple
plt.figure(figsize=(10, 6))

# Plot predictions
plt.scatter(x_test['age'], x_test['duration'], c=y_pred, alpha=0.7)

# Add color bar
plt.colorbar(label='Prediction: 0=No Subscribe, 1=Subscribe')

# Labels
plt.xlabel('Age')
plt.ylabel('Duration')
plt.title('KNN Predictions')

plt.show()

 
