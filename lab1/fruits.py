# Reading the data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

data = pd.read_csv('fruits.csv')

# Showing the first 5 lines of data on the screen
print (data.head(5))


from sklearn import model_selection

# Preparing data and create training and test inputs and labels
inputs_train, inputs_test, labels_train, labels_test = \
       model_selection.train_test_split(data[['Weight', 'Surface']].values, data['Fruit'].values, test_size=0.2)

print("Training set size:", inputs_train.shape)
print("Test set size:", inputs_test.shape)



mpl.rcParams['lines.markersize'] = 12

# Plotting data
plt.figure(figsize=(12, 8))

plt.scatter(inputs_train[labels_train==1, 0], inputs_train[labels_train==1, 1], c='orange',  label='Oranges')
plt.scatter(inputs_train[labels_train==0, 0], inputs_train[labels_train==0, 1], c='green', label='Apples')
plt.scatter(inputs_test[labels_test==1, 0], inputs_test[labels_test==1, 1], c='orange',  label='Oranges (Test)',  marker='*')
plt.scatter(inputs_test[labels_test==0, 0], inputs_test[labels_test==0, 1], c='green', label='Apples (Test)', marker='*')

plt.xlabel('Weight')
plt.ylabel('Surface')
plt.legend()
plt.show()


from sklearn.svm import SVC

# Selecting the classifier we want to use
svm = SVC(kernel="linear")

# Learning (or training our model) based on inputs and labels from our dataset
svm.fit(inputs_train, labels_train)


# INPUT: enter weight and surface value for a imaginary and unknown fruit
weight  = 160
surface = 0.3

# Use our model to predict which fruit this is
fruit_type = svm.predict([[weight, surface]])
fruit_type = "orange" if     == 1 else "apple"
print (fruit_type)



from sklearn.metrics import accuracy_score

# Classify what the fruits are based on the test data
classifications = svm.predict(inputs_test)

# Print the score on the test data
print("SVM Test Set Score:")
print(accuracy_score(labels_test, classifications)*100)


from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, classifications)



from sklearn.metrics import precision_score, recall_score

print('Precision:', '{:0.2f}'.format(precision_score(labels_test, classifications)))
print('Recall:', '{:0.2f}'.format(recall_score(labels_test, classifications)))