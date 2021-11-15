# LAB-1


![technology Python](https://img.shields.io/badge/technology-python-blue.svg)




Now let's train a classifier to classify security vulnerabilities according to their CVSS severity class. The dataset provided below contains the CVSS features and severity for all of the vulnerabilities reported in 2020. The vulnerabilities are separated in four severity classes:

<!-- blank line -->
* 0 = 'LOW'
* 1 = 'MIDDLE'
* 2 = 'HIGH'
* 3 = 'CRITICAL'



In this dataset we have seven features, corresponding to fields used to compute the CVSS score and severity:

* Attack Vector
* Attack Complexity
* Privileges Required
* User Interaction
* Confidentiality Impact
* Integrity Impact
* Availability Impact

Note that the features in the dataset were converted to a numerical score representation, following the CVSS score metrics. We have to do this conversion because the SVM algorithm expects numbers as input.

Now it's your turn, use this dataset to train and evaluate a SVM classifier that learns to classify the severity of a vulnerability based on its CVSS features. The steps for this are outlined in the cell below, we have already split the data into training and test for you, fill the remaining gaps with your code.


# Questions

*  Fit the SVM classifier
* Try out your classifier using the vulnerability below:
* Use your classifier to classify the test set
* Compute the final accuracy, precision, and recall on the test set
* Compute the confusion matrix for the testing set



# Quick start
* Install the Python Version 3.8.X and Anaconda (https://www.anaconda.com/)

#### Load Data
```
cvss_data = pd.read_csv('2020cvss.csv')
print (cvss_data.iloc[0:20])
```

#### Split dataset into training and testing sets

```
inputs_train, inputs_test, labels_train, labels_test = model_selection.train_test_split(
      cvss_data[["attackVector", "attackComplexity", "privilegesRequired", "userInteraction", "confidentialityImpact", "integrityImpact", "availabilityImpact"]].values,
      cvss_data['label'].values,
      test_size=0.2
    )
 
```

```
print("Training set size:", inputs_train.shape)
print("Test set size:", inputs_test.shape)
````


#### Fitting the SVM classifier



~~~
Selecting the classifier we want to use
svm = SVC(kernel="linear")

Learning (or training our model) based on inputs and labels from our dataset
svm.fit(inputs_train, labels_train)
~~~


#### Try out your classifier using the vulnerability below:
~~~
vulnerability = [[0.85, 0.44, 0.27, 0.85, 0.66, 0.  , 0.  ]]
cvss = svm.predict(vulnerability)
cvss = score(cvss)
print(cvss)
~~~

#### Use your classifier to classify the test set
~~~
classifications = svm.predict(inputs_test)
print("SVM Test Set Score:")
print(accuracy_score(labels_test, classifications)*100)
~~~

#### Finally, compute the confusion matrix for the testing set
~~~
print('Precision:', '{:0.2f}'.format(precision_score(labels_test, classifications,average='micro')))
print('Recall:', '{:0.2f}'.format(recall_score(labels_test, classifications, average='micro')))
print (confusion_matrix(labels_test, classifications))
~~~



