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
