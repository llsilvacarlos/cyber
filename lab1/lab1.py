def score(num):
    '''Function that returns the severity of the classes 
    according to the constructed label'''
    if cvss == 0:
        return "LOW"
    elif cvss == 1:
        return "MIDDLE" 
    elif cvss == 2:
        return "HIGH"
    else: 
        return "CRITICAL"


# Reading the data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl


cvss_data = pd.read_csv('2020cvss.csv')
print (cvss_data.iloc[0:20])

# Showing the first 5 lines of data on the screen

from sklearn import model_selection
#### Split dataset into training and testing sets
inputs_train, inputs_test, labels_train, labels_test = model_selection.train_test_split(
      cvss_data[["attackVector", "attackComplexity", "privilegesRequired", "userInteraction", "confidentialityImpact", "integrityImpact", "availabilityImpact"]].values,
      cvss_data['label'].values,
      test_size=0.2
    )


print("Training set size:", inputs_train.shape)
print("Test set size:", inputs_test.shape)


mpl.rcParams['lines.markersize'] = 12

# Plotting data
#plt.figure(figsize=(12, 8))

#plt.scatter(inputs_train[labels_train==1, 0], inputs_train[labels_train==1, 1], c='green',  label='ALOW')
#plt.scatter(inputs_train[labels_train==0, 0], inputs_train[labels_train==0, 1], c='orange', label='MIDDLE')
#plt.scatter(inputs_train[labels_train==2, 0], inputs_train[labels_train==2, 1], c='blue', label='HIGH')
#plt.scatter(inputs_train[labels_train==3, 0], inputs_train[labels_train==3, 1], c='red', label='CRITICAL')

#plt.xlabel('Weight')
#plt.ylabel('Surface')
#lt.legend()
#plt.show()

from sklearn.svm import SVC
### Fit the SVM classifier

svm = SVC(kernel="linear")
svm.fit(inputs_train, labels_train)

### Try out your classifier using the vulnerability below:
vulnerability = [[0.85, 0.44, 0.27, 0.85, 0.66, 0.  , 0.  ]]

cvss = svm.predict(vulnerability)
cvss = score(cvss)
print(cvss)




### Use your classifier to classify the test set

from sklearn.metrics import accuracy_score


classifications = svm.predict(inputs_test)
print("SVM Test Set Score:")
print(accuracy_score(labels_test, classifications)*100)

### Compute the final accuracy, precision, and recall on the test set
# tip: use sklearn's "macro" averaging to compute precision and recall


### Finally, compute the confusion matrix for the testing set
from sklearn.metrics import precision_score, recall_score


print('Precision:', '{:0.2f}'.format(precision_score(labels_test, classifications,average='micro')))
print('Recall:', '{:0.2f}'.format(recall_score(labels_test, classifications, average='micro')))

from sklearn.metrics import confusion_matrix
print (confusion_matrix(labels_test, classifications))