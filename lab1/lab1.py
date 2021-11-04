# Reading the data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

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




cvss_data = pd.read_csv('2020cvss.csv')
print (cvss_data.iloc[0:20])


#### Split dataset into training and testing sets
inputs_train, inputs_test, labels_train, labels_test = model_selection.train_test_split(
      cvss_data[["attackVector", "attackComplexity", "privilegesRequired", "userInteraction", "confidentialityImpact", "integrityImpact", "availabilityImpact"]].values,
      cvss_data['label'].values,
      test_size=0.2
    )

#### Plotting data


mpl.rcParams['lines.markersize'] = 12

plt.figure(figsize=(12, 10))

plt.scatter(inputs_train[labels_train==0, 1] + inputs_train[labels_train==0, 2]
             + inputs_train[labels_train==0, 3],inputs_train[labels_train==0, 0], c='green', label='Low')

plt.scatter(inputs_train[labels_train==1, 0] + inputs_train[labels_train==1, 2]
            + inputs_train[labels_train==1, 3], inputs_train[labels_train==1, 1],c='blue',label='Middle')

plt.scatter(inputs_train[labels_train==2, 0] + inputs_train[labels_train==2, 1]
            + inputs_train[labels_train==2, 3],inputs_train[labels_train==2, 2],c='orange',label='High')

plt.scatter(inputs_train[labels_train==3, 0] + inputs_train[labels_train==3, 1]
            +inputs_train[labels_train==3, 2],inputs_train[labels_train==3, 3],c='red',label='Critical')


plt.scatter(inputs_test[labels_test==0, 1] + inputs_test[labels_test==0, 2]
            + inputs_test[labels_test==0, 3],inputs_test[labels_test==0, 0], c='green', label='Low Test'
           ,marker='*')

plt.scatter(inputs_test[labels_test==1, 0] + inputs_test[labels_test==1, 2]
            + inputs_test[labels_test==1, 3],inputs_test[labels_test==1, 1], c='blue', label='Middle Test'
           ,marker='*')

plt.scatter(inputs_test[labels_test==2, 0] + inputs_test[labels_test==2, 1]
            + inputs_test[labels_test==2, 3],inputs_test[labels_test==2, 2],c='orange',label='High Test'
           ,marker='*')


plt.scatter(inputs_test[labels_test==3, 0] + inputs_test[labels_test==3, 1] 
            +inputs_test[labels_test==3, 2],inputs_test[labels_test==3, 3],c='red', label='Critical Test'
           ,marker='*')

plt.legend()
plt.show()


print("Training set size:", inputs_train.shape)
print("Test set size:", inputs_test.shape)




### Fit the SVM classifier
svm = SVC(kernel="linear")
svm.fit(inputs_train, labels_train)



### Try out your classifier using the vulnerability below:
vulnerability = [[0.85, 0.44, 0.27, 0.85, 0.66, 0.  , 0.  ]]
cvss = svm.predict(vulnerability)
cvss = score(cvss)
print(cvss)




### Use your classifier to classify the test set
classifications = svm.predict(inputs_test)
print("SVM Test Set Score:")
print(accuracy_score(labels_test, classifications)*100)


### Finally, compute the confusion matrix for the testing set
print('Precision:', '{:0.2f}'.format(precision_score(labels_test, classifications,average='micro')))
print('Recall:', '{:0.2f}'.format(recall_score(labels_test, classifications, average='micro')))


print (confusion_matrix(labels_test, classifications))