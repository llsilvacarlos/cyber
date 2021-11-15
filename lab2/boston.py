import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import model_selection
from sklearn.linear_model import LinearRegression


boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston = boston[["CRIM", "NOX", "RM"]]
boston["MEDV"] = boston_dataset.target
print("The size of our dataset (lines, columns):", boston.shape)
boston.head()

print (boston.iloc[0:20])

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=50)
plt.show()

print("Min:", min(boston['MEDV']))
print("Max:", max(boston['MEDV']))
print("Mean:", np.mean(boston['MEDV']))
print("Median:", np.median(boston['MEDV']))
print("Mode:", max(set(boston['MEDV']), key=list(boston["MEDV"]).count))
print("Skewness:", stats.skew(boston['MEDV']))


# plt.figure(figsize=(20, 5))

# features = ["CRIM", "NOX", "RM"]
# target = boston["MEDV"]

# for i, col in enumerate(features):
#     plt.subplot(1, len(features) , i+1)
#     x = boston[col]
#     y = target
#     plt.scatter(x, y, marker='o')
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel('MEDV')


# #plt.show()


# # It is common practice in machine learning to call data X and labels y
# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#       boston[["CRIM", "NOX", "RM"]].values,
#       boston[["MEDV"]].values,
#       test_size=0.2
#     )

# print("Training and test sizes:", X_train.shape, X_test.shape)


# linear_reg = LinearRegression()
# linear_reg.fit(X_train, y_train)
# linear_reg


