import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


# Read it and print the first five rows
cvss_data = pd.read_csv('2020cvss_score.csv')
cvss_data.head()
print (cvss_data.iloc[0:20])

vulColumns = ["attackVector","attackComplexity", "privilegesRequired","userInteraction","confidentialityImpact","integrityImpact",	"availabilityImpact"]

#First, let's take a deeper look into our datasets. 
# Plot the distribution of severity scores for the dataset. 
# Compute also the minimum, maximum, mean, median, and mode statistics for the severity scores.


# vul = pd.DataFrame(cvss_data, columns=cvss_data.feature_names)
# vul = vul[["attackVector","attackComplexity", "privilegesRequired","userInteraction","confidentialityImpact","integrityImpact",	"availabilityImpact","score"]]
# vul["score"] = cvss_data.target
# print("The size of our dataset (lines, columns):", vul.shape)


vul = pd.DataFrame(cvss_data)

vul = vul[vulColumns]
vul["score"] = cvss_data.score

vul.head()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(vul['score'], bins=50)
#plt.show()

print("Min:", min(vul['score']))
print("Max:", max(vul['score']))
print("Mean:", np.mean(vul['score']))
print("Median:", np.median(vul['score']))
print("Mode:", max(set(vul['score']), key=list(vul["score"]).count))
print("Skewness:", stats.skew(vul['score']))


plt.figure(figsize=(20, 5))

features = vulColumns
target = vul["score"]

# for i, col in enumerate(features):
#     plt.subplot(1, len(features) , i+1)
#     x = vul[col]
#     y = target
#     plt.scatter(x, y, marker='o')
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel('score')

#plt.show()


X_train, X_test, y_train, y_test = model_selection.train_test_split(
      vul[vulColumns].values,
      vul[["score"]].values,
      test_size=0.2
    )

print("Training and test sizes:", X_train.shape, X_test.shape)



linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg

y_preds  = linear_reg.predict(X_test)

point_indices = range(20)
plt.plot(point_indices, y_preds[-20:], label="prediction")
plt.plot(point_indices, y_test[-20:], label="ground truth")
plt.legend()
plt.show()


kf = model_selection.KFold(n_splits=5, shuffle=True)

X, y = vul[vulColumns].values, vul[["score"]].values

print("Total samples:", X.shape[0])
for train_indices, test_indices in kf.split(X):
  print("Split:", train_indices.shape, test_indices.shape, test_indices[:5])




linear_reg = LinearRegression()
mses = []
for train_indices, test_indices in kf.split(X):
  X_train, X_test = X[train_indices], X[test_indices]
  y_train, y_test = y[train_indices], y[test_indices]
  linear_reg.fit(X_train, y_train)

  y_preds_train  = linear_reg.predict(X_train)
  y_preds_test  = linear_reg.predict(X_test)

  train_mse = mean_squared_error(y_preds_train, y_train)
  test_mse = mean_squared_error(y_preds_test, y_test)
  print("Training:", train_mse)
  print("Test:", test_mse, "\n")

  mses.append(test_mse)

print("Average test MSE:", np.mean(mses))



def adjusted_r2_score(y_preds, y_label, num_features):
  r2 = r2_score(y_preds, y_label)
  adjusted_r2 = (1 - (1 - r2) * ((y_label.shape[0] - 1)/(y_label.shape[0] - num_features - 1)))
  return adjusted_r2

num_features = X.shape[1]
linear_reg = LinearRegression()
r2s = []
adjusted_r2s = []
for train_indices, test_indices in kf.split(X):
  print("Fold", len(r2s))
  X_train, X_test = X[train_indices], X[test_indices]
  y_train, y_test = y[train_indices], y[test_indices]
  linear_reg.fit(X_train, y_train)

  y_preds_train  = linear_reg.predict(X_train)
  y_preds_test  = linear_reg.predict(X_test)

  train_r2 = r2_score(y_preds_train, y_train)
  test_r2 = r2_score(y_preds_test, y_test)
  train_adjusted_r2 = adjusted_r2_score(y_preds_train, y_train, num_features)
  test_adjusted_r2 = adjusted_r2_score(y_preds_test, y_test, num_features)

  print("Training R^2:", train_r2)
  print("Training adjusted R^2:", train_adjusted_r2)
  print("Test R^2:", test_r2)
  print("Test adjusted R^2:", test_adjusted_r2, "\n")

  r2s.append(test_r2)
  adjusted_r2s.append(test_adjusted_r2)

print("Average performance")
print("Average test r2:", np.mean(r2s))
print("Average test adjusted r2:", np.mean(adjusted_r2s))



####Polynomial Regression 

degrees = [1, 2, 3, 4, 5]
num_features = X_train.shape[1]
mean_r2s = []
mean_adjusted_r2s = []
for degree in degrees:
  poly_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
  r2s = []
  adjusted_r2s = []
  for train_indices, test_indices in kf.split(X):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    poly_reg.fit(X_train, y_train)
    y_preds_test  = poly_reg.predict(X_test)

    test_r2 = r2_score(y_preds_test, y_test)
    adjusted_test_r2 = adjusted_r2_score(y_preds_test, y_test, num_features)

    r2s.append(test_r2)
    adjusted_r2s.append(adjusted_test_r2)

  mean_r2 = np.mean(r2s)
  mean_adjusted_r2 = np.mean(adjusted_r2s)
  print(f"Degree = {degree}")
  print(f"Average test r2: {mean_r2}")
  print(f"Average test adjusted r2: {mean_adjusted_r2} \n")

  mean_r2s.append(mean_r2)
  mean_adjusted_r2s.append(mean_adjusted_r2)






barWidth = 0.3
r1 = range(1, len(mean_r2s)+1)
r2 = [x + barWidth for x in r1]

# Plot bars
r2_bars = plt.bar(r1, mean_r2s, width=barWidth, label="R$^{2}$")
adjusted_r2_bars = plt.bar(r2, mean_adjusted_r2s, width=barWidth, label="Adjusted R$^{2}$")

# Add text to top of bars
for rect in r2_bars + adjusted_r2_bars:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' %height, ha='center', va='bottom')

# Polish axes
tick_locations = [(r1[i] + r2[i])/2 for i in range(len(r1))]
plt.xticks(tick_locations, labels=r1)
plt.xlabel("Polynomial Degree")
plt.ylabel("Score")
plt.legend()

plt.show();