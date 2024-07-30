# Machine-Learning-
ML Classifications
# **INTRODUCTION**
Here we are selecting an Insurance dataset, where we are looking through independent varaiables:


1.   Age
2.   Sex
1.   bmi
2.   Children
1.   Region
2.   Charges

Here the data-set goes through the insurance rates of individuals who are living in various localities. There age, sex and smoking status is being considered and the charges are predicted


# **Model 1:LINEAR REGRESSION**

Linear regression is a statistical method used to model the relationship between a dependent variable (often denoted as
y) and one or more independent variables (often denoted as
x). It assumes that this relationship can be approximated by a linear equation of the form:

y=mx+b+ϵ

Linear regression is widely used for prediction and inference in various fields, including economics, finance, engineering, and social sciences. It's a simple yet powerful tool for understanding the relationship between variables and making predictions based on that relationship. Additionally, linear regression can be extended to handle more complex relationships through techniques like polynomial regression or adding interactions between variables 
# **Model 2:SVM MODEL**


svm model or the support vector model is the next model we are making in our project with the same pre-processing technique we did earlier.
Befor that a short summary on SVM and its main advantages, the few main advantages of using a svm model are:



*   Works well with limited taining data
*   Resilient to outliers
*   Less prone to overfitting
*   Can use different kerel functions
*   Memory efficient


Getting started with,

#1.Appling Feature encoding
le = LabelEncoder()
data["sex"] = le.fit_transform(data[["sex"]])
data["smoker"] = le.fit_transform(data[["smoker"]])
data["region"] = le.fit_transform(data[["region"]])
#2.Data cleaning
import pandas as pd
from sklearn.ensemble import IsolationForest
X = data.drop(columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

isolation_forest = IsolationForest(contamination=0.1, random_state=42)

isolation_forest.fit(X)

outlier_preds = isolation_forest.predict(X)

outlier_df = pd.DataFrame({'Outlier': outlier_preds})

data_with_outliers = pd.concat([data, outlier_df], axis=1)

outliers = data_with_outliers[data_with_outliers['Outlier'] == -1]
print("Outliers:")
print(outliers)
#We had 29 outliers in our dataset we have removed the outliers as a part of data-preprosessing
import pandas as pd

z_scores = (data - data.mean()) / data.std()

threshold = 3

outlier_rows = z_scores[(z_scores > threshold).any(axis=1)]

cleaned_data = data.drop(outlier_rows.index)

print("Number of outliers removed:", len(outlier_rows))
#3 Model Preperation
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = cleaned_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
Y = cleaned_data[['charges']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.25, random_state=42)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

svm_regressor = SVR(kernel='rbf',gamma='auto')

svm_regressor.fit(X_train_scaled, y_train)

# Evaluate the model
# Initialize SVM regressor with different kernels
kernels = ['linear', 'poly', 'sigmoid', 'rbf']
for kernel in kernels:
    svm_regressor = SVR(kernel=kernel, gamma='auto')
    svm_regressor.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = svm_regressor.predict(X_test_scaled)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (Kernel: {}): {:.2f}".format(kernel, mse))
#4calculating the R-score for the above
r2_svm = r2_score(y_test, y_pred)
print(" R^2 Score:",r2_svm) 
#5.Applying Cross validation
from sklearn.metrics import r2_score
cv_scores = cross_val_score(svm_regressor, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()
r2_scores = cross_val_score(svm_regressor, X_train_scaled, y_train, cv=5, scoring='r2')
mean_r2 = r2_scores.mean()

# Print mean MSE score, standard deviation, and mean R^2 score
print("Mean Squared Error (Cross-Validation):", mean_mse)
print("Standard Deviation of MSE (Cross-Validation):", std_mse)
print("Mean R^2 Score (Cross-Validation):", mean_r2)
#6.Coverting the Regression to classification by applying a threshold

from sklearn.metrics import confusion_matrix

threshold = 5000

# Convert predicted charges to binary labels based on the threshold
y_pred_binary = (y_pred > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)
from sklearn.metrics import classification_report
class_report = classification_report(y_test_binary, y_pred_binary)
print("Classification Report:")
print(class_report)
# **Model 3:DECISION TREE MODEL**

As the third model we are making a decision tree model. Some of the brief idea about the decision tree model is methioned below.

A decision model is a structured representation of a decision-making process that helps individuals or organizations make informed choices by evaluating various alternatives and their potential outcomes. These models are often used in business, finance, engineering, healthcare, and other fields where decision-making plays a crucial role


Some of the main advantages of decision tree model is:




*   Handling nonlinear Relationships
*   Robustness to Outliers and Missing Values
*   Scalability
*   No Assumptions about Data


The pre-prcosessing done in the decision tree model are:


1.Handling missing values

2.converting categotical values

3.Feature scaling

#1 Data pre-processing

#cheking for missing values

import pandas as pd
data = pd.read_csv('insurance 2.csv')
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values[missing_values > 0])

#yes , we don't have any missing values moving forward
#outlier detection and removal of outliers
import pandas as pd

z_scores = (data - data.mean()) / data.std()

threshold = 3

outlier_rows = z_scores[(z_scores > threshold).any(axis=1)]

cleaned_data = data.drop(outlier_rows.index)

print("Number of outliers removed:", len(outlier_rows))
#Model preperation


from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Separate features (X) and target variable (y)
X = cleaned_data.drop(columns=['charges'])
y = cleaned_data['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

tree_regressor = DecisionTreeRegressor()
tree_regressor.fit(X_train_scaled, y_train)

y_pred = tree_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



# Compute MSE using cross-validation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
cv_scores_mse = cross_val_score(tree_regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores_mse
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

# Compute R-squared (R^2) score using cross-validation
cv_scores_r2 = cross_val_score(tree_regressor, X_scaled, y, cv=5, scoring='r2')
mean_r2_tree = cv_scores_r2.mean()

# Print mean MSE score, standard deviation, and mean R^2 score
print("Mean Squared Error (Cross-Validation):", mean_mse)
print("Standard Deviation of MSE (Cross-Validation):", std_mse)
print("Mean R^2 Score (Cross-Validation):", mean_r2_tree)
#4calculating the R-score for the above


r2_tree = r2_score(y_test, y_pred)
print("R^2 Score:", r2_tree)
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Define thresholds for classification
threshold_low = 20000
threshold_high = 40000

# Convert regression predictions to classes based on thresholds
y_pred_classes = np.where(y_pred < threshold_low, 'Low', np.where(y_pred < threshold_high, 'Medium', 'High'))

y_test_classes = np.where(y_test < threshold_low, 'Low', np.where(y_test < threshold_high, 'Medium', 'High'))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))
# **4.RANDOM FOREST REGRESSION**

As the fourth model we are creating a RANDOM FOREST model. So giving a small into to RANDOM FOREST model.

Random Forest Regression is a machine learning technique that belongs to the ensemble learning family. It is an extension of decision tree regression and is widely used for both regression and classification tasks. In general, Random Forest Regression combines the power of multiple decision trees through ensemble learning, leveraging randomness and aggregation to create a robust and accurate regression model. It is widely used in various applications due to its simplicity, flexibility, and effectiveness in handling complex datasets.


Some of the main advantages of the model are:



*   Simple and Easy to Implement
*   Efficient in Training and Prediction
*   Handles High-Dimensional Data Well
*   Good Performance with Small Data
*   Robust to Irrelevant Features
*   Effectiveness in handling complex datasets

# data pre-processing
#applying encoding

le = LabelEncoder()
data["sex"] = le.fit_transform(data[["sex"]])
data["smoker"] = le.fit_transform(data[["smoker"]])
data["region"] = le.fit_transform(data[["region"]])
# outlier detection
import pandas as pd
from sklearn.ensemble import IsolationForest
X = data.drop(columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

isolation_forest = IsolationForest(contamination=0.1, random_state=42)

isolation_forest.fit(X)

outlier_preds = isolation_forest.predict(X)

outlier_df = pd.DataFrame({'Outlier': outlier_preds})

data_with_outliers = pd.concat([data, outlier_df], axis=1)

outliers = data_with_outliers[data_with_outliers['Outlier'] == -1]
print("Outliers:")
print(outliers)
#outlier removal

#We had 29 outliers in our dataset we have removed the outliers as a part of data-preprosessing
import pandas as pd

z_scores = (data - data.mean()) / data.std()

threshold = 3

outlier_rows = z_scores[(z_scores > threshold).any(axis=1)]

cleaned_data = data.drop(outlier_rows.index)

print("Number of outliers removed:", len(outlier_rows))
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

X = cleaned_data.drop(columns=['charges'])
y = cleaned_data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize and train the Random Forest Regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

y_pred = rf_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
from sklearn.metrics import r2_score
r2_tree = r2_score(y_test, y_pred)
print("R-squared Score:", r2_tree)
# Performing cross validation

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

X = cleaned_data.drop(columns=['charges'])
y = cleaned_data['charges']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

# Print mean MSE score and standard deviation
print("Mean Squared Error (Cross-Validation):", mean_mse)
print("Standard Deviation of MSE (Cross-Validation):", std_mse)
#So we have successfully developed four regression models now we are going to compare the four models and choose which one is the best model. For this i am just printing #all the R-score values into a matrix formal for comparison purposes.
# **CONCLUSION**
#After successfully creating the four regression models here we are comparing the performance of these models by printing the r2_score of the four models. A Dataframe has #been created to compare the values of the models.
