import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

#load training and test data
train = pd.read_csv("data/BT2024262_train_var2.csv")
test = pd.read_csv("data/BT2024262_test_var2.csv")

#separate features and target
feature_columns = train.columns.drop("y")
X_train = train[feature_columns].values
y_train = train["y"].values
X_test = test[feature_columns].values

#check correlation of each feature with y to understand feature importance
print("Feature correlations with y:")
print(train.corr()["y"].sort_values(ascending=False))

#search over polynomial degrees 1 to 20 using cross-validation to find the best degree
print("\nDegree search (cross-validation R2):")
best_degree = 1
best_score = -np.inf
for degree in range(1, 21):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    model = Ridge(alpha=1e-3)
    scores = cross_val_score(model, X_poly, y_train, cv=5, scoring="r2")
    mean_score = scores.mean()
    print(f"  Degree {degree}: CV R2 = {mean_score:.6f} +/- {scores.std():.6f}")
    if mean_score > best_score:
        best_score = mean_score
        best_degree = degree

print(f"\nBest degree selected: {best_degree} with CV R2 = {best_score:.6f}")

#fit final polynomial regression model on full training data using best degree
poly_final = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly_final.fit_transform(X_train)
X_test_poly = poly_final.transform(X_test)

#train ridge regression on the full polynomial-expanded training set
final_model = Ridge(alpha=1e-3)
final_model.fit(X_train_poly, y_train)

#evaluate on training data to confirm fit quality
y_pred_train = final_model.predict(X_train_poly)
print(f"\nTraining R2  : {r2_score(y_train, y_pred_train):.6f}")
print(f"Training MSE : {mean_squared_error(y_train, y_pred_train):.6f}")

#predict on test set
y_pred_test = final_model.predict(X_test_poly)

#save predictions in the required submission format
submission = pd.DataFrame({"y": y_pred_test})
submission.to_csv("BT2024262_pred_var2.csv", index=False)
print("\nSaved predictions to BT2024262_pred_var2.csv")
print(submission.describe())