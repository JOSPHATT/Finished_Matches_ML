import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the processed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten to 1D array
y_test = pd.read_csv("y_test.csv").values.ravel()

# 1. Train a LightGBM Model
# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define LightGBM parameters
params = {
    "objective": "multiclass",  # Since Match_Outcome has three classes: -1, 0, 1
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "n_estimators": 100,
}

# Train the model
print("Training LightGBM model...")
model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=10)

# 2. Make predictions
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = [list(proba).index(max(proba)) for proba in y_pred_proba]

# 3. Evaluate the model
print("\nModel Performance:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=["Away Win (-1)", "Draw (0)", "Home Win (1)"])
print(report)

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 4. Hyperparameter Tuning (Optional)
# Use GridSearchCV for hyperparameter tuning
param_grid = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
}
lgb_estimator = lgb.LGBMClassifier(objective="multiclass", num_class=3)
grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Save the trained model
model.save_model("lightgbm_model.txt")
print("\nTrained LightGBM model saved as lightgbm_model.txt")

# Recommendations based on performance
print("\nRecommendations:")
if accuracy < 0.6:
    print("- The accuracy is low. Consider improving feature engineering (e.g., adding new features, encoding strategies).")
    print("- Perform additional hyperparameter tuning.")
    print("- Check for potential data imbalance and apply resampling techniques if needed.")
elif 0.6 <= accuracy < 0.8:
    print("- The accuracy is decent, but there’s room for improvement.")
    print("- Explore more advanced feature engineering techniques and additional models for comparison.")
else:
    print("- The model is performing well. Consider deploying it after further validation on unseen data.")