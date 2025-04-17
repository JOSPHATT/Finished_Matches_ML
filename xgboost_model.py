import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Load the processed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten to 1D array
y_test = pd.read_csv("y_test.csv").values.ravel()

# 1. Train an XGBoost Model
# Define the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", use_label_encoder=False)

# Train the model
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# 2. Make predictions
y_pred = xgb_model.predict(X_test)

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
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 200],
    "subsample": [0.8, 1.0],
}
grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", use_label_encoder=False),
                           param_grid=param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Save the trained model
xgb_model.save_model("xgboost_model.json")
print("\nTrained XGBoost model saved as xgboost_model.json")

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