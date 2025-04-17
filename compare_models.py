import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the processed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten to 1D array
y_test = pd.read_csv("y_test.csv").values.ravel()

# Initialize models
models = {
    "LightGBM": lgb.LGBMClassifier(objective="multiclass", num_class=3),
    "XGBoost": xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", use_label_encoder=False),
    "Logistic Regression": LogisticRegression(multi_class="multinomial", max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
}

# Dictionary to store performance metrics
performance_metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Store metrics
    performance_metrics["Model"].append(model_name)
    performance_metrics["Accuracy"].append(accuracy)
    performance_metrics["Precision"].append(precision)
    performance_metrics["Recall"].append(recall)
    performance_metrics["F1-Score"].append(f1)

# Convert metrics to a DataFrame
metrics_df = pd.DataFrame(performance_metrics)

# Display the metrics
print("\nModel Performance Comparison:")
print(metrics_df)

# Plot metrics for comparison
plt.figure(figsize=(12, 8))
metrics_df.set_index("Model").plot(kind="bar", figsize=(12, 8), rot=0, width=0.8)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()