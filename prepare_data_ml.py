import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
url = "https://raw.githubusercontent.com/JOSPHATT/Finished_Matches/main/Finished_matches.csv"
df = pd.read_csv(url)

# 1. Data Cleaning
# Remove duplicate rows
df = df.drop_duplicates()

# Check for missing values and drop rows with missing data
df = df.dropna()

# 2. Feature Engineering
# Create a goal difference column
df['Goal_Difference'] = df['H_GOALS'] - df['A_GOALS']

# Create a target column for match outcomes (1 = Home Win, 0 = Draw, -1 = Away Win)
def calculate_outcome(row):
    if row['H_GOALS'] > row['A_GOALS']:
        return 1  # Home win
    elif row['H_GOALS'] < row['A_GOALS']:
        return -1  # Away win
    else:
        return 0  # Draw

df['Match_Outcome'] = df.apply(calculate_outcome, axis=1)

# Encode categorical variables (HOME and AWAY teams)
label_encoder = LabelEncoder()
df['HOME_encoded'] = label_encoder.fit_transform(df['HOME'])
df['AWAY_encoded'] = label_encoder.fit_transform(df['AWAY'])

# Drop unnecessary columns to keep the dataset clean
df = df.drop(columns=['HOME', 'AWAY', 'Date'])  # Drop text columns not useful for ML

# 3. Splitting Data
# Define features (X) and target (y)
X = df.drop(columns=['Match_Outcome'])
y = df['Match_Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Normalization/Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the processed data to files
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
X_train_df.to_csv("X_train.csv", index=False)
X_test_df.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data preparation complete. Processed files saved:")
print("- X_train.csv")
print("- X_test.csv")
print("- y_train.csv")
print("- y_test.csv")