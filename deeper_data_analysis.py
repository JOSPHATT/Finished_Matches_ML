import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/JOSPHATT/Finished_Matches/main/Finished_matches.csv"
df = pd.read_csv(url)

# 1. Inspect the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 2. Clean the data
# Remove duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# 3. Add a total goals column
df['Total_Goals'] = df['H_GOALS'] + df['A_GOALS']

# 4. Descriptive Statistics
# Calculate average goals scored by home and away teams
avg_home_goals = df['H_GOALS'].mean()
avg_away_goals = df['A_GOALS'].mean()
print(f"\nAverage Home Goals: {avg_home_goals}")
print(f"Average Away Goals: {avg_away_goals}")

# Find matches with the highest and lowest goal counts
max_goals = df['Total_Goals'].max()
min_goals = df['Total_Goals'].min()
highest_scoring_matches = df[df['Total_Goals'] == max_goals]
lowest_scoring_matches = df[df['Total_Goals'] == min_goals]
print("\nHighest Scoring Matches:")
print(highest_scoring_matches)
print("\nLowest Scoring Matches:")
print(lowest_scoring_matches)

# 5. Team Performance Analysis
# Calculate total goals scored by each team (home + away)
home_goals = df.groupby('HOME')['H_GOALS'].sum().reset_index(name='Home_Goals')
away_goals = df.groupby('AWAY')['A_GOALS'].sum().reset_index(name='Away_Goals')
team_performance = pd.merge(home_goals, away_goals, left_on='HOME', right_on='AWAY', how='outer').fillna(0)
team_performance['Total_Goals'] = team_performance['Home_Goals'] + team_performance['Away_Goals']
print("\nTotal Goals Scored by Each Team:")
print(team_performance.sort_values(by='Total_Goals', ascending=False))

# 6. Visualization
# Plot the distribution of total goals scored
plt.figure(figsize=(10, 6))
sns.histplot(df['Total_Goals'], kde=True, bins=10, color='blue')
plt.title('Distribution of Total Goals Scored in Matches')
plt.xlabel('Total Goals')
plt.ylabel('Frequency')
plt.show()

# Plot total goals scored by each team (bar chart for top 10 teams)
plt.figure(figsize=(12, 8))
team_performance = team_performance.sort_values(by='Total_Goals', ascending=False).head(10)  # Select top 10 teams
sns.barplot(x='HOME', y='Total_Goals', data=team_performance, palette='viridis')
plt.xticks(rotation=90)
plt.title('Total Goals Scored by Top 10 Teams')
plt.xlabel('Team')
plt.ylabel('Total Goals')
plt.show()

# Save the cleaned dataset and performance analysis to CSV files
cleaned_dataset_path = "Cleaned_Finished_matches.csv"
team_performance_path = "Team_Performance.csv"
df.to_csv(cleaned_dataset_path, index=False)
team_performance.to_csv(team_performance_path, index=False)

print(f"\nCleaned dataset saved to {cleaned_dataset_path}")
print(f"Team performance data saved to {team_performance_path}")