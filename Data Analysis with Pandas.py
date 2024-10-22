import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
# Display the first few rows of the dataset
print(df.head())

# Get a summary of the dataset
print(df.info())

# Get basic statistics of numerical columns
print(df.describe())
# Check for missing values
print(df.isnull().sum())

# Fill missing values in 'Age' with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to too many missing values
df.drop(columns=['Cabin'], inplace=True)
# Number of passengers by class
print(df['Pclass'].value_counts())

# Survival rate by class
print(df.groupby('Pclass')['Survived'].mean())

# Survival rate by gender
print(df.groupby('Sex')['Survived'].mean())

# Average age of survivors vs non-survivors
print(df.groupby('Survived')['Age'].mean())
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

# Bar chart: Number of passengers per class
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', data=df, palette='viridis')
plt.title('Number of Passengers per Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# Scatter plot: Age vs Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', data=df, hue='Survived', palette='coolwarm')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Heatmap: Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
