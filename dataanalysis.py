#Task 1
#Step 1:Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
#Step 2: Load the dataset
# Attempt to load the dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print(" Dataset loaded successfully!")
except Exception as e:
    print(" Error loading dataset:", e)
#Step 3: Take a quick look at the data
# Show the first few rows
df.head()
#Step 4:Understand the structure
# Check data types and look for missing values
print("\nData types:\n")
print(df.dtypes)

print("\nMissing values:\n")
print(df.isnull().sum())
#Task 2
#Step 1: Summary Statistics
# Show descriptive stats for numerical columns
df.describe()
#Step 2:Mean Value
# Group by species and compute the mean for each group
grouped_means = df.groupby('species').mean()
grouped_means
#Task 3
#Line chart
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length', color='red')
plt.title("Trend of Sepal and Petal Length (Sample Index)")
plt.xlabel("Sample Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.grid(True)
plt.show()
#Bar chart
grouped_means['petal length (cm)'].plot(kind='bar', color='lightgreen', edgecolor='black', figsize=(7, 5))
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
#Histogram
plt.figure(figsize=(7, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='orange', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
#Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set1')
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.show()
