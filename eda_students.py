import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="darkgrid")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Display first rows
print("\nFIRST 5 ROWS")
print(df.head())

# Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Column names
print("\nCOLUMN NAMES")
print(df.columns)

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Create Average Score column
df['Average Score'] = (
    df['math score'] +
    df['reading score'] +
    df['writing score']
) / 3

# -------------------------------
# 1. Gender Distribution
# -------------------------------

plt.figure(figsize=(6,5))

sns.countplot(x='gender', data=df)

plt.title("Gender Distribution of Students",
          fontsize=16)

plt.savefig("outputs/gender_distribution.png")

plt.show()

# -------------------------------
# 2. Average Scores by Gender
# -------------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    x='gender',
    y='Average Score',
    data=df
)

plt.title("Average Score by Gender",
          fontsize=16)

plt.savefig("outputs/average_by_gender.png")

plt.show()

# -------------------------------
# 3. Parental Education Impact
# -------------------------------

plt.figure(figsize=(12,6))

sns.barplot(
    x='parental level of education',
    y='Average Score',
    data=df
)

plt.xticks(rotation=30)

plt.title("Impact of Parent Education on Scores",
          fontsize=16)

plt.savefig("outputs/parent_education.png")

plt.show()

# -------------------------------
# 4. Correlation Heatmap
# -------------------------------

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include='number')

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap",
          fontsize=16)

plt.savefig("outputs/heatmap.png")

plt.show()

# -------------------------------
# 5. Subject Score Distribution
# -------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df['math score'],
    kde=True
)

plt.title("Math Score Distribution",
          fontsize=16)

plt.savefig("outputs/math_distribution.png")

plt.show()

# -------------------------------
# 6. Lunch Effect on Performance
# -------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='lunch',
    y='Average Score',
    data=df
)

plt.title("Lunch Type vs Performance",
          fontsize=16)

plt.savefig("outputs/lunch_effect.png")

plt.show()

# -------------------------------
# 7. Test Preparation Impact
# -------------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    x='test preparation course',
    y='Average Score',
    data=df
)

plt.title("Effect of Test Preparation",
          fontsize=16)

plt.savefig("outputs/test_prep.png")

plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")