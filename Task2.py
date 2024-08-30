import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess dataset
df = pd.read_csv("train.csv")

# Fill missing values and handle categorical variables
df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0], 'Fare': df['Fare'].median()}, 
          inplace=True)
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Outlier removal using IQR
for col in ['Age', 'Fare']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Visualize missing values before handling them
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.title('Missing Values in Each Column')
    plt.show()

# Visualize Survived distribution
sns.set_style("whitegrid")
sns.countplot(x="Survived", data=df)
plt.title('Survival Count')
plt.show()

# Plot Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='red')
plt.title('Age Distribution')
plt.show()

# Plot Fare Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Fare'], bins=20, kde=True, color='purple')
plt.title('Fare Distribution')
plt.show()

# Plot Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', annot_kws={"size": 10})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Correlation Heatmap', pad=20)
plt.show()

# Plot Pairplot with adjusted marker size, transparency, and layout
pairplot = sns.pairplot(df, plot_kws={'s': 25, 'alpha': 0.6}, height=2.5)
plt.suptitle('Pairplot of the Dataset', y=1.0, fontsize=24, weight='bold')
pairplot.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
for ax in pairplot.axes.flat:
    ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=30)
plt.show()

# Plot Catplots and Boxplots
catplot_titles = [
    ('Pclass', 'Survived', 'Survival Count by Passenger Class'),
    ('Sex_male', 'Survived', 'Survival Count by Gender'),
    ('Embarked_Q', 'Survived', 'Survival Count by Embarkation Point')
]
for x, hue, title in catplot_titles:
    sns.catplot(data=df, x=x, hue=hue, kind='count', palette='husl', height=4, aspect=1.5).set_titles(title)
    plt.show()

boxplot_titles = [
    ('Pclass', 'Age', 'Survived', 'Age Distribution across Passenger Classes with Survival'),
    ('Pclass', 'Fare', 'Survived', 'Fare Distribution across Passenger Classes with Survival')
]
for x, y, hue, title in boxplot_titles:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue, palette='husl')
    plt.title(title)
    plt.show()

# Feature Scaling using StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Prepare data for training
X, y = df.drop('Survived', axis=1), df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nAccuracy Score:\n{accuracy_score(y_test, y_pred)}")

# Feature Importance
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, 
                                   columns=['Importance']).sort_values('Importance', ascending=False)
print(f"\nFeature Importances:\n{feature_importances}")
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.title('Feature Importance')
plt.show()
