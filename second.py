# Stroke Prediction
# Dataset: Healthcare Stroke Dataset (AIKosh.indiaai.gov.in)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# setting the style for all graphs
sns.set_theme(style='whitegrid', palette='Set2')

# reading the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print(df.head())
print(df.info())
print(df.describe())
print('\nMissing values:')
print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())

# ---------------- Data Preprocessing ----------------

df = df.drop_duplicates()
df = df.drop(columns=['id'])

# bmi has missing values so filling them with median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# gender has only one 'Other' value, replacing it with the most common gender
df['gender'] = df['gender'].replace('Other', np.nan)
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# smoking_status has 'Unknown' values, replacing them with the most common value
df['smoking_status'] = df['smoking_status'].replace('Unknown', np.nan)
df['smoking_status'] = df['smoking_status'].fillna(df['smoking_status'].mode()[0])


# OBJECTIVE 1: Distribution of stroke cases
print('\nOBJECTIVE 1: Stroke distribution')
print(df['stroke'].value_counts())
print('Stroke percentage:')
print((df['stroke'].value_counts(normalize=True) * 100).round(2))

sns.countplot(data=df, x='stroke', hue='stroke', legend=False)
plt.title('Stroke Distribution')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Number of Patients')
plt.show()


# OBJECTIVE 2: Age and stroke
print('\nOBJECTIVE 2: Age and stroke')
print('Average age by stroke:')
print(df.groupby('stroke')['age'].mean())

# making age groups to see which group has more strokes
age_group = pd.cut(df['age'], bins=[0, 18, 40, 60, 80, 120],
                   labels=['0-18', '19-40', '41-60', '61-80', '81+'])
age_rate = df.groupby(age_group, observed=True)['stroke'].mean() * 100

print('\nStroke rate (%) by age group:')
print(age_rate.round(2))

sns.barplot(x=age_rate.index, y=age_rate.values, hue=age_rate.index, legend=False)
plt.title('Stroke Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Stroke Rate (%)')
plt.show()


# OBJECTIVE 3: Medical conditions and stroke
print('\nOBJECTIVE 3: Medical conditions and stroke')

for col in ['hypertension', 'heart_disease']:
    rate = df.groupby(col)['stroke'].mean() * 100
    print('\nStroke rate (%) by', col)
    print(rate.round(2))

    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, legend=False)
    plt.title('Stroke Rate by ' + col)
    plt.xlabel(col)
    plt.ylabel('Stroke Rate (%)')
    plt.show()


# OBJECTIVE 4: Glucose level, BMI and stroke
print('\nOBJECTIVE 4: Glucose level, BMI and stroke')
print(df.groupby('stroke')[['avg_glucose_level', 'bmi']].mean())

for col in ['avg_glucose_level', 'bmi']:
    sns.boxplot(data=df, x='stroke', y=col, hue='stroke', legend=False)
    plt.title(col + ' vs Stroke')
    plt.xlabel('Stroke (0 = No, 1 = Yes)')
    plt.ylabel(col)
    plt.show()


# OBJECTIVE 5: Lifestyle factors and stroke
print('\nOBJECTIVE 5: Lifestyle factors and stroke')

for col in ['smoking_status', 'work_type', 'ever_married', 'Residence_type']:
    rate = df.groupby(col)['stroke'].mean() * 100
    print('\nStroke rate (%) by', col)
    print(rate.round(2).sort_values(ascending=False))

    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, legend=False)
    plt.title('Stroke Rate by ' + col)
    plt.xlabel(col)
    plt.ylabel('Stroke Rate (%)')
    plt.xticks(rotation=15)
    plt.show()


# ---------------- Machine Learning ----------------

# converting the text columns into numbers
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type',
                                 'Residence_type', 'smoking_status'], dtype=int)

# input (X) and output (y)
X = df.drop(columns=['stroke'])
y = df['stroke']

# train test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# stroke cases are very less in the data, so balancing only the training data using SMOTE
print('\nTraining data before SMOTE:')
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print('\nTraining data after SMOTE:')
print(y_train.value_counts())

# MODEL 1: KNN
print('\nMODEL 1: K-Nearest Neighbors')
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print('Accuracy:', round(accuracy_score(y_test, knn_pred), 3))
print('Classification Report:')
print(classification_report(y_test, knn_pred, zero_division=0))

# MODEL 2: Naive Bayes
print('\nMODEL 2: Gaussian Naive Bayes')
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print('Accuracy:', round(accuracy_score(y_test, nb_pred), 3))
print('Classification Report:')
print(classification_report(y_test, nb_pred, zero_division=0))

# MODEL 3: Logistic Regression
print('\nMODEL 3: Logistic Regression')
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('Accuracy:', round(accuracy_score(y_test, lr_pred), 3))
print('Classification Report:')
print(classification_report(y_test, lr_pred, zero_division=0))

# MODEL 4: Random Forest
print('\nMODEL 4: Random Forest')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('Accuracy:', round(accuracy_score(y_test, rf_pred), 3))
print('Classification Report:')
print(classification_report(y_test, rf_pred, zero_division=0))


# ---------------- Model Comparison ----------------
print('\nMODEL COMPARISON')

predictions = {
    'KNN': knn_pred,
    'Naive Bayes': nb_pred,
    'Logistic Regression': lr_pred,
    'Random Forest': rf_pred
}

# accuracy is for all patients, precision/recall/f1 are for stroke class (1)
results = pd.DataFrame(index=predictions.keys())
results['Accuracy'] = [accuracy_score(y_test, p) for p in predictions.values()]
results['Precision'] = [precision_score(y_test, p, zero_division=0) for p in predictions.values()]
results['Recall'] = [recall_score(y_test, p, zero_division=0) for p in predictions.values()]
results['F1 Score'] = [f1_score(y_test, p, zero_division=0) for p in predictions.values()]

print(results.round(3))

print('\nBest model for each metric:')
print(results.idxmax())

# best model is selected on the basis of accuracy
best_name = results['Accuracy'].idxmax()
print('\nBest model based on accuracy:', best_name)
print('Best accuracy:', round(results['Accuracy'].max(), 3))

print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions[best_name]))

# graph to compare all the models
results.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Four Models')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.legend(loc='upper right')
plt.show()

print('\nModel comparison is complete.')
