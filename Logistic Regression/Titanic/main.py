import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sb
from utils_preprocessing import clean_data, drop_columns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

df = pd.read_csv("data_in/train.csv")

missing_values = df.isnull().sum().sort_values(ascending=False)
print("Valori lipsa inițiale:\n", missing_values[missing_values > 0])

df = drop_columns(df, ['Cabin'])
df = clean_data(df, verbose=True)
print("\nVerificare finala lipsuri:\n", df.isnull().sum())

df_encoded = df.copy()

#label encoding
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

#one-hot encoding
df_encoded = pd.get_dummies(df_encoded, columns=['Embarked'], prefix='Embarked')

print(df_encoded.head())

columns_to_drop = ['PassengerId', 'Name', 'Ticket']
df_model = df_encoded.drop(columns=columns_to_drop)

X = df_model.drop('Survived', axis=1)
y = df_model['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
plt.savefig("data_out/confusion_matrix.png")
plt.close()

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("data_out/classification_report.csv")
print(report_df)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("data_out/roc_curve.png")
plt.close()

print(f"AUC: {roc_auc:.4f}")

coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coeffs)
coeffs.to_csv("data_out/feature_coefficients.csv", index=False)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("CV scores:", scores)
print("Average accuracy:", scores.mean())


