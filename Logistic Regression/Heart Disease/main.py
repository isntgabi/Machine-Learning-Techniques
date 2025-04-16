import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sb
from utils import clean_data, drop_columns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

df = pd.read_csv("data_in/heart_disease_uci.csv")
# print(df.head())

missing_values = df.isnull().sum().sort_values(ascending=False)
print("Valori lipsa: ", missing_values[missing_values > 0]) #thal and slope can be eliminated

df = drop_columns(df, ['thal', 'slope'])
df = clean_data(df, verbose=True)

df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df_encoded = df.copy()

#label encoding
df_encoded['sex'] = df_encoded['sex'].map({'Male': 0, 'Female': 1})

#one-hot encoding
df_encoded = pd.get_dummies(df_encoded, columns=['cp'], prefix='cp')

columns_to_drop = ['restecg']
df_model = df_encoded.drop(columns=columns_to_drop)
# print(df_model.head())

X = df_model.drop(columns=['target', 'id', 'dataset', 'num'], errors='ignore')
y = df_model['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(5, 4))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
plt.savefig("data_out/confusion_matrix.png")
plt.close()
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("data_out/roc_heart.png")
plt.close()
