import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("hand_data.csv")

y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, shuffle=True
)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

import joblib

joblib.dump(model, "asl_model.pkl")

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
