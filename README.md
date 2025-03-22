# Diabetes-Ai-Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("/content/diabetes (2).csv")
df.head()
df.head(10)
df.tail()
df.describe()
df.shape
print("/nMissing Values/n", df.isnull().sum())
x=df.drop(columns=['Outcome'])
y=df['Outcome']
counts=df['Outcome'].value_counts()
counts.plot(kind='bar', color=['blue','orange'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title("Diabetes Class Distribution")
plt.xticks(rotation=0)
plt.show()
import matplotlib.pyplot as plt
plt.hist(df[df['Outcome']==0]['Glucose'], bins=20, alpha=0.5, label="Non-Diabetic(0)", color='blue')
plt.hist(df[df['Outcome']==1]['Glucose'], bins=20, alpha=0.5, label="Diabetic(1)", color='red')
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.title("Glucose Level Distribution: Diabetic v/s Non-Diabetic")
plt.legend()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)*100
print(f"\nModel Accuracy: {accuracy:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
new_patient=np.array([[2,130,90,45,100,35.0,0.5,55]])
new_patient_scaled=scaler.transform(new_patient)
prediction=model.predict(new_patient_scaled)
print("\nPredicted Diabetes Outcome for New Patient:", "Diabetic"
    if prediction[0] == 1
      else
      "Non-Diabetic")
