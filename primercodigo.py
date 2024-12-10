# Importar bibliotecas necesarias
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Cargar el archivo CSV
file_path = '/mnt/data/lung cancer data.csv'
data = pd.read_csv(file_path)

# Preprocesamiento
le_gender = LabelEncoder()
le_cancer = LabelEncoder()

data['GENDER'] = le_gender.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le_cancer.fit_transform(data['LUNG_CANCER'])

# Separar variables predictoras y objetivo
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# ====== Primera Ejecución: 80% Entrenamiento / 20% Prueba ======
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar Árbol de Decisión
dt_classifier1 = DecisionTreeClassifier(random_state=42)
dt_classifier1.fit(X_train1, y_train1)

# Predicciones y evaluación
y_pred1 = dt_classifier1.predict(X_test1)
accuracy1 = accuracy_score(y_test1, y_pred1)
conf_matrix1 = confusion_matrix(y_test1, y_pred1)

# ====== Splits 100 veces (50/50) ======
accuracies = []
for i in range(100):
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.5, random_state=i)
    dt_classifier2 = DecisionTreeClassifier(random_state=42)
    dt_classifier2.fit(X_train2, y_train2)
    y_pred2 = dt_classifier2.predict(X_test2)
    accuracy2 = accuracy_score(y_test2, y_pred2)
    accuracies.append(accuracy2)

median_accuracy = np.median(accuracies)

# Resultados
accuracy1, conf_matrix1, median_accuracy
