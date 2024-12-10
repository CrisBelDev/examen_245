# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import numpy as np

# Cargar el archivo CSV
file_path = '/mnt/data/lung cancer data.csv'
data = pd.read_csv(file_path)

# Preprocesamiento: Codificar variables categóricas
le_gender = LabelEncoder()
le_cancer = LabelEncoder()

data['GENDER'] = le_gender.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = le_cancer.fit_transform(data['LUNG_CANCER'])

# Separar variables predictoras y objetivo
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Estandarización de los datos para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA con diferentes números de componentes
componentes_principales = [12, 10, 11, 9, 5, 3]
pca_resultados = {}

for n_componentes in componentes_principales:
    # Aplicar PCA
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    # Entrenar modelo supervisado
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    pca_resultados[n_componentes] = accuracy

# Aplicar aprendizaje no supervisado (K-Means sin "y")
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Resultados
pca_resultados, kmeans.labels_[:10]
