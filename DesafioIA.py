import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregar Dataset Iris
iris = datasets.load_iris() 

# Separar dados de treino e teste
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="flores")

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1 )

# Modelo
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

# Métricas
acuracia = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)
matriz_df = pd.DataFrame(matriz, index=iris.target_names, columns=iris.target_names)
relatorio = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
relatorio_df = pd.DataFrame(relatorio).transpose().round(2)

print(f"\nAcuracy: {acuracia:.2f}")
print("\nConfusion Matrix:\n")
print(matriz_df)
print("\nClassification Report:\n")
print(relatorio_df)
