#Ihssane Aoune TP ACP
import numpy as np

def acp(X, k):

    # Calculer la matrice de Covariance 
    moy_vec = np.mean(X, axis=0)
    X_centre = X - moy_vec
    matrice_cov = np.dot(X_centre.T, X_centre) / (X.shape[0] - 1)

    # Calculer les vecteurs et valeurs propres :
    val_propres, vec_props = np.linalg.eig(matrice_cov)

    # Composantes principales:
    indices = np.argsort(val_propres)[::-1]  # Tri des valeurs propres du plus grand au plus petit
    selected_vec_props = vec_props[:, indices[:k]]

    #Projection des données
    donnee_reduite = np.dot(X_centre, selected_vec_props)

    return donnee_reduite


num_samples = int(input("Entrer le nombre des échantillons : "))
num_features = int(input("Entrer le nombre des features: "))

print("La matrice des données:")
data = np.zeros((num_samples, num_features))
for i in range(num_samples):
    for j in range(num_features):
        data[i][j] = float(input(f"Valuer de donnée :[{i}][{j}]: "))

# Appliquer ACP avec deux composantes principales k=2 

k = 2
donnee_reduite = acp(data, k)

print("Data shape d'origine:", data.shape)
print("Data shape réduit:", donnee_reduite.shape)
print("Données réduites:")
print(donnee_reduite)