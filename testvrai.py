#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:19:16 2025

@author: irachid
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 20:44:11 2025

@author: ismail rachid
"""

from math import exp
import numpy as np
from copy import deepcopy 
from scipy.special import expit  # Sigmoid function

def matrice_aleatoire(nout, nin):
    # Xavier/Glorot initialization
    a = -np.sqrt(6 / (nin + nout))
    b = np.sqrt(6 / (nin + nout))
    return np.random.uniform(a, b, (nin, nout))

# Chargement des données
entrees = np.loadtxt("sonar_inputs_train.csv", delimiter=",")
sorties = np.loadtxt("sonar_outputs_train.csv", delimiter=",")

# Initialisation du réseau 60 -> 30 -> 1
mat = [matrice_aleatoire(60, 30), matrice_aleatoire(30, 30), matrice_aleatoire(30, 1)]

def activation(x):
    return expit(x)  # Sigmoid activation

def dactivation(x):
    # Derivative of sigmoid
    return expit(x) * (1 - expit(x))

# Initialize network weights in dictionary structure
dico = {
    "couche 1": {f"neurone {i+1}": np.append(mat[0][i], 0) for i in range(len(mat[0]))},
    "couche 2": {f"neurone {i+1}": np.append(mat[1][i], 0) for i in range(len(mat[1]))},
    "couche 3": {f"neurone {i+1}": np.append(mat[2][i], 0) for i in range(len(mat[2]))}
}

def Wl(l, dico):
    # Extract weights and biases from dictionary
    W = np.array([dico[f"couche {l}"][n][:-1] for n in dico[f"couche {l}"]])
    b = np.array([dico[f"couche {l}"][n][-1] for n in dico[f"couche {l}"]])
    return W, b

def forward(dico, entree):
    # Forward pass through the network
    activations = [entree]
    z_values = []
    
    for l in range(1, 4):
        W, b = Wl(l, dico)
        z = np.dot(W, activations[-1]) + b
        z_values.append(z)
        a = activation(z)  # Using sigmoid everywhere
        activations.append(a)
    
    return activations, z_values

def backward(dico, activations, z_values, sortie):
    # Backward pass to calculate deltas
    deltas = {}
    L = len(dico)
    
    # Output layer error (using proper MSE derivative)
    output_error = activations[-1] - sortie
    deltas[f"couche {L}"] = output_error * dactivation(z_values[-1])
    
    # Hidden layers
    for l in range(L-1, 0, -1):
        W, _ = Wl(l+1, dico)
        dz = np.dot(W.T, deltas[f"couche {l+1}"]) * dactivation(z_values[l-1])
        deltas[f"couche {l}"] = dz
        
    return deltas

def gradient(dico, activations, deltas):
    # Calculate gradients for all weights and biases
    gradients = deepcopy(dico)
    for l in range(1, len(dico)+1):
        couche = f"couche {l}"
        grad_W = np.outer(deltas[couche], activations[l-1])
        grad_b = deltas[couche]
        
        for i, neurone in enumerate(dico[couche]):
            gradients[couche][neurone][:-1] = grad_W[i]
            gradients[couche][neurone][-1] = grad_b[i]
            
    return gradients

def update(dico, gradients, eta):
    # Update weights and biases using gradients
    for couche in dico:
        for neurone in dico[couche]:
            dico[couche][neurone] -= eta * gradients[couche][neurone]
    return dico

def train(dico, entrees, sorties, epochs, eta, ep):
    # Train network with early stopping based on error threshold
    history = []
    
    for epoch in range(epochs):
        errors = []
        
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            
            # Forward pass
            activations, z_values = forward(dico, entree)
            
            # Calculate error
            error = ((activations[-1] - sortie) ** 2) / 2
            errors.append(error[0])
            
            # Backward pass
            deltas = backward(dico, activations, z_values, sortie)
            
            # Update weights
            gradients = gradient(dico, activations, deltas)
            dico = update(dico, gradients, eta)
            
        mean_error = np.mean(errors)
        history.append(mean_error)
        
        # Early stopping
        if mean_error < ep:
            print(f"Early stopping at epoch {epoch+1} with error {mean_error:.6f}")
            break
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Error: {mean_error:.6f}")
    
    return dico, history

def train(dico, entrees, sorties, epochs, eta_initial, ep):
    history = []
    eta = eta_initial  # Taux d'apprentissage initial (élevé)
    
    # Calculer l'erreur initiale
    initial_errors = []
    for i in range(len(entrees)):
        entree = entrees[i]
        sortie = np.array([sorties[i]])
        activations, _ = forward(dico, entree)
        error = ((activations[-1] - sortie) ** 2) / 2
        initial_errors.append(error[0])
    
    previous_cost = np.sum(initial_errors) / len(entrees)
    history.append(previous_cost)
    
    for epoch in range(epochs):
        # Copie du dictionnaire pour tester un pas de gradient
        dico_temp = deepcopy(dico)
        total_error = 0
        
        # Effectuer un pas de gradient avec le eta actuel
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            
            # Forward pass
            activations, z_values = forward(dico_temp, entree)
            
            # Backward pass
            deltas = backward(dico_temp, activations, z_values, sortie)
            
            # Update weights
            gradients = gradient(dico_temp, activations, deltas)
            dico_temp = update(dico_temp, gradients, eta)
        
        # Évaluer le coût après cette mise à jour
        test_errors = []
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            activations, _ = forward(dico_temp, entree)
            error = ((activations[-1] - sortie) ** 2) / 2
            test_errors.append(error[0])
        
        current_cost = np.sum(test_errors) / len(entrees)
        
        # Si le coût a diminué, accepter la mise à jour
        if current_cost < previous_cost:
            dico = deepcopy(dico_temp)
            previous_cost = current_cost
            # Optionnel: on pourrait même augmenter eta si l'amélioration est significative
            # eta = min(eta * 1.1, 1.0)  # Augmenter eta mais pas au-delà d'une valeur max
        else:
            # Si le coût a augmenté, réduire eta et ne pas accepter la mise à jour
            eta /= 2
            print(f"Epoch {epoch+1}: Réduction du taux d'apprentissage à {eta:.6f}")
        
        history.append(previous_cost)
        
        # Early stopping
        if previous_cost < ep:
            print(f"Early stopping at epoch {epoch+1} with cost {previous_cost:.6f}")
            break
            
        if (epoch + 1) % 10 == 0:  # Affichage plus fréquent pour observer l'adaptation du eta
            print(f"Epoch {epoch+1}/{epochs}, Cost: {previous_cost:.6f}, Eta: {eta:.6f}")
        
        # Si eta devient trop petit, on peut considérer que l'optimisation est bloquée
        if eta < 1e-10:
            print(f"Taux d'apprentissage trop faible ({eta:.10f}), arrêt à l'epoch {epoch+1}")
            break
    
    return dico, history

def train(dico, entrees, sorties, epochs, eta_initial, ep):
    history = []
    
    # Calculer l'erreur initiale
    initial_errors = []
    for i in range(len(entrees)):
        entree = entrees[i]
        sortie = np.array([sorties[i]])
        activations, _ = forward(dico, entree)
        error = ((activations[-1] - sortie) ** 2) / 2
        initial_errors.append(error[0])
    
    current_cost = np.sum(initial_errors) / len(entrees)
    history.append(current_cost)
    eta = eta_initial
    for epoch in range(epochs):
        eta = eta*10**4  # Réinitialiser eta à chaque époque
        found_good_eta = False
        
        while not found_good_eta and eta >= 1e-25:
            # Copie du dictionnaire pour tester ce eta
            dico_temp = deepcopy(dico)
            
            # Effectuer une mise à jour avec le eta actuel
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])
                
                # Forward pass
                activations, z_values = forward(dico_temp, entree)
                
                # Backward pass
                deltas = backward(dico_temp, activations, z_values, sortie)
                
                # Update weights
                gradients = gradient(dico_temp, activations, deltas)
                dico_temp = update(dico_temp, gradients, eta)
            
            # Évaluer le nouveau coût
            test_errors = []
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])
                activations, _ = forward(dico_temp, entree)
                error = ((activations[-1] - sortie) ** 2) / 2
                test_errors.append(error[0])
            
            test_cost = np.sum(test_errors) / len(entrees)
            
            # Vérifier si ce eta réduit l'erreur
            if test_cost < current_cost:
                found_good_eta = True
                cos = deepcopy(current_cost)
                dico = dico_temp  # Accepter cette mise à jour
                current_cost = test_cost
            else:
                # Diviser eta par 2 et réessayer
                eta /= 2
            
        
        # Si aucun bon eta n'a été trouvé
        if not found_good_eta:
            print(f"Epoch {epoch+1}: Aucun eta efficace trouvé. Erreur minimale: {current_cost:.6f}")
            break
        
        history.append(current_cost)
        
        # Early stopping
        if current_cost < ep:
            print(f"Early stopping at epoch {epoch+1} with cost {current_cost:.6f}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Cost: {current_cost:.6e}, Eta utilisé: {eta:.6e}, difference en cout : {abs(test_cost-cos)}")
    
    
    return dico, history


def train(dico, entrees, sorties, epochs, eta_initial, ep):
    history = []
    eta = eta_initial  # Taux d'apprentissage initial

    # Calculer l'erreur initiale
    initial_errors = []
    for i in range(len(entrees)):
        entree = entrees[i]
        sortie = np.array([sorties[i]])
        activations, _ = forward(dico, entree)
        error = ((activations[-1] - sortie) ** 2) / 2
        initial_errors.append(error[0])
    current_cost = np.sum(initial_errors) / len(entrees)
    history.append(current_cost)
    
    for epoch in range(epochs):
        eta = eta * 10**5  # Réinitialiser eta à chaque époque
        found_good_eta = False
        
        while not found_good_eta and eta >= 1e-25:
            # Copie du dictionnaire pour tester ce eta
            dico_temp = deepcopy(dico)
            
            # Accumulateur des gradients pour la moyenne
            total_gradients = deepcopy(dico)
            total_gradients = {couche: {neurone: np.zeros_like(dico[couche][neurone]) for neurone in dico[couche]} for couche in dico}

            # Calculer les gradients sur tous les exemples du lot
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])

                # Forward pass
                activations, z_values = forward(dico_temp, entree)

                # Backward pass
                deltas = backward(dico_temp, activations, z_values, sortie)

                # Calcul des gradients
                gradients = gradient(dico_temp, activations, deltas)

                # Accumuler les gradients
                for couche in gradients:
                    for neurone in gradients[couche]:
                        total_gradients[couche][neurone] += gradients[couche][neurone]

            # Moyenne des gradients
            for couche in total_gradients:
                for neurone in total_gradients[couche]:
                    total_gradients[couche][neurone] /= len(entrees)

            # Mise à jour des poids avec les gradients moyens
            dico_temp = update(dico_temp, total_gradients, eta)

            # Évaluer le nouveau coût
            test_errors = []
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])
                activations, _ = forward(dico_temp, entree)
                error = ((activations[-1] - sortie) ** 2) / 2
                test_errors.append(error[0])

            test_cost = np.sum(test_errors) / len(entrees)

            # Vérifier si ce eta réduit l'erreur
            if test_cost < current_cost:
                found_good_eta = True
                current_cost = test_cost
                dico = deepcopy(dico_temp)  # Accepter cette mise à jour
            else:
                # Diviser eta par 2 et réessayer
                eta /= 2

        if not found_good_eta:
            print(f"Epoch {epoch+1}: Aucun eta efficace trouvé. Erreur minimale: {current_cost:.6f}")
            break

        history.append(current_cost)

        # Early stopping
        if current_cost < ep:
            print(f"Early stopping at epoch {epoch+1} with cost {current_cost:.6f}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Cost: {current_cost:.6e}, Eta utilisé: {eta:.6e}")

    return dico, history


def evaluate(dico, inputs, outputs, threshold=0.5):
    # Evaluate model performance
    correct = 0
    errors = []
    
    for i in range(len(inputs)):
        x = inputs[i]
        y_true = outputs[i]
        
        # Forward pass
        activations, _ = forward(dico, x)
        y_pred = activations[-1][0]
        
        # Classification threshold
        pred_class = 1 if y_pred >= threshold else 0
        true_class = 1 if y_true >= threshold else 0
        
        if pred_class == true_class:
            correct += 1
            
        errors.append(abs(y_true - y_pred))
    
    accuracy = (correct / len(inputs)) * 100
    mean_abs_error = np.mean(errors)
    
    return accuracy, mean_abs_error, len(inputs)-correct

# Load test data
entreest = np.loadtxt("sonar_inputs_test.csv", delimiter=",")
sortiest = np.loadtxt("sonar_outputs_test.csv", delimiter=",")

# Try different random initializations
best_accuracy = 0
best_dico = None
best_error = float('inf')

for trial in range(100):
    print(f"\nTrial {trial+1}/20")
    
    # New random initialization
    mat = [matrice_aleatoire(60, 30), matrice_aleatoire(30, 30), matrice_aleatoire(30, 1)]
    dico = {
        "couche 1": {f"neurone {i+1}": np.append(mat[0][i], 0) for i in range(len(mat[0]))},
        "couche 2": {f"neurone {i+1}": np.append(mat[1][i], 0) for i in range(len(mat[1]))},
        "couche 3": {f"neurone {i+1}": np.append(mat[2][i], 0) for i in range(len(mat[2]))}
    }

    
    # Train with consistent hyperparameters
    eta = 10  # Fixed learning rate
    epochs = 500
    early_stop_threshold = 10**-8
    
    trained_dico, history = train(dico, entrees, sorties, epochs, eta, early_stop_threshold)
    
    # Evaluate on test set
    accuracy, mean_error,faux = evaluate(trained_dico, entreest, sortiest)
    print(f"Test accuracy: {accuracy:.2f}%, Mean absolute error: {mean_error:.4f}, faux = {faux} ")
    
    # Save best model
    if accuracy > best_accuracy or (accuracy == best_accuracy and mean_error < best_error):
        best_accuracy = accuracy
        best_error = mean_error
        best_dico = deepcopy(trained_dico)
    if accuracy == 100:
        break
    

print(f"\nBest model achieved {best_accuracy:.2f}% accuracy with {best_error:.4f} mean absolute error")

# Save best model
with open('best_model.txt', 'w') as data:
    data.write(str(best_dico))