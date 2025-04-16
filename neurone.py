from math import exp
import numpy as np
from copy import deepcopy 
from scipy.special import expit  # Sigmoid function

def matrice_aleatoire(nout, nin):
    '''
    Parameters
    ----------
    nout : Int
        Nombre de sortie
    nin : Int
        Nombre d'entrée

    Returns
    -------
    Array
        Matrice d'éléments tirés aléatoirement suivant une loi uniforme'
    '''
    a = -np.sqrt(6 / (nin + nout))
    b = np.sqrt(6 / (nin + nout))
    return np.random.uniform(a, b, (nin, nout))

def activation(x):
    '''
    Parameters
    ----------
    x : Vecteur (array)
        Applique la fonction sigmoid à toute les composantes du vecteur x
    Returns
    -------
    Un vecteur (array) de toute les composantes de x passées par sigmoid
    '''
    return expit(x)

def dactivation(x):
    '''
    Parameters
    ----------
    x : Vecteur (array)
        Applique la fonction dérivée de sigmoid à toute les composantes du vecteur x
    Returns
    -------
    Un vecteur (array) de toute les composantes de x passées par la dérivée de sigmoid
    '''
    return expit(x) * (1 - expit(x))

def Wl(l, dico):
    '''
    Parameters
    ----------
    l : Entier
        Numéro de la couche
    dico : Dict
        Dictionnaire des poids par neurone par couche
    Returns
    -------
    W : array
        Vecteur de dictionnaire par couche des poids (peut être représenté en matrice)
    b : array
        Vecteur de dicionnaire par couche des biais (peut être représenté en matrice)
    '''
    W = np.array([dico[f"couche {l}"][n][:-1] for n in dico[f"couche {l}"]])
    b = np.array([dico[f"couche {l}"][n][-1] for n in dico[f"couche {l}"]])
    return W, b

def passe_avant(dico, entree):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des couches
    entree : Array
        Vecteur contenant les différentes entrées
    Returns
    -------
    activations : List
        Liste des différentes sorties par couche passées par la fonction d'activation
    z_values : List
        Liste des différentes sorties par couche
    '''
    activations = [entree]
    z_values = []
    
    for l in range(1, 4):
        W, b = Wl(l, dico)
        z = np.dot(W, activations[-1]) + b
        z_values.append(z)
        a = activation(z)
        activations.append(a)
    
    return activations, z_values

def retropropagation(dico, activations, z_values, sortie):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    activations : List
        Liste des différentes sorties par couche passées par la fonction d'activation
    z_values : List
        Liste des différentes sorties par couche
    sortie : Array
        Vecteur des différentes sorties attendues
    Returns
    -------
    deltas : Dict
        Dictionnaire des deltas par couche
    '''
    
    deltas = {}
    L = len(dico)
    
    #Calcul de l'erreur à la dernière couche
    derniere_erreur = (activations[-1] - sortie)
    deltas[f"couche {L}"] = derniere_erreur * dactivation(z_values[-1])
    
    #Application de la rétro-propagation
    for l in range(L-1, 0, -1):
        W, _ = Wl(l+1, dico)
        dz = np.dot(W.T, deltas[f"couche {l+1}"]) * dactivation(z_values[l-1])
        deltas[f"couche {l}"] = dz
        
    return deltas

def gradient(dico, activations, deltas):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    activations : List
        Liste des différentes sorties par couche passées par la fonction d'activation
    deltas : Dict
        Dictionnaire des deltas par couche
    Returns
    -------
    gradients : Dict
        Dictionnaire de dictionnaire des dérivées partielles de l'erreur par rapport aux poids par neurone par couche
    '''
    gradients = deepcopy(dico)
    for l in range(1, len(dico)+1):
        couche = f"couche {l}"
        for i, neurone in enumerate(dico[couche]):
            #Dérivée par rapport au poids
            gradients[couche][neurone][:-1] = deltas[couche][i] * activations[l-1]
            #Dérivée par rapport au biais
            gradients[couche][neurone][-1] = deltas[couche][i]
                
    return gradients

def update(dico,gradients,eta):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    gradients : Dict
        Dictionnaire de dictionnaire de la somme dérivées partielles de l'erreur des différentes sorties par rapport aux poids par neurone par couche 
    eta : Float
        Facteur devant le gradient
    Returns
    -------
    Le nouveau dictionnaire des poids par neurone par couche avec les poids mis à jour
    '''

def update(dico, gradients, eta):
    # Update weights and biases using gradients
    dicos = deepcopy(dico)
    for couche in dico:
        for neurone in dico[couche]:
            dicos[couche][neurone] -= eta * gradients[couche][neurone]
    return dicos

def train(dico, entrees, sorties, epochs, eta, ep):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    entrees : Array
        Lot de plusieurs entrées de test
    sorties : Array
        Lot de plusieurs sorties de test
    epochs : Int
        Nombre d'époque maximal
    eta : Float
        Facteur devant le gradient
    ep : Float
        Critère d'arrêt
    Returns
    -------
    dico : Dict
        Dictionnaire des poids par neurone par couche après entrainement
    history : List
        Liste des différents J qu'on a pu avoir par époque
    '''
    history = []
    et = eta
    for epoch in range(epochs):
        erreurs = []
        gradients_sum = deepcopy(dico)
        for couche in gradients_sum:
            for neurone in gradients_sum[couche]:
                gradients_sum[couche][neurone] = np.zeros_like(gradients_sum[couche][neurone])
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            
            # Passe avant
            activations, z_values = passe_avant(dico, entree)
            
            # Calcul de l'erreur
            erreur = ((activations[-1] - sortie) ** 2) / 2
            erreurs.append(erreur[0])
            
            # Rétro-propagation
            deltas = retropropagation(dico, activations, z_values, sortie)
            
            # Update des poids
            gradients = gradient(dico, activations, deltas)
            for couche in gradients_sum:
                for neurone in gradients_sum[couche]:
                    gradients_sum[couche][neurone] += gradients[couche][neurone]/(len(entrees))
        aerreurs = []
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            activations, _ = passe_avant(dico, entree)
            erreur = ((activations[-1] - sortie) ** 2) / 2
            aerreurs.append(erreur[0])
        
        acout = np.sum(aerreurs) / len(entrees)
        etat = et*10**4
        while etat>=10**-20:
            dicos = deepcopy(dico)
            dicos = update(dico, gradients_sum,etat)
            terreurs = []
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])
                activations, _ = passe_avant(dicos, entree)
                erreur = ((activations[-1] - sortie) ** 2) / 2
                terreurs.append(erreur[0])
            tcout = np.sum(terreurs) / len(entrees)
            print(dicos==dicod)
            if tcout<=acout: 
                et = etat
                break
            else: etat = etat/2
        dico = update(dico, gradients_sum, et)
            
        moy_erreur = np.mean(erreurs)
        history.append(moy_erreur)
        
        # Critère d'arrêt
        if moy_erreur < ep:
            print(f"On arrête par critère d'arrêt à l'époque : {epoch+1} avec une erreur de : {moy_erreur:.6f}")
            break
            
        if (epoch + 1) % 100 == 0:
            pass
    print(f"On arrète à l'époque : {epoch+1} sans critère d'arrêt avec une erreur de : {moy_erreur:.6f}")
    return dico, history


def train_egd(dico, entrees, sorties, epochs, eta, ep):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    entrees : Array
        Lot de plusieurs entrées de test
    sorties : Array
        Lot de plusieurs sorties de test
    epochs : Int
        Nombre d'époque maximal
    eta : Float
        Facteur devant le gradient
    ep : Float
        Critère d'arrêt
    Returns
    -------
    dico : Dict
        Dictionnaire des poids par neurone par couche après entrainement avec la méthode EGD
    history : List
        Liste des différents J qu'on a pu avoir par époque
    '''
    history = []
    etat = eta
    for epoch in range(epochs):
        erreurs = []
        gradients_sum = deepcopy(dico)
        for couche in gradients_sum:
            for neurone in gradients_sum[couche]:
                gradients_sum[couche][neurone] = np.zeros_like(gradients_sum[couche][neurone])
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            
            # Passe avant
            activations, z_values = passe_avant(dico, entree)
            
            # Calcul de l'ereur
            erreur = ((activations[-1] - sortie) ** 2) / 2
            erreurs.append(erreur[0])
            
            # Retro-propagation
            deltas = retropropagation(dico, activations, z_values, sortie)
            
            # Update des poids
            
            gradients = gradient(dico, activations, deltas)
            for couche in gradients_sum:
                for neurone in gradients_sum[couche]:
                    gradients_sum[couche][neurone] += gradients[couche][neurone]/(len(entrees))
            
        moy_erreur = np.mean(erreurs)
        trouve = False
        et = etat*10**3
        while trouve == False and et>10**-30:
            dicos = update(dico, gradients_sum, et)
            erreurst = []
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])
                activationst, z_values = passe_avant(dicos, entree)
                erreur = ((activationst[-1] - sortie) ** 2) / 2
                erreurst.append(erreur[0])
            moy_erreurt = np.mean(erreurst)
            if moy_erreurt< moy_erreur:
                dico = update(dico, gradients_sum, et)
                etat = et
                trouve = True
            else: 
                et= et/2
        
        history.append(moy_erreur)
        
        # Critère d'arrêt
        if moy_erreur < ep:
            print(f"On arrête par critère d'arrêt à l'époque : {epoch+1} avec une erreur de : {moy_erreur:.6f}")
            
            break
            
        if (epoch + 1) % 100 == 0:
            pass
    
    return dico, history



# Chargement des données d'entrainement
entrees = np.loadtxt("sonar_inputs_train.csv", delimiter=",")
sorties = np.loadtxt("sonar_outputs_train.csv", delimiter=",")

# Chargement des données de test
entreest = np.loadtxt("sonar_inputs_test.csv", delimiter=",")
sortiest = np.loadtxt("sonar_outputs_test.csv", delimiter=",")

lstt = []
lstx = []

for trial in range(0,200):
    mat = [matrice_aleatoire(60, 30),matrice_aleatoire(30, 30),matrice_aleatoire(30, 1)]
    dicod = {
        "couche 1": {f"neurone {i+1}": np.append(mat[0][i], 0) for i in range(len(mat[0]))},
        "couche 2": {f"neurone {i+1}": np.append(mat[1][i], 0) for i in range(len(mat[1]))},
        "couche 3": {f"neurone {i+1}": np.append(mat[2][i], 0) for i in range(len(mat[2]))}
    }
    
    print(f"\nTrial {trial+1}/20")
    lstx.append(trial)
    
    # Nouvelle initialisation aléatoire
    dico = deepcopy(dicod)    
    trained_dico, history = train_egd(dico, entrees, sorties, epochs=900, eta=10, ep=1e-8)
