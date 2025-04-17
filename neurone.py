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
    val_x : List
        Liste des différentes sorties par couche passées par la fonction d'activation
    val_z : List
        Liste des différentes sorties par couche
    '''
    val_x = [entree]
    val_z = []
    
    for l in range(1, 4):
        W, b = Wl(l, dico)
        z = np.dot(W, val_x[-1]) + b
        val_z.append(z)
        a = activation(z)
        val_x.append(a)
    
    return val_x, val_z

def retropropagation(dico, val_x, val_z, sortie):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    val_x : List
        Liste des différentes sorties par couche passées par la fonction d'activation
    val_z : List
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
    derniere_erreur = (val_x[-1] - sortie)
    deltas[f"couche {L}"] = derniere_erreur * dactivation(val_z[-1])
    
    #Application de la rétro-propagation
    for l in range(L-1, 0, -1):
        W, _ = Wl(l+1, dico)
        dz = np.dot(W.T, deltas[f"couche {l+1}"]) * dactivation(val_z[l-1])
        deltas[f"couche {l}"] = dz
        
    return deltas

def gradient(dico, val_x, deltas):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    val_x : List
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
            gradients[couche][neurone][:-1] = deltas[couche][i] * val_x[l-1]
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
    
    dicos = deepcopy(dico)
    for couche in dico:
        for neurone in dico[couche]:
            dicos[couche][neurone] -= eta * gradients[couche][neurone]
    return dicos

def entrainement_gd(dico, entrees, sorties, epoques, eta, ep):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    entrees : Array
        Lot de plusieurs entrées de test
    sorties : Array
        Lot de plusieurs sorties de test
    epoques : Int
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
    for epoque in range(epoques):
        erreurs = []
        gradients_somme = deepcopy(dico)
        for couche in gradients_somme:
            for neurone in gradients_somme[couche]:
                gradients_somme[couche][neurone] = np.zeros_like(gradients_somme[couche][neurone])
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            
            # Passe avant
            val_x, val_z = passe_avant(dico, entree)
            
            # Calcul de l'ereur
            erreur = ((val_x[-1] - sortie) ** 2) / 2
            erreurs.append(erreur[0])
            
            # Retro-propagation
            deltas = retropropagation(dico, val_x, val_z, sortie)
            
            # Update des poids
            
            gradients = gradient(dico, val_x, deltas)
            for couche in gradients_somme:
                for neurone in gradients_somme[couche]:
                    gradients_somme[couche][neurone] += gradients[couche][neurone]/(len(entrees))
            
        moy_erreur = np.mean(erreurs)
        dico = update(dico, gradients_somme, eta)
        
        history.append(moy_erreur)
        
        # Critère d'arrêt
        if moy_erreur < ep:
            print(f"On arrête par critère d'arrêt à l'époque : {epoque+1} avec une erreur de : {moy_erreur:.6f}")
            
            break
            
        if (epoque + 1) % 100 == 0:
            print(f"Epoque {epoque}/{epoques}")
    return dico, history


def entrainement_egd(dico, entrees, sorties, epoques, eta, ep):
    '''
    Parameters
    ----------
    dico : Dict
        Dictionnaire des poids par neurone par couche
    entrees : Array
        Lot de plusieurs entrées de test
    sorties : Array
        Lot de plusieurs sorties de test
    epoques : Int
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
    for epoque in range(epoques):
        erreurs = []
        gradients_somme = deepcopy(dico)
        for couche in gradients_somme:
            for neurone in gradients_somme[couche]:
                gradients_somme[couche][neurone] = np.zeros_like(gradients_somme[couche][neurone])
        for i in range(len(entrees)):
            entree = entrees[i]
            sortie = np.array([sorties[i]])
            
            # Passe avant
            val_x, val_z = passe_avant(dico, entree)
            
            # Calcul de l'ereur
            erreur = ((val_x[-1] - sortie) ** 2) / 2
            erreurs.append(erreur[0])
            
            # Retro-propagation
            deltas = retropropagation(dico, val_x, val_z, sortie)
            
            # Update des poids
            
            gradients = gradient(dico, val_x, deltas)
            for couche in gradients_somme:
                for neurone in gradients_somme[couche]:
                    gradients_somme[couche][neurone] += gradients[couche][neurone]/(len(entrees))
            
        moy_erreur = np.mean(erreurs)
        trouve = False
        et = etat*10**3
        while trouve == False and et>10**-30:
            dicos = update(dico, gradients_somme, et)
            erreurst = []
            for i in range(len(entrees)):
                entree = entrees[i]
                sortie = np.array([sorties[i]])
                val_xt, val_z = passe_avant(dicos, entree)
                erreur = ((val_xt[-1] - sortie) ** 2) / 2
                erreurst.append(erreur[0])
            moy_erreurt = np.mean(erreurst)
            if moy_erreurt< moy_erreur:
                dico = update(dico, gradients_somme, et)
                etat = et
                trouve = True
            else: 
                et= et/2
        
        history.append(moy_erreur)
        
        # Critère d'arrêt
        if moy_erreur < ep:
            print(f"On arrête par critère d'arrêt à l'époque : {epoque+1} avec une erreur de : {moy_erreur:.6f}")
            
            break
            
        if (epoque + 1) % 100 == 0:
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

for essai in range(0,200):
    mat = [matrice_aleatoire(60, 30),matrice_aleatoire(30, 30),matrice_aleatoire(30, 1)]
    dicod = {
        "couche 1": {f"neurone {i+1}": np.append(mat[0][i], 0) for i in range(len(mat[0]))},
        "couche 2": {f"neurone {i+1}": np.append(mat[1][i], 0) for i in range(len(mat[1]))},
        "couche 3": {f"neurone {i+1}": np.append(mat[2][i], 0) for i in range(len(mat[2]))}
    }
    
    print(f"\nessai {essai+1}/20")
    lstx.append(essai)
    
    # Nouvelle initialisation aléatoire
    dico = deepcopy(dicod)    
    trained_dico, history = entrainement_egd(dico, entrees, sorties, epoques=900, eta=10, ep=1e-8)
