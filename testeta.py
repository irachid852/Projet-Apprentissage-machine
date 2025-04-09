#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 22:25:19 2025

@author: irachid
"""


### 1er neurone et carte de sensibilité

import numpy as np
from random import randint, random
import matplotlib.pyplot as plt

def sigma(x):
    return x**2 - 1

def sigmaprime(x):
    return 2 * x


X = [0, 1]
Ysol = [0, 0]

eta = 0.001

def jerreur(theta):
    return 0.25 * (sigma(theta[0] * X[0] + theta[1])**2 + sigma(theta[0] * X[1] + theta[1])**2)

def gradient(theta):
    w, b = theta
    return [0.5 * sigma(w + b) * sigmaprime(w + b), 0.5 * (sigma(b) * sigmaprime(b) + sigma(w + b) * sigmaprime(w + b))]

def pasfixe(theta):
    n = 0
    et = eta
    while n < 300:
        n += 1
        if jerreur(theta) < 10**-15:
            #print("on converge")
            break
        etaf = float(et*10**3)
        grad = gradient(theta)
        tet = [theta[0],theta[1]]
        tet[0] -= etaf * grad[0]
        tet[1] -= etaf * grad[1]
        #print(jerreur(tet))
        #print(jerreur(theta)  < jerreur(tet))
        g = 0
        while jerreur(theta)  < jerreur(tet) and etaf >10**-100:
            etaf = etaf/2
            g+=1
            #print('boom')
            tet[0] =theta[0] -etaf * grad[0]
            tet[1] = theta[1] -etaf * grad[1]
        #print(g)
        et = float(etaf)
        #print(et)
        theta[0] -= et * grad[0]
        theta[1] -= et * grad[1]
    return theta


listetheta = []
lst = [[3 * random() * (-1)**randint(0, 1), 3 * random() * (-1)**randint(0, 1)] for i in range(100000)]
nb = 0
for i in lst:
    nb += 1
    if nb % 50 == 0:
        print(nb)
    theta = i[:]
    CalculTheta = pasfixe(theta)
    listetheta.append(CalculTheta)

print("\nLa liste de tous les theta obtenus doit comprendre des couples proches de [0,1] ou [0,-1] ou [-2,1] ou [-1,2] car valeurs de convergence\n")
for i in range(len(listetheta)):
    for j in range(2):
        listetheta[i][j] = round(listetheta[i][j], 2)
print(listetheta)

plt.plot(3, 3, 'bo', label="[-2,1]")
plt.plot(3, 3, 'co', label="[2,-1]")
plt.plot(3, 3, 'go', label="[0,1]")
plt.plot(3, 3, 'ro', label="pas convergent")
plt.plot(3, 3, 'mo', label="[0,1]")
plt.xlabel('w')
plt.ylabel('b')

for i in range(len(lst)):
    if [int(listetheta[i][0]), int(listetheta[i][1])] in [[-2, 1]]:
        plt.plot(lst[i][0], lst[i][1], "bo")
    elif [int(listetheta[i][0]), int(listetheta[i][1])] in [[2, -1]]:
        plt.plot(lst[i][0], lst[i][1], "co")
    elif [int(listetheta[i][0]), int(listetheta[i][1])] in [[0, -1]]:
        plt.plot(lst[i][0], lst[i][1], "go")
    elif [int(listetheta[i][0]), int(listetheta[i][1])] in [[0, 1]]:
        plt.plot(lst[i][0], lst[i][1], "mo")
    elif int(listetheta[i][0])**2 + int(listetheta[i][1])**2 >= 0:
        plt.plot(lst[i][0], lst[i][1], "ro")

plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.title('Cartes de sensibilité g(x) = x**2 - 1')
plt.show()
