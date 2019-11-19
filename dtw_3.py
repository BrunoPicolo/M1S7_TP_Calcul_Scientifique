import math
import scipy
import sklearn
from collections import defaultdict
import librosa
import sys
import numpy as np
# from google.colab import drive
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

def norm(A, B):
    """
        norm of two vector of the same size
    """
    s = 0 
    for i in range(len(A)):
        s += (A[i] - B[i])**2
    return math.sqrt(s)

# ------------------------------------------------------------------------------------- # 
 
def dtw(A, B, I, J, x):
    w0 = 1
    w1 = 2
    w2 = 1
    
    g = np.zeros((I+1, J+1))
    for j in range(1, J+1):
        g[0][j] = float("inf")
    
    for i in range(1, I+1):
        g[i][0] = float("inf")
        for j in range(1, J+1):
            d = x(A[i-1], B[j-1])
            g[i][j] = min(g[i-1][j] + w0*d,
                           g[i-1][j-1] + w1*d,    
                           g[i][j-1] + w2*d)
    return g[I-1][J-1] / (I+J)

# ------------------------------------------------------------------------------------- # 

def compare2samples(sample1, sample2):
    y1, sr1 = librosa.load(sample1)
    y2, sr2 = librosa.load(sample2)
    
    mfcct = np.array(librosa.feature.mfcc(y=y1, sr=sr1, hop_length=1024, htk=True, n_mfcc=12)).transpose()
    mfcca = np.array(librosa.feature.mfcc(y=y2, sr=sr2, hop_length=1024, htk=True, n_mfcc=12)).transpose()

    dist = dtw(mfcct, mfcca, mfcct.shape[0], mfcca.shape[0], norm) 
    return dist

# ------------------------------------------------------------------------------------- # 

def compare_sample_with_BT(sample, baseTest):
    """ 
        Compare un seul sample avec une base d'apprentissage dont
        le but de trouver le correspondant le plus similaires.
    """
    distPerFile = list()
    for f in baseTest:
        if (sample != f):
            distPerFile.append(compare2samples(sample, f))
        else:
            distPerFile.append(999);
   
    for i in range(len(baseTest)):
        print(baseTest[i], distPerFile[i], "\n")

    index = distPerFile.index(min(distPerFile)) 
    return baseTest[index].split('/')[-1]

# ------------------------------------------------------------------------------------- # 

def compare_BA_BT(baseA, baseT):
    """ 
        Compare une base de test avec une base d'apprentissage.
    """
    I = len(baseA)
    J = len(baseT)

    compareMatrix = np.empty((I, J), dtype=float)
    for i in range(I):
        for j in range(J):
            compareMatrix[i][j] = compare2samples(baseA[i], baseT[j]) 
    
    return compareMatrix

# ------------------------------------------------------------------------------------- # 

def find_best_comparation_per_file(baseA, baseT, compareMatrix):
    """ 
        Trouve la meilleur comparaison pour chaque fichier de la base
        de test avec la base d'apprentissage.
    """
   
    listBestComp = list()
    line = list()

    for i in range(len(baseA)):
        line = list(compareMatrix[i])
        indexBestComp = line.index(min(compareMatrix[i])) 
        listBestComp.append(baseT[indexBestComp])

    return listBestComp

# ------------------------------------------------------------------------------------- # 
    
def naif(baseA, bestCompBaseT):
    results = confusion_matrix(baseA, bestCompBaseT)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :',accuracy_score(baseA, bestCompBaseT))
    print('Report : ')
    print(classification_report(baseA, bestCompBaseT))


# ------------------------------------------------------------------------------------- # 

# TODO ajouter la localisation dans la première ligne du fichier
def file2list(location, inputBA):
    """
        Construit une liste avec un fichier contenant le 
        nom des fichiers de la Base d'apprentissage.
    """
    with open(inputBA, "r") as ba:
        tmp = ba.readlines();
    tmp = [x.strip() for x in tmp]
   
    ba_list = list()
    for x in tmp:
        ba_list.append(location + x)
    ba.close()
    return ba_list 

# ------------------------------------------------------------------------------------- # 

if (len(sys.argv) != 3):
    print("Usage: $python3 %s <base d'aprentisage> <base de test>" % sys.argv[0])
    exit(-1)

print("Comparing base d'apprentisage", sys.argv[1].split('/')[-1], " avec base de test ", sys.argv[2].split('/')[-1])


ba_list = file2list("data/bruite/", sys.argv[1])
bt_list = file2list("data/bruite/", sys.argv[2]) 
matrix = compare_BA_BT(ba_list, bt_list)
comparations= find_best_comparation_per_file(ba_list, bt_list, matrix)

for i in range(len(comparations)):
    print( ba_list[i], " ", comparations[i], "\n" )

naif(ba_list, comparations)

#########################################################################################

# Exemple d'exécution :
# $ python3 dtw_3.py data/bruite/M02_arretetoi.wav M02_bruite_BA.txt 
# Le M02_bruite_BA.txt est la base de test et M02_arretetoi.wav est l'audio à comparer 
# Il faut modifier  l'atribut location avec le repertoire ou se trouvent les fichiers de la base de test. Dans mon cas je les ai placé dans "data/bruite/"

# https://github.com/roadroller2da/sound-recognition/blob/master/svm_mfcc_librosa.py
# https://mc.ai/sound-classification-using-deep-learning/
# https://github.com/chaosparrot/parrot.py/blob/master/lib/machinelearning.py

# Il faut comparer une base de tests et une base source avec une mfcc. -> On trouve une matrice.
# Ensuite on prend le minimum de chaque ligne (de chaque distance) 
# Il faut trouver les fichiers qui correspondent à la case de la distance minimal de chaque ligne -> Deux listes avec les noms des audios.
# Finalement il faut utiliser confusion_matrix avec les deux listes précédentes. 
