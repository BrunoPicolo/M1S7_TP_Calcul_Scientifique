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
    
    mfcc1 = np.array(librosa.feature.mfcc(y=y1, sr=sr1, hop_length=1024, htk=True, n_mfcc=12)).transpose()
    mfcc2 = np.array(librosa.feature.mfcc(y=y2, sr=sr2, hop_length=1024, htk=True, n_mfcc=12)).transpose()

    dist = dtw(mfcc1, mfcc2, mfcc1.shape[0], mfcc2.shape[0], norm) 
    return dist

# ------------------------------------------------------------------------------------- # 

def compare_BT_BA(baseT, baseA):
    """ 
        Compare une base de test avec une base d'apprentissage.
    """
    I = len(baseT)
    J = len(baseA)

    compareMatrix = np.empty((I, J), dtype=float)
    for i in range(I):
        for j in range(J):
            compareMatrix[i][j] = compare2samples(baseT[i], baseA[j]) 
    
    return compareMatrix

# ------------------------------------------------------------------------------------- # 

def find_best_comparation_per_file(baseT, baseA, compareMatrix):
    """ 
        Trouve la meilleur comparaison pour chaque fichier de la base
        de test avec la base d'apprentissage.
    """
   
    listBestComp = list()
    listIndexBestComp = list()

    for i in range(len(baseT)):
        line = list(compareMatrix[i])
        indexBestComp = line.index(min(compareMatrix[i])) 
        listIndexBestComp.append(indexBestComp)
        listBestComp.append(baseA[indexBestComp])

    return (listIndexBestComp, listBestComp)

# ------------------------------------------------------------------------------------- # 
    
def confusion_m(actual, predicted):
    results = confusion_matrix(actual, predicted)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :',accuracy_score(actual, predicted))
    print('Report : ')
    print(classification_report(actual, predicted))


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

# --------------------------------------------- MAIN --------------------------------------------- # 

if (len(sys.argv) != 3):
    print("Usage: $python3 %s <base d'aprentisage> <base de test>" % sys.argv[0])
    exit(-1)

print("Comparing base d'apprentisage", sys.argv[1].split('/')[-1], " avec base de test ", sys.argv[2].split('/')[-1])


ba_list = file2list("corpus/", sys.argv[1]	)
bt_list = file2list("corpus/BaseT/", sys.argv[2]) 
matrix = compare_BT_BA(bt_list, ba_list)
(index, comparations)= find_best_comparation_per_file(bt_list, ba_list, matrix)

for i in range(len(comparations)):
    print( bt_list[i].split('/')[-1][:-4], "\t\t\t", comparations[i].split('/')[-1][:-4] )

tab = [x for x in range(len(ba_list))]
confusion_m(tab, index)

