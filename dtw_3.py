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

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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

    reduct1 = average_mfcc(mfcc1)
    reduct2 = average_mfcc(mfcc2)

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

# ------------------------------------------- PARTIE 3 ------------------------------------------ # 

def average_mfcc(mfcc):
    a = 0
    n = list()
    for x in mfcc:
        n.append(sum(x)/len(x))
    return n

def calculate_principal_axes_BA(baseA):
    listReductionsBaseA = list()
    I = len(baseA)
    for i in range (I):
        y, sr = librosa.load(baseA[i])
        mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True, n_mfcc=12)).transpose()

    pca = PCA(n_components=12,svd_solver='full')
    pca.fit(mfcc)

    list_vp = pca.singular_values_
    list_vp = sorted(list_vp,reverse=True)
    return list_vp[0:3]

def part3Test (baseA):
	print(calculate_principal_axes_BA(baseA))

# ------------------------------------------------------------------------------------- # 

def compare2samples(sample1, sample2):
    y1, sr1 = librosa.load(sample1)
    y2, sr2 = librosa.load(sample2)
    
    mfcc1 = np.array(librosa.feature.mfcc(y=y1, sr=sr1, hop_length=1024, htk=True, n_mfcc=12)).transpose()
    mfcc2 = np.array(librosa.feature.mfcc(y=y2, sr=sr2, hop_length=1024, htk=True, n_mfcc=12)).transpose()

    reduct1 = average_mfcc(mfcc1)
    reduct2 = average_mfcc(mfcc2)

    dist = dtw(mfcc1, mfcc2, mfcc1.shape[0], mfcc2.shape[0], norm) 
    return dist

# --------------------------------------------- MAIN --------------------------------------------- # 
def mainPartie2():
	matrix = compare_BT_BA(bt_list, ba_list)
	(index, comparations)= find_best_comparation_per_file(bt_list, ba_list, matrix)

	for i in range(len(comparations)):
		print( bt_list[i].split('/')[-1][:-4], "\t\t\t", comparations[i].split('/')[-1][:-4] )

	tab = [x for x in range(len(ba_list))]
	confusion_m(tab, index)

def mainPartie3():
	part3Test(ba_list)

# MAIN 
if (len(sys.argv) != 2):
    print("Usage: $python3 %s ./corpus/FichierTest/<Nomfichier>" % sys.argv[0])
    exit(-1)

print("Comparing base d'apprentisage baseDonnee_BA avec base de test ", sys.argv[1].split('/')[-1])


ba_list = file2list("corpus/BaseA/","./corpus/baseDonnee_BA")
bt_list = file2list("corpus/BaseT/", sys.argv[1]) 
mainPartie3()


