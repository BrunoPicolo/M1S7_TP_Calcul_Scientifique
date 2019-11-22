import math
import scipy
import sklearn
from collections import defaultdict
import librosa
import sys
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

"""
Act as a constant in Python3.
@return : a constant number equals to the number of words we will work on
"""
def NBMOTS():
  return 13

"""
Act as three constants in Python3.
@return : coefficients (weight) applied to the top, left and diagonal distances
          in the DTW matrix
"""
def coeffT():
  return 1
def coeffL():
  return 1
def coeffD():
  return 2

"""
Calculates and returns the norm of two vectors of the same size.
@param vectorA : a vector of n numbers
@param vectorB : a vector of n numbers
@return : norm of the two vectors
"""
def norm(vectorA, vectorB):
    s = 0
    for i in range(len(vectorA)):
        s += (vectorA[i] - vectorB[i])**2
    return math.sqrt(s)

"""
Algorithme Dynamic Time Warping
@param vectorA : first sequence of numbers
@param vectorB : second sequence of numbers
@param coeffL : left weight
@param coeffT : top weight
@param coeffD : diagonal weight
@param distance_function : function used to calculate the distance between an
                          element of the vectorA and a element of the vectorB
@return : a scalar that represents how different the vectorA and vectorB are
"""
def dtw(vectorA, vectorB, distance_function):
    sizeA = vectorA.shape[0]
    sizeB = vectorB.shape[0]

    mat = np.zeros((sizeA+1, sizeB+1))
    for j in range(1, sizeB+1):
        mat[0][j] = float("inf")

    for i in range(1, sizeA+1):
        mat[i][0] = float("inf")
        for j in range(1, sizeB+1):
            d = distance_function(vectorA[i-1], vectorB[j-1])
            mat[i][j] = min(mat[i-1][j] + coeffT()*d,
                           mat[i-1][j-1] + coeffD()*d,
                           mat[i][j-1] + coeffL()*d)
    return mat[sizeA-1][sizeB-1] / (sizeA+sizeB)


"""
Transform two audio files in vectors and return how different the
two audio files are.
@param audioFileA : first audio
@param audioFileB : second audio
@return : a scalar that represents how different the audioFileA and audioFileB are
"""
def compare2audio(audioFileA, audioFileB):
    yA, srA = librosa.load(audioFileA)
    yB, srB = librosa.load(audioFileB)

    mfccA = np.array(librosa.feature.mfcc(y=yA, sr=srA, hop_length=1024, htk=True, n_mfcc=12)).transpose()
    mfccB = np.array(librosa.feature.mfcc(y=yB, sr=srB, hop_length=1024, htk=True, n_mfcc=12)).transpose()

    return dtw(mfccA, mfccB, norm)

"""
Calculates a scalar for each audio file of baseT associated to every audio
file from baseA and returns the corresponding matrix
@param baseT : a vector of audio file names in the test base
@param baseA : a vector of audio file names in the learning base
@return : a matrix that represents how every audio file from baseT is different
from every audio file from baseA
"""
def compare_BT_BA(baseT, baseA):
    nbAudioTest = len(baseT)
    nbAudioAppr = len(baseA)

    compareMatrix = np.empty((nbAudioTest, nbAudioAppr), dtype=float)
    for i in range(nbAudioTest):
        for j in range(nbAudioAppr):
            compareMatrix[i][j] = compare2audio(baseT[i], baseA[j])

    return compareMatrix

"""
Find for each audio file from baseT the less different (best recognition)
audio file from baseA based on compareMatrix
@param baseT : a vector of audio file names in the test base
@param baseA : a vector of audio file names in the learning base
@param compareMatrix : comparison matrix
@return : the recognized order and the nationality of the recognized order
from baseA for each order of the baseT
"""
def find_best_comparisions(baseT, baseA, compareMatrix):
    listBestComp = list()
    listIndexBestComp = list()

    for i in range(len(baseT)):
        line = list(compareMatrix[i])
        indexBestComp = line.index(min(compareMatrix[i]))
        listIndexBestComp.append(indexBestComp%NBMOTS())
        listBestComp.append(baseA[indexBestComp])

    return (listIndexBestComp, listBestComp)

"""
Display the confusion matrix and an accuracy score.
@param actual : a vector of the orders it should recognize
@param predicted : a vector of recognized orders
"""
def confusion_m(actual, predicted):
    results = confusion_matrix(actual, predicted)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :', accuracy_score(actual, predicted))
    print('Report : ')
    print(classification_report(actual, predicted))

"""
Get a list of audio file names from a specified location
@param location : path of the audio file base
@param inputB : path of the file containing the audio file names
@return : a list of paths to audio file names
"""
def file2list(location, inputB):
    with open(inputB, "r") as b:
        tmp = b.readlines();
    b.close()
    tmp = [x.strip() for x in tmp]

    b_list = list()
    for x in tmp:
        b_list.append(location + x)

    return b_list

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

def mainPartie3():
	part3Test(ba_list)

# --------------------------------------------- MAIN --------------------------------------------- #
#Begin
if (len(sys.argv) != 2):
    print("Usage: $python3 %s ./corpus/FichierTest/<Nomfichier>" % sys.argv[0])
    exit(-1)

print("Comparing base d'apprentisage baseDonnee_BA avec base de test ", sys.argv[1].split('/')[-1])


ba_list = file2list("corpus/BaseA/","./corpus/baseDonnee_BA")
bt_list = file2list("corpus/BaseT/", sys.argv[1])

matrix = compare_BT_BA(bt_list, ba_list)
(index, comparations)= find_best_comparisions(bt_list, ba_list, matrix)

#for i in range(len(comparations)):
#    print( bt_list[i].split('/')[-1][:-4], "\t\t\t", comparations[i].split('/')[-1][:-4] )

tab = [x for x in range(NBMOTS())]
#print(index)
confusion_m(tab, index)
#End
