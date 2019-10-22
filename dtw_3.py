import numpy as np
import math
import scipy
import sklearn
from collections import defaultdict
import librosa
# from google.colab import drive

def euclidienne( a, b ):
    return abs( a - b )

def lettres( a, b ):
    return 0 if a==b else 1

def sons( a, b, d ):
    dsons = np.array( [[0,1,1,1,2],
                       [1,0,1,1,2],
                       [1,1,0,1,1],
                       [1,1,1,0,2],
                       [2,2,1,2,0]] )
    return ( dsons[d[a]][d[b]] )
 
def sonsTD (a, b):
    d = { 'X':0, 'C':1, 'U':2, 'Ux':3, 'V':4 }
    return sons(a, b, d)
 
def dtw( A, B, I, J, x ):
    w0 = 1
    w1 = 2
    w2 = 1
    
    g = np.zeros( (I, J) )
    for j in range( 1, J ):
        g[0][j] = float("inf")
    
    for i in range( 1, I ):
        g[i][0] = float("inf")
        for j in range( 1, J ):
            d = x( A[i], B[j] )
            g[i][j] = min( g[i-1][j] + w0*d,
                           g[i-1][j-1] + w1*d,    
                           g[i][j-1] + w2*d )
    print( g )
    return g[I-1][J-1] / ( I+J )

  
def main():
    A = [ -2, 10, -10, 15, -13, 20, -5, 14, 2 ]
    B = [ 3, -13, 14, -7, 9, -2 ]

    s1 = [ 'X', 'C', 'U' ]
    s2 = [ 'X', 'C', 'U' ]
    
    obs = [ 'X', 'X', 'V', 'U', 'X', 'C', 'X']
    cinq = [ 'X', 'V', 'V','C','X']
    cent = [ 'X', 'V', 'V']
    vingth = ['Ux', 'V', 'V']

    z =  dtw( A, B, 5, 5, euclidienne)
    x1 = dtw( obs, cinq, len(obs), len(cinq), sonsTD )
    x2 = dtw( obs, cent, len(obs), len(cent), sonsTD )
    x3 = dtw( obs, cent, len(obs), len(vingth), sonsTD )

    print(z, x1, x2, x3)   
    
    # y, sr = librosa.load("corpus/dronevolant_bruite/M01_arretetoi.wav", offset=30, duration=5)
    y, sr = librosa.load("M01_arretetoi.wav")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True, n_mfcc=12)
    print( y, mfcc )
    
main()

## Rapport : differents ponderations, couts ...

