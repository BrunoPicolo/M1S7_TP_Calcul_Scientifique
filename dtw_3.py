import math
import scipy
import sklearn
from collections import defaultdict
import librosa
import sys
import numpy as np
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
    
    g = np.zeros( (I+1, J+1) )
    for j in range( 1, J+1 ):
        g[0][j] = float("inf")
    
    for i in range( 1, I+1 ):
        g[i][0] = float("inf")
        for j in range( 1, J+1 ):
            d = x( A[i-1], B[j-1] )
            g[i][j] = min( g[i-1][j] + w0*d,
                           g[i-1][j-1] + w1*d,    
                           g[i][j-1] + w2*d )
    return g[I-1][J-1] / ( I+J )

# np.linalg.norm( fi, fg )
def compare_mfcc( tFile, aFile ):
    y1, sr1 = librosa.load( tFile )
    y2, sr2 = librosa.load( aFile )
    
    mfcc1 = np.array( librosa.feature.mfcc( y=y1, sr=sr1, hop_length=1024, htk=True, n_mfcc=12 ) ) 
    mfcc2 = np.array( librosa.feature.mfcc( y=y2, sr=sr2, hop_length=1024, htk=True, n_mfcc=12 ) )

    
# TODO le contenu de dist est peut être pas bon 
# ------------------------------------------------------------------------------------- # 


A = [ -2, 10, -10, 15, -13, 20, -5, 14, 2 ]
B = [ 3, -13, 14, -7, 9, -2 ]

s1 = [ 'X', 'C', 'U' ]
s2 = [ 'X', 'C', 'U' ]

obs = [ 'X', 'X', 'V', 'U', 'X', 'C', 'X']
cinq = np.array( [ 'X', 'V', 'V','C','X'] )
cent = np.array( [ 'X', 'V', 'V'] )
vingth = ['Ux', 'V', 'V']

z =  dtw( B, A, 6, 9, euclidienne)
x1 = dtw( cinq, obs,  len(cinq), len(obs), sonsTD )
x2 = dtw( cent, obs,  len(cent), len(obs), sonsTD )
x3 = dtw( vingth, obs,  len(vingth),len(obs), sonsTD )

print(z, x1, x2, x3)

if ( len( sys.argv ) != 3 ):
    print( "Usage: $python3 %s <file1.wav> <file2.wav>" % sys.argv[0] )
    exit( -1 )
    
print( "Comparing", sys.argv[1], sys.argv[2] )
compare_mfcc( sys.argv[1], sys.argv[2] )
	

## Rapport : differents ponderations, couts ...
## scénarios à comparer:
#	-> bruite - nonbruite ( homme )
#	-> femme - femme
# 	-> homme - homme
# 	-> home - femme

# compare est une fonction de comparaison 
# remplacer librosa.load et décomenter l'import de google colab
# Ex y, sr = librosa.load("corpus/dronevolant_bruite/M01_arretetoi.wav", offset=30, duration=5)
# https://stackoverflow.com/questions/6932096/matching-two-series-of-mfcc-coefficients
# https://stackoverflow.com/questions/8433401/defining-a-matrix-norm-to-compare-two-mfcc-matrices

