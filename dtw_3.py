import math
import scipy
import sklearn
from collections import defaultdict
import librosa
import sys
import numpy as np
# from google.colab import drive

# norm of two vector of the same size
def norm( A, B ):
    s = 0 
    for i in range( len(A) ):
        s += ( A[i] - B[i] )**2
    return math.sqrt( s )
 
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

def compare2samples( sample1, sample2 ):
    y1, sr1 = librosa.load( sample1 )
    y2, sr2 = librosa.load( sample2 )
    
    mfcct = np.array( librosa.feature.mfcc( y=y1, sr=sr1, hop_length=1024, htk=True, n_mfcc=12 ) ).transpose()
    mfcca = np.array( librosa.feature.mfcc( y=y2, sr=sr2, hop_length=1024, htk=True, n_mfcc=12 ) ).transpose()

    dist = dtw( mfcct, mfcca, mfcct.shape[0], mfcca.shape[0], norm ) 
    return dist

# Compare la base de test avec la base d'apprentisage pour trouver
# l'audio le plus similaire
def compare_sample_with_BT( sample, baseTest ):
    distPerFile = list()
    for f in baseTest:
        if (sample != f):
            distPerFile.append( compare2samples( sample, f ) )
        else:
            distPerFile.append(999);
   
    for i in range(len(baseTest)):
        print( baseTest[i], distPerFile[i], "\n" )

    index = distPerFile.index( min( distPerFile ) ) 
    return baseTest[index].split('/')[-1]

# Construit une liste avec un fichier contenant le 
# nom des fichiers de la Base d'apprentissage.
def BA_file2list( inputBA ):
    location = "data/bruite/" # En cas ou les fichiers ne soient dans le même repertoire que dtw.py 
    with open( inputBA, "r" ) as ba:
        tmp = ba.readlines();
    tmp = [x.strip() for x in tmp]
   
    ba_list = list()
    for x in tmp:
        ba_list.append( location + x )
    ba.close()
    return ba_list 

# ------------------------------------------------------------------------------------- # 

if ( len( sys.argv ) != 3 ):
    print( "Usage: $python3 %s <file1.wav> <base_d'aprentisage>" % sys.argv[0] )
    exit( -1 )

print( "Comparing", sys.argv[1].split('/')[-1], "avec base d'apprentisage ", sys.argv[2].split('/')[-1] )

ba_list =  BA_file2list( sys.argv[2] )
print( compare_sample_with_BT( sys.argv[1], ba_list ) )

# Exemple d'exécution :
# $ python3 dtw_3.py data/bruite/M02_arretetoi.wav M02_bruite_BA.txt 
# Le M02_bruite_BA.txt est la base de test et M02_arretetoi.wav est l'audio à comparer 
# Il faut modifier  l'atribut location avec le repertoire ou se trouvent les fichiers de la base de test. Dans mon cas je les ai placé dans "data/bruite/"

# TODO ajouter un nouveau champ sur le fichier M02_bruite_BA.txt contenant la localisation des fichiers.


