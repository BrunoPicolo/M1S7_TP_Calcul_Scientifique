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

def compare_mfcc( tFile, aFile ):
    y1, sr1 = librosa.load( tFile )
    y2, sr2 = librosa.load( aFile )
    
    mfcct = np.array( librosa.feature.mfcc( y=y1, sr=sr1, hop_length=1024, htk=True, n_mfcc=12 ) ).transpose()
    mfcca = np.array( librosa.feature.mfcc( y=y2, sr=sr2, hop_length=1024, htk=True, n_mfcc=12 ) ).transpose()

    dist = dtw( mfcct, mfcca, mfcct.shape[0], mfcca.shape[0], norm ) 
    print( dist )

# TODO Verify
def compare_test_with_BA( tFile, aFiles ):

    distPerFile = list()
    for f in aFiles:
        distPerFile.append( compare_mfcc( tFile, f ) )

    print( aFiles.index( min(distPerFile) ) )

# TODO prendre en compte le chemin absolue des fichiers
def BA_file2list( inputBA ):
    with open( inputBA, "r" ) as ba:
        ba_list = ba.readlines();
    ba_list = [x.strip() for x in ba_list]
    ba.close()

    return ba_list


# ------------------------------------------------------------------------------------- # 

# if ( len( sys.argv ) != 3 ):
#     print( "Usage: $python3 %s <file1.wav> <file2.wav>" % sys.argv[0] )
#     exit( -1 )

# print( "Comparing", sys.argv[1], sys.argv[2] )
# compare_mfcc( sys.argv[1], sys.argv[2] )	

print( BA_file2list( sys.argv[1] ) )
