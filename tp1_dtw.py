
import numpy as np
import math
from collections import defaultdict

def euclidienne( a, b ):
    return abs( a - b )

def lettres( a, b ):
    return 0 if a==b else 1

def sons( a, b ):
    dsons = np.array( [[0,1,1,1,2], 
                       [1,0,1,1,2], 
                       [1,1,0,1,1],
                       [1,1,1,0,2],
                       [2,2,1,2,0]] )
    d = { 'X':0, 'C':1, 'U':2, 'Ux':3, 'V':4 }
    return ( dsons[d[a]][d[b]] )
 
def dtw( A, B, I, J, x ):
    w0 = 1
    w1 = 2
    w2 = 1
    
    g = np.zeros( (I, J) )
    for j in range( 1, J ):
        g[0][j] = 1000000000000000000
    
    for i in range( 1, I ):
        g[i][0] = 100000000000000000
        for j in range( 1, J ):
            d = x( A[i], B[j] )
            g[i][j] = min( g[i-1][j] + w0*d,
                           g[i-1][j-1] + w1*d,    
                           g[i][j-1] + w2*d ) 
    print( g )
    return g[I-1][J-1] / ( I+J )

A = [ 1, 2, 3, 4, 5 ]
B = [ 1, 2, 3, 4, 5 ]

s1 = [ 'X', 'C', 'U' ]
s2 = [ 'X', 'C', 'U' ]

dtw( A, B, 5, 5, euclidienne)

dtw( s1, s2, 3, 3, sons )

# TODO test with true values the function 
## RApport : diferents ponderations, couts ...
