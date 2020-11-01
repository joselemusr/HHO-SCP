import numpy as np
from datetime import datetime

class PermRank:

    def __init__(self):
        #print("init")
        self.cache = {}


    def genBinPermutation(self, numBits):
        res = []
        for numOnes in range(numBits+1):
            temp = np.zeros((numBits), dtype='B')
            if numOnes > 0: temp[-numOnes:] = 1
            res.append(temp.copy())
            while True:       
                #Find the largest x such that P[x]<P[x+1].
                largestX = -1
                for idx in range(numBits - 1):
                    if temp[idx] < temp[idx+1]:
                        largestX = idx
                
                #(If there is no such x, P is the last permutation.)
                if largestX == -1: break

                #Find the largest y such that P[x]<P[y].
                largestY = -1
                for idx in range(numBits):
                    if(temp[largestX] < temp[idx]):
                        largestY = idx

                #Swap P[x] and P[y].

                aux = temp[largestX]
                temp[largestX] = temp[largestY]
                temp[largestY] = aux

                #Reverse P[x+1 .. n].

                temp[largestX+1:] = np.flip(temp[largestX+1:])

                res.append(temp.copy())

        self.cache[numBits] = np.array(res)

    def getRank(self, arr):
        arr = np.array(arr)
        if not arr.shape[0] in self.cache:
            self.genBinPermutation(arr.shape[0])
        return np.argwhere(np.all(self.cache[arr.shape[0]]==arr,axis=1))[0,0]
        
    def unrank(self, nBits, rank):
        if not nBits in self.cache:
            self.genBinPermutation(nBits)
        return self.cache[nBits][rank]

    def totalPerm(self, nBits):
        if not nBits in self.cache:
            self.genBinPermutation(nBits)
        return self.cache[nBits].shape[0]

"""
permRank = PermRank()
nBits = 3
totalPerm = permRank.totalPerm(nBits)

for perm in range(totalPerm):
    inicio = datetime.now()
    unranked = permRank.unrank(nBits,perm)
    fin = datetime.now()
    print(f"unranked = {unranked}, time = {fin-inicio}")

    inicio = datetime.now()
    ranked = permRank.getRank(unranked)
    fin = datetime.now()
    print(f"ranked = {ranked}, time = {fin-inicio}")
"""