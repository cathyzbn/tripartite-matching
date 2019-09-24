#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:50:42 2019

@author: zhoubeining
"""

# -*- coding: utf-8 -*-

import numpy as np
import copy
import math
import random
from munkres import Munkres
import itertools
from timeit import default_timer as timer

class vertex:
    """
    The class vertex is the structure of all vertices. 
    @param name: used for printing out graphs or matchings. 
    It should be inputed as a string and is automatically set as None.
    """
    def __init__(self, name=None):
        self.name = name

    def __eq__(self, otherVtx):
        """
        @param otherVtx: an instance of class vertex;
        this is the other vertex that is compared to
        @return: return boolean of whether 'self' and 'otherVtx' are the same
        """
        if isinstance(otherVtx, self.__class__):
            return hash(id(self)) == hash(id(otherVtx))
        else:
            return False
        
    def getVtx(self):
        """
        @return: name of vertex
        """
        return self.name

#Below are some methods related to vertices
def vlist(num):
    """
    Makes a list of vertices of size num 
    with names 0, 1, ..., num-1
    @param num: number of vertex in the list
    @return: list of vertices with names 0, 1, ..., num-1
    """
    vtxList = []
    for i in range(0, num):
        vtxList.append(vertex(str(i)))
    return vtxList

def getVlist(vtxList):
    """
    Returns a list with the names of all vertices
    @param vtxList: the list of all vertices to print
    @return: the list of names of all vertices
    """
    printList = []
    for vtx in vtxList:
        printList.append(vtx.getVtx())
    return printList


class edge:
    # v is a list of vertices
    # w is int: the weight of the edge
    def __init__(self, u, v, w=0):
        """
        @param u: starting vertex
        @param v: ending vertex 
        @param w: weight (integer or float)
        """
        self.u = u
        self.v = v
        self.w = w

    def getEdge(self):
        return [self.u.getVtx(), self.v.getVtx(), self.w]
    
    def getVfrom(self):
        return self.u
    
    def getVto(self):
        return self.v
    
    def getw(self):
        return self.w
    
    def getEdgeVtxIndex(self, vnum):
        """
        When all vertices of a bipartite graph is put into one list, this function outputs the index of the starting vertex index and the ending vertex index of an edge (self)
        @param vnum: number of vertices in one section of the bipartite graph
        @return: starting vertex index and the ending vertex index, tuple of two integers.
        """
        uIdx = int(self.getVfrom().getVtx())
        vIdx = int(self.getVto().getVtx()) + vnum
        return uIdx, vIdx

def getElist(elist):
    """
    print out the edges in the list
    @param elist: list to be printed
    @return: 2-dim list of with list of name of starting vertex, name of ending vertex, and weight for each edge
    """
    printList = []
    for e in elist:
        printList.append(e.getEdge())
    return printList


class matching:
    # matchlist is a list of tuples
    def __init__(self, matchlist):
        """
        @param matchlist: a list of tuples(edge) with starting vertex index and ending vertex index
        """
        self.matchlist = matchlist
    
    def isMatched(self, start, end):
        """
        @param start: starting vertex index
        @param end: ending vertex index
        """
        tup = (start, end)
        for element in self.matchlist:
            if element == tup: return True
            return False


class graph:
    def __init__(self, v, e):
        """
        @param v: list of vertices
        @param e: list of edges
        """
        self.v = v
        self.e = e

    def getWeight(self, u, v):
        """
        list the weight of the edge with starting vertex u and ending vertex v
        @param u: starting vertex
        @param v: ending vertex
        @return: the weight of the edge (u, v)
        """
        e = self.e
        for edge in e:
            if ((edge.u == u) and (edge.v == v)):
                return edge.w


class bipartite(graph):
    def __init__(self, v1=[], v2=[], e=[]):
        """
        @param v1: list of first disjoint set of vertices
        @param v2: list of second disjoint set of vertices
        @param e: list of edges
        """
        self.v1 = v1
        self.v2 = v2
        self.e = e

    def getvlen(self):
        """
        Get number of vertices in a disjoint set.
        """
        return len(self.v1)

    def getbp(self):
        """
        Printing the bipartite graph
        @return: list with list of names of vertices in v1 and v2 and list of edges.
        """
        v1names = []
        v2names = []
        enames = []
        for v in self.v1:
            v1names.append(v.getVtx())
        for v in self.v2:
            v2names.append(v.getVtx())
        for e in self.e:
            enames.append(e.getEdge())
        return [str(v1names), str(v2names), str(enames)]

    def getMatrix(self):
        """
        Putting the graph into Matrix form
        @return: a 2-dim matrix with with [u][v]th element denoting the weight of the edge with starting vertex index u and ending vertex index v
        """
        mtrx = []
        vnum = self.getvlen()
        for i in range(vnum):
            mtrx.append([])
        for i in range(vnum):
            for j in range(vnum):
                mtrx[i].append(0)
        for i in range(vnum):
            for j in range(vnum):
                u = self.v1[i]
                v = self.v2[j]
                mtrx[i][j] = self.getWeight(u, v)
        for i in range(len(mtrx)):
            for j in range(len(mtrx[i])):
                if mtrx[i][j] == None:
                    mtrx[i][j] = 0
        return mtrx

    def munks(self):
        """
        Applying the Hungarian algorithm.
        @return the flow of the maximum matching (int) and the indices (list of tuples with length 2) of the matrix in the maximum matching
        """
        matrix = self.getMatrix()
        neg = copy.deepcopy(matrix)
        # obtaining the negative since all the Munkres() method is for minimum-weight matching.
        for i in range(len(neg)):
            for j in range(len(neg[i])):
                neg[i][j] = -neg[i][j]
        m = Munkres()
        indices = m.compute(neg)
        # calculates the total flow
        total = 0
        for row, column in indices:
            value = matrix[row][column]
            total += value
        return total
    
    def assignWeight(self, weightlist):
        """
        Assigns weights to edges in a bipartite graph
        @param weightlist: list length of vnum**2 containing integers for each weight
        @return: None
        """
        A = self.v1
        B = self.v2
        E = self.e
        i = 0
        for a in A:
            for b in B:
                E.append(edge(a, b, weightlist[i]))
                i = i+1 
    
    def listAppend(self, vtxIndex, vnum):
        returnlist = []
        for edge in self.e:
            if edge.getEdgeVtxIndex(vnum)[0] == vtxIndex: returnlist.append((edge.getVfrom(), edge.getVto()))
            if edge.getEdgeVtxIndex(vnum)[1] == vtxIndex: returnlist.append((edge.getVfrom(), edge.getVto()))
        return returnlist

    
    def agmPath(self, startIdx, endIdx, Mx):
        """
        Searches for augmenting path regarding a bipartite graph with given index of starting vertex, index of ending vertex and matching
        @param startIdex: index of starting vertex; index is original index
        @param endIdex: index of ending vertex
        @param Mx: matching to apply augmenting path on
        @return: list of original indices of vertices that the path goes through
        """
        vnum = self.getvlen()
        visited = [] # list of visited vertices
        goal = endIdx + vnum
        queue = [] # queue to draw out vertices to check
        #innitialize
        path = [startIdx] 
        queue.append((startIdx, path))
        while queue:
            (current, path) = queue.pop(0)
            if current not in visited:
                if current == goal:return path
                visited.append(current)
                # Search for matched and unmatched edges alternatingly
                if current < vnum:
                    for tup in self.listAppend(current, vnum):
                        vtx = int(tup[1].getVtx()) + vnum
                        queue.append((vtx, path+[vtx]))
                elif current >= vnum:
                    for tup in self.listAppend(current-vnum, vnum):
                        vtx = int(tup[0].getVtx())
                        if Mx.isMatched(vtx, current-vnum):
                            queue.append((vtx, path+[vtx]))


# Some iteration methods for listing neighbors
def choose_iter(elements, length):
    """
    yields all permutations of length k
    @param elements: elements to choose from for permutation
    @param length: length of desired permutation
    @return: list of permutations in form of tuples
    """
    for i in range(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next

def choose(n, k):
    """
    Generate a list of permutations of length k; Elements in permuation are numbers less than n.
    @param n: integer
    @param k: integer, length of permutation
    @return: the desired list
    """
    l = []
    for i in range(n):
        l.append(i)
    return list(choose_iter(l, k))

class tripartite(graph):
    def __init__(self, v1, v2, v3, X, Y):
        """
        @param v1, v2, v3: list of disjoint vertices
        @param X, Y: list of edges in tripartite graph
        """
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.X = X
        self.Y = Y
    
    def gettp(self):
        """
        Printing the tripartite graph
        """
        return [getVlist(self.v1), getVlist(self.v2), getVlist(self.v3), getElist(self.X), getElist(self.Y)]
    
    def updateY(self, Mx):
        """
        Given the left matching, this produces a bipartite graph of equivalence maximum flow.
        @param Mx: matching, the left matching of the graph on X
        @return: bipartite graph
        """
        vnum = len(self.v1)
        selfX = bipartite(self.v1, self.v2, self.X) #no change
        selfY = bipartite(self.v2, self.v3, self.Y) #no change
        bpY = bipartite(copy.deepcopy(self.v2), copy.deepcopy(self.v3))
        for tup in Mx.matchlist:
            v1_self  = self.v1[tup[0]]
            v2_self = self.v2[tup[1]]
            wx = selfX.getWeight(v1_self, v2_self)
            for index in range(vnum):
                originalWeight = selfY.getWeight(selfY.v1[tup[1]], selfY.v2[index])
                startVtx = bpY.v1[tup[1]]
                endVtx = bpY.v2[index]
                weight = min(originalWeight, wx)
                bpY.e.append(edge(startVtx, endVtx, weight))
        return bpY
    
    def getMaxY(self, Mx):
        """
        Calculates the maximum flow givin the left matching Mx
        @param Mx: matching, the left matching of the graph on X
        @return: flow of the maximum-weight matching
        """
        bpY = self.updateY(Mx)
        return bpY.munks()
    
    def getNeighbor(self, Mx, k):
        """
        @param Mx: current matching
        @param k: integer, swap size
        @return: list of matching neighbors
        """
        vnum = self.getvlen()
        l = choose(vnum, k)
        matchl = []
        for element in l:
            neighbor = copy.deepcopy(Mx)
            for i in range(len(element)):
                neighbor.matchlist[element[i-1]] = list(neighbor.matchlist[element[i-1]])
                neighbor.matchlist[element[i-1]][1] = Mx.matchlist[element[i]][1]
                neighbor.matchlist[element[i-1]] = tuple(neighbor.matchlist[element[i-1]])
            matchl.append(neighbor)
        return matchl
    

def generator(vnum, distribution, param):
    """
    @param vnum: int, number of vertices in each group
    @param distribution: string, either "random", "normal" or "uniform"
    @param param:
        in random:  largest edge weight
        in nomal: list of mu and sigma (mean and variance)
        in uniform: list of range (high and low) 
    """
    A = vlist(vnum)
    B = vlist(vnum)
    C = vlist(vnum)
    X = []
    Y = []
    
    global M
    list1 = []
    for i in range(vnum):
        list1.append((i, i))
    M = matching(list1)
    
    if distribution == "random":
        wx = np.random.randint(param, size = np.square(vnum))
        wy = np.random.randint(param, size = np.square(vnum))
        
    elif distribution == "normal":
        mu = param[0]
        sigma = param[1]
        w = np.random.normal(mu, sigma, 2*np.square(vnum))
        for i in range(len(w)):
            w[i] = int(w[i])
        wx = w[:np.square(vnum)]
        wy = w[np.square(vnum):]
        
    elif distribution == "uniform":
        low = param[0]
        high = param[1]
        w = np.random.uniform(low, high, 2*np.square(vnum))
        for i in range(len(w)):
            w[i] = int(w[i])
        wx = w[:np.square(vnum)]
        wy = w[np.square(vnum):]
        
    bipartite(A, B, X).assignWeight(wx)
    bipartite(B, C, Y).assignWeight(wy)
    return tripartite(A, B, C, X, Y)


def hill(graph, Mx, k = 2):
    # as explained in section 4.1
    maxnum = graph.getMaxY(Mx)[0]
    maxmatch = Mx
    neighbors = graph.getNeighbor(Mx, k)
    num = len(neighbors)
    while neighbors != None:
        index = np.random.randint(num)
        matching = neighbors.pop(index)
        num = num - 1
        result = graph.getMaxY(matching)[0]
        if result > maxnum:
            maxnum = result
            maxmatch = matching
            return hill(graph, maxmatch)
    return maxnum


def hill2(graph, Mx = M, k = 2):
    # as explained in section 4.2
    maxnum = graph.getMaxY(Mx)[0]
    maxmatch = Mx
    neighbors = graph.getNeighbor(Mx, k)
    bestNeighbor = None
    bestNeighborNum = 0
    for matching in neighbors:
        result = graph.getMaxY(matching)[0]
        boolean = False
        if np.logical_and((result > maxnum), (bestNeighborNum == 0)):
            boolean  = True
        elif np.logical_and((bestNeighborNum != 0), (result > bestNeighborNum)):
            boolean = True
        if boolean:
            maxnum = result
            maxmatch = matching
            currentNeighbor = {}
            for matching in neighbors:
                currentNeighbor[graph.getMaxY(matching)[0]] = matching
            bestNeighborNum = max(currentNeighbor.keys())
            if bestNeighborNum > maxnum:
                bestNeighbor = currentNeighbor.get(bestNeighborNum)
    if bestNeighbor == None: 
        return maxnum
    else: 
        return hill2(graph, bestNeighbor)

def anneal(graph, cycles, Mx = M, k=2):
    # as explained in section 4.3
    acceptedSol = 0.0
    pWorseStart = 0.7
    pWorseEnd = 0.001
    t1 = -1.0/math.log(pWorseStart)
    tFinal = -1.0/math.log(pWorseEnd)
    frac = (tFinal/t1)**(1.0/(cycles-1.0))#reduction of T
    currentMatching = Mx
    currentValue = graph.getMaxY(Mx)[0]
    acceptedSol = acceptedSol + 1.0
    tCurrent = t1
    deltaE_avg = 0.0
    for i in range(cycles):
        print("Cycle: "+str(i)+" with Temperature: "+str(tCurrent))
        neighborList = graph.getNeighbor(Mx, k)
        neighborNum = len(neighborList)
        index = np.random.randint(neighborNum)
        matching = neighborList[index]
        value = graph.getMaxY(matching)[0]
        print("value: "+str(value)+" current value: "+str(currentValue))
        deltaE = abs(value - currentValue)
        if (value < currentValue):
            if (i==0): deltaE_avg = deltaE
            p = math.exp(-deltaE/(deltaE_avg * tCurrent))
            if (random.random()<p):
                accept = True
            else:accept = False
        else: accept = True
        if (accept==True):
            currentMatching = matching
            currentValue = graph.getMaxY(matching)[0]
            acceptedSol = acceptedSol + 1.0
            deltaE_avg = (deltaE_avg * (acceptedSol-1.0) +  deltaE) / acceptedSol
        tCurrent = frac * tCurrent
    return currentValue


def getMatchingWeights(graph, matching):
    """
    @param graph: bipartite graph
    @param matching: matching on the bipartite graph
    @return: list of all weights in the matching
    """
    mlist = matching.matchlist
    weightlist = []
    for tup in mlist:
        u = graph.v1[tup[0]]
        v = graph.v2[tup[1]]
        weight = graph.getWeight(u, v)
        weightlist.append(weight)
    return weightlist   

def bigOplus(matchlist, path):
    """
    Calculate the symmetrical difference
    @param matchlist: list of tuples, matchlist of matching
    @param path: list of tuples representing edges in path
    return: matchlist after the symmetrical difference
    """
    result = []
    for tup in matchlist:
        if tup not in path:
            result.append(tup)
    for tup in path:
        if tup not in matchlist:
            result.append(tup)
    return result        
    
def bottle(graph):
    """
    Calculate the bottleneck matching of the bipartite graph
    @param graph: bipartite graph
    @return: list of weights of matching
    """
    Mx = copy.deepcopy(M)
    path = []
    vnum = graph.getvlen()
    while path !=  None:
        weightList = getMatchingWeights(graph, Mx)
        minEdge = min(weightList)
        minIndex = weightList.index(minEdge)
        minE = Mx.matchlist[minIndex]
        for edge in graph.e:
            if minE == (int(edge.getVfrom().getVtx()), int(edge.getVto().getVtx())):
                graph.e.remove(edge)
            elif edge.getw() < minEdge:
                graph.e.remove(edge)
        theDeletedEdge = Mx.matchlist[minIndex]
        start = theDeletedEdge[0]
        end = theDeletedEdge[1]
        Mx.matchlist.pop(minIndex)
        path = graph.agmPath(start, end, Mx)
        if path is None:
            return getMatchingWeights(graph, Mx)
        pathEdge = []
        for index in range(len(path)-1):
            if path[index+1] >= vnum:
                path[index+1] = path[index+1] - vnum
            pathEdge.append((path[index], path[index+1]))
        Mx.matchlist = bigOplus(Mx.matchlist, pathEdge)
    return getMatchingWeights(graph, Mx)
         
def bottleMin(graph):
    """
    Calculate the bottleneck estimation on the maximum-weight matching of a tripartite graph
    @param graph: tripartite graph
    @return: estimation
    """
    selfX = bipartite(graph.v1, graph.v2, graph.X) #no change
    selfY = bipartite(graph.v2, graph.v3, graph.Y) #no change
    btx = bottle(selfX)
    bty = bottle(selfY)
    x = min(btx)
    y = min(bty)
    minimum = min(x, y)
    vnum  = graph.getvlen()
    return minimum * vnum
     

def ennumerator(graph):
    """
    Ennumerates the matchinigs and calculates the maximum-weight matching for tripartite graphs.
    @param graph: tripartite graph
    @return: tuple of flow of maximum-weight matching and the matching
    """
    vnum = len(graph.v1)
    vlist = []
    for i in range(vnum):
        vlist.append(i) 
    permuteList = list(itertools.permutations(vlist))
    result = 0
    maxMatch = None
    for permutation in permuteList:
        matchlist = []
        for i in range(vnum):
            matchlist.append((i, permutation[i]))
        Mx = matching(matchlist)
        current = graph.getMaxY(Mx)[0]
        if current > result: 
            result = current
            maxMatch = Mx
    return result, maxMatch.matchlist

#random matching generator
def matchGenerator(vnum):
    matchlist = []
    order = np.random.permutation(vnum)
    for i in range(vnum): matchlist.append((i, order[i]))
    return matching(matchlist)
   
f = open('output.text', 'a')
vnum = 50
print('start\n')
mrand = matchGenerator(vnum)
g = generator(vnum, "random", 10)  
for n in range (1):
    print("new")
    start = timer()
    print('k = 2' + str(hill(g, mrand))+'\n')
    end = timer()
    print('time:' + str(end-start)+ '\n')
f.close

