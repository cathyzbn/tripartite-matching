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

def vlist(num):
    # make list of vertex of size num
    v = []
    for i in range(0, num):
        v.append(vertex(str(i)))
    return v

def getVlist(vlist):
    # printing list of vertices
    vp = []
    for v in vlist:
        vp.append(v.getVtx())
    return vp


class edge:
    # v is a list of vertices
    # w is int: the weight of the edge
    def __init__(self, u, v, w=0):
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
        uIdx = int(self.getVfrom().getVtx())
        vIdx = int(self.getVto().getVtx()) + vnum
        return uIdx, vIdx

def getElist(elist):
    # printing list of vertices
    ep = []
    for e in elist:
        ep.append(e.getEdge())
    return ep
    
#def isin(u, v, edgelist):
#    boo = False
#    for edge in edgelist:
#        if (u == edge.getVfrom()) & (v == edge.getVto()):
#            boo = True
#            return boo
#    return boo


class graph:
    # v and e: list of vertices and edges
    def __init__(self, v, e):
        self.v = v
        self.e = e


def getWeight(graph, u, v):
    e = graph.e
    for edge in e:
        if ((edge.u == u) and (edge.v == v)):
            return edge.w

def isMatched(start, end, Mx):
    """
    start is the starting vertex index
    end is the ending vertex index
    """
    tup = (start, end)
    for element in Mx.matchlist:
        if element == tup: return True
    return False


class bipartite(graph):
    def __init__(self, v1=[], v2=[], e=[]):
        self.v1 = v1
        self.v2 = v2
        self.e = e

    def getvlen(self):
        return len(self.v1)

    def getbp(self):
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

#    def allEdges0(self):
#        new = bipartite(self.v1[:], self.v2[:], self.e[:])
#        for i in new.v1:
#            for j in new.v2:
#                new.e.append(edge(i, j, 0))
#        return new

    def getMatrix(self):
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
                mtrx[i][j] = getWeight(self, u, v)
        for i in range(len(mtrx)):
            for j in range(len(mtrx[i])):
                if mtrx[i][j] == None:
                    mtrx[i][j] = 0
        return mtrx

    def munks(self):
        matrix = self.getMatrix()
        neg = copy.deepcopy(matrix)
        for i in range(len(neg)):
            for j in range(len(neg[i])):
                neg[i][j] = -neg[i][j]
        m = Munkres()
        indexes = m.compute(neg)
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
        return total, indexes
    
    def assignWeight(self, weightlist):
        """
        weightlist is a list length of vnum containing ints for each weight
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
        vnum = len(self.v1)
        # whole vertex list with first group followed by 2nd group
        visited = []
        goal = endIdx + vnum
        # queue to draw out vertices to check
        queue = []
        path = [startIdx]
        queue.append((startIdx, path))
        
        while queue:
            (current, path) = queue.pop(0)
            if current not in visited:
                if current == goal:return path
                visited.append(current)
                if current < vnum:
                    for tup in self.listAppend(current, vnum):
                        vtx = int(tup[1].getVtx()) + vnum
                        queue.append((vtx, path+[vtx]))
                elif current >= vnum:
                    for tup in self.listAppend(current-vnum, vnum):
                        vtx = int(tup[0].getVtx())
                        if isMatched(vtx, current-vnum, Mx):
                            queue.append((vtx, path+[vtx]))
                            
#            elif isMatched(path[-1], path[-2]-vnum, Mx):
#                oldPath = copy.copy(path)
#                for edge in self.listAppend(current, vnum):
#                    vtx = edge.getEdgeVtxIndex(vnum)[1]
#                    if vtx == endIdx:
#                        return path
#                    if not(isMatched(path[-1], vtx, Mx)):
#                        queue.append(vtx)
#                if oldPath == path:
#                    queue.pop(current)
#            else:
#                oldPath = copy.copy(path)
#                for edge in self.listAppend(current, vnum):
#                    vtx = edge.getEdgeVtxIndex(vnum)[0]
#                    if vtx == endIdx:
#                        return path
#                    if isMatched(path[-1]-vnum, vtx, Mx):
#                        queue.append(vtx)
#                if oldPath == path:
#                    queue.pop(current)
                    
                

class matching:
    # matchlist is a list of tuples
    def __init__(self, matchlist):
        self.matchlist = matchlist
        
       

def choose_iter(elements, length):
    for i in range(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next


def choose(n, k):
    l = []
    for i in range(n):
        l.append(i)
    return list(choose_iter(l, k))


class tripartite(graph):
    # this is a complete tripartite
    def __init__(self, v1, v2, v3, X, Y):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.X = X
        self.Y = Y
    
    def gettp(self):
        return [getVlist(self.v1), getVlist(self.v2), getVlist(self.v3), getElist(self.X), getElist(self.Y)]
    
    def updateY(self, Mx):
        # Mx is a matching
        vnum = len(self.v1)
        selfX = bipartite(self.v1, self.v2, self.X) #no change
        selfY = bipartite(self.v2, self.v3, self.Y) #no change
        bpY = bipartite(copy.deepcopy(self.v2), copy.deepcopy(self.v3))
        for tup in Mx.matchlist:
            v1_self  = self.v1[tup[0]]
            v2_self = self.v2[tup[1]]
            wx = getWeight(selfX, v1_self, v2_self)
            for index in range(vnum):
                originalWeight = getWeight(selfY, selfY.v1[tup[1]], selfY.v2[index])
                startVtx = bpY.v1[tup[1]]
                endVtx = bpY.v2[index]
                weight = min(originalWeight, wx)
                bpY.e.append(edge(startVtx, endVtx, weight))
        return bpY
    
    def getMaxY(self, Mx):
        bpY = self.updateY(Mx)
        return bpY.munks()
    
    def getXasbp(self, Mx):
        bpX = bipartite(self.v1, self.v2, [])
        for tup in Mx:
            vtx1 = self.v1[tup[0]]
            vtx2 = self.v2[tup[1]]
            wx = getWeight(bpX, vtx1, vtx2)
            bpX.e.append(edge(vtx1, vtx2, wx))
    
    def getNeighbor(self, Mx, k):
        """
        return list of matching neighbors
        """
        vnum = len(self.v1)
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
    vnum: int, number of vertices in each group
    distribution: string,
    rangenum:
        in random:  largest edge weight
        in nomal: list, mu and sigma (mean and variance)
        in uniform: list, smallest - largest
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
    print("enter")
    maxnum = graph.getMaxY(Mx)[0]
    maxmatch = Mx
    neighbors = graph.getNeighbor(Mx, k)
    print("neighbors calculated")
    for matching in neighbors:
        result = graph.getMaxY(matching)[0]
        if result > maxnum:
            print("enter if")
            maxnum = result
            maxmatch = matching
            return hill(graph, maxmatch)
    return maxnum, maxmatch.matchlist

def hill2(graph, Mx, k = 2):
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
        return maxnum, maxmatch.matchlist
    else: 
        return hill2(graph, bestNeighbor)

def anneal(graph, cycles, Mx, k=2):
#    cycles = 20
    acceptedSol = 0.0
    pWorseStart = 0.7
    pWorseEnd = 0.001
    t1 = -1.0/math.log(pWorseStart)
    tFinal = -1.0/math.log(pWorseEnd)
    # Fractional reduction every cycle
    frac = (tFinal/t1)**(1.0/(cycles-1.0))
    currentMatching = Mx
    currentValue = graph.getMaxY(Mx)[0]
    acceptedSol = acceptedSol + 1.0
    tCurrent = t1
    deltaE_avg = 0.0
    for i in range(cycles):
#        print('Cycle: ' + str(i) + ' with Temperature: ' + str(tCurrent))
        neighborList = graph.getNeighbor(Mx, k)
        neighborNum = len(neighborList)
        index = np.random.randint(neighborNum)
        matching = neighborList[index]
        value = graph.getMaxY(matching)[0]
#        print("value: " + str(value) + " current value: " + str(currentValue))
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
    return currentValue, currentMatching.matchlist

#def anneal(graph, Mx, k=2):
#    cycles = 20
#    trialsPerCycle = 3
#    acceptedSol = 0.0
#    pWorseStart = 0.7
#    pWorseEnd = 0.001
#    t1 = -1.0/math.log(pWorseStart)
#    tFinal = -1.0/math.log(pWorseEnd)
#    # Fractional reduction every cycle
#    frac = (tFinal/t1)**(1.0/(cycles-1.0))
#    currentMatching = Mx
#    currentValue = graph.getMaxY(Mx)[0]
#    acceptedSol = acceptedSol + 1.0
#    tCurrent = t1
#    deltaE_avg = 0.0
#    for i in range(cycles):
#        print('Cycle: ' + str(i) + ' with Temperature: ' + str(tCurrent))
#        neighborList = graph.getNeighbor(Mx, k)
#        neighborNum = len(neighborList)
#        index = np.random.randint(neighborNum)
#        matching = neighborList[index]
#        value = graph.getMaxY(matching)[0]
#        print("value: " + str(value) + " current value: " + str(currentValue))
#        deltaE = abs(value - currentValue)
#        if (value < currentValue):
#            if (i==0): deltaE_avg = deltaE
#            p = math.exp(-deltaE/(deltaE_avg * tCurrent))
#            if (random.random()<p):
#                accept = True
#            else:accept = False
#        else: accept = True
#        if (accept==True):
#            currentMatching = matching
#            currentValue = graph.getMaxY(matching)[0]
#            acceptedSol = acceptedSol + 1.0
#            deltaE_avg = (deltaE_avg * (acceptedSol-1.0) +  deltaE) / acceptedSol
#        tCurrent = frac * tCurrent
#    return currentValue, currentMatching.matchlist

def ennumerator(graph):
    '''
    graph is tripartite
    '''
    print("begin enumerating")
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
        print("per")
        if current > result: 
            result = current
            maxMatch = Mx
    
    return result, maxMatch.matchlist
        
    

def getMatchingWeights(graph, matching):
    mlist = matching.matchlist
    weightlist = []
    for tup in mlist:
        u = graph.v1[tup[0]]
        v = graph.v2[tup[1]]
        weight = getWeight(graph, u, v)
        weightlist.append(weight)
    return weightlist   


def bigOplus(matchlist, path):
    #two inputs are both list of tuples representing edges
    newMatchlist = []
    for tup in matchlist:
        if tup not in path:
            newMatchlist.append(tup)
    for tup in path:
        if tup not in matchlist:
            newMatchlist.append(tup)
    return newMatchlist

def getFlow(graph, Mx):
    flow = 0
    for tup in Mx.matchlist:
        #calculate matching flow
        v1_self  = graph.v1[tup[0]]
        v2_self = graph.v2[tup[1]]
        wx = getWeight(graph, v1_self, v2_self)
        flow = flow + wx
    return flow
        
    

def bottle(graph):
    """
    graph: complete bipartite, should be copied aready
    output: matching
    """
    Mx = copy.deepcopy(M)
    path = []
    vnum = len(graph.v1)
    print(vnum)
    while path !=  None:
        weightList = getMatchingWeights(graph, Mx)
        print(weightList)
        minEdge = min(weightList)
        minIndex = weightList.index(minEdge)
        minE = Mx.matchlist[minIndex]
        print("here is minE")
        print(minE)
        for edge in graph.e:
            if minE == (int(edge.getVfrom().getVtx()), int(edge.getVto().getVtx())):
                graph.e.remove(edge)
                print("removed")
            elif edge.getw() < minEdge:
                graph.e.remove(edge)
        theDeletedEdge = Mx.matchlist[minIndex]
        start = theDeletedEdge[0]
        end = theDeletedEdge[1]
        Mx.matchlist.pop(minIndex)
        print("here is the start and end: " )
        print(start, end)
        path = graph.agmPath(start, end, Mx)
        print("this is path list of vtxes")
        print(path)
        if path is None:
            print("about to return")
            return getMatchingWeights(graph, Mx)
        pathEdge = []
        for index in range(len(path)-1):
            if path[index+1] >= vnum:
                path[index+1] = path[index+1] - vnum
            pathEdge.append((path[index], path[index+1]))
        print("this is the path")
        print(pathEdge)
        Mx.matchlist = bigOplus(Mx.matchlist, pathEdge)
        print("this is the matching")
        print(Mx.matchlist)
    return getMatchingWeights(graph, Mx)
    
    
            
def bottleMin(graph):
    """
    graph is tripartite
    """
    selfX = bipartite(graph.v1, graph.v2, graph.X) #no change
    selfY = bipartite(graph.v2, graph.v3, graph.Y) #no change
    print(selfY.getbp())
    btx = bottle(selfX)
    print("exited")
    print(selfX.getbp())
    print(selfY.getbp())
    bty = bottle(selfY)
    x = min(btx)
    y = min(bty)
    minimum = min(x, y)
    vnum  = len(graph.v1)
    print(minimum)
    return minimum * vnum
        
  
#    T = 1.0
#    T_min = 0.00001
#    alpha = 0.9
#    max_time = k
#    old = Mx
#    while T > T_min:
#        i = 1
#        while i <= max_time:
#            neighborList = graph.getNeighbor(Mx, k)
#            neighborResults = []
#            for element in neighborList:
#                neighborResults.append(graph.getMaxY(element))
                

# class k_pmt:
#    def __init__(self, swapnum):
#
#
#

# testing up to bipartite
#U = vlist(10)
#V = vlist(10)
#E = [edge(U[0], V[0], 2), edge(U[1], V[3], 7)]
#bp = bipartite(U, V, E)
#print(bp.getMatrix())
#print(bp.munks())



# print(bp.getbp())
# new = bp.allEdges0()
# print(bp.getbp())
# print(new.getbp())

## Munkre's Testing
#A = vlist(3)
#B = vlist(3)
#C = vlist(3)
#X = [edge(A[0], B[0], 1), edge(A[0], B[1], 2), edge(A[2], B[2], 7), edge(A[0], B[2], 2), edge(A[1], B[0], 8), edge(A[1], B[1], 5)]
#Y = [edge(B[0], C[0], 2), edge(B[1], C[0], 4), edge(B[1], C[1], 7), edge(B[1], C[2], 3), edge(B[0], C[2], 2), edge(B[0], C[1], 3), edge(B[2], C[2], 3)]
#gr = tripartite(A, B, C, X, Y)
##hill(gr)
#Mx = matching([(0, 0), (1, 1), (2, 2)])
#neighbors = gr.getNeighbor(Mx, 2)
#for element in neighbors:
#    print(element.matchlist)
#up = gr.updateY(Mx)
#
#print(up.getbp())
#print(up.munks())
#M2 = matching([(0, 0), (1, 2), (2, 1)])
#M1 = matching([(0, 1), (1, 0), (2, 2)])
g1 = generator(50, "random", 50)
print("generated")
#g2 = bipartite(g1.v1, g1.v2, g1.X) #no change
#btm = bottleMin(g1)
#print("this is min")
#print(btm)

#print(g1.updateY(M).getbp())
start1 = timer()
print(hill(g1, M))
end1 = timer()
print(end1-start1)
start2 = timer()
print(hill2(g1))
end2 = timer()
print(end2-start2)
start3 = timer()
print(anneal(g1, 30))
end3 = timer()
print(end3-start3)


#TESTING FOR CYCLE NUM FOR ANNEALING
#num1 = 0
#num2 = 0
#num3 = 0
#cycle = 5
#while cycle < 90:
#    num1 = num2
#    num2 = num3
#    start = timer()
#    num3 = anneal(g1, cycle)[0]
#    print(num3)
#    
#    print(end-start)
#    if np.logical_and((num1 == num2),(num1 == num3)):
#        print("result: " + str(cycle-10))
#        print("num " + str(num3))
#    cycle = cycle + 5



