class Apriori(object):
    
    def __init__(self, data, minSupport = 0.7, minConf = 0.5):
        
        def calcConf(freqSet, H, supD, ans):
            prunedH = []
            for conseq in H:
                conf = supD[freqSet] / supD[freqSet - conseq]
                if conf >= minConf:
                    ans.append((freqSet-conseq, conseq, conf))
                    prunedH.append(conseq)
            return prunedH

        def rulesFromConseq(freqSet, H, supD, ans):
            m = len(H[0])
            if (len(freqSet) > (m + 1)):
                Hmp1 = aprioriGen(H, m+1)
                Hmp1 = calcConf(freqSet, Hmp1, supD, ans)
                if len(Hmp1) > 1:
                    rulesFromConseq(freqSet, Hmp1, supD, ans)
        
        def createC1(data):
            C1 = []
            for group in data:
                for ele in group:
                    if not [ele] in C1:
                        C1.append([ele])
            C1.sort()
            return map(frozenset, C1)

        def scanData(Ck):
            ssCnt = {}
            for example in data:
                for can in Ck:
                    if can.issubset(example):
                        ssCnt[can] = ssCnt.get(can, 0) + 1
            numItems = float(len(data))
            retList = []
            supportData = {}
            for key in ssCnt:
                support = ssCnt[key]/numItems
                if support >= minSupport:
                    retList.insert(0, key)
                supportData[key] = support
            return retList, supportData

        def aprioriGen(Lk, k):
            retList = []
            lenLk = len(Lk)
            for i in range(lenLk):
                for j in range(i+1, lenLk):
                    L1 = list(Lk[i])[:k-2]
                    L2 = list(Lk[j])[:k-2]
                    L1.sort()
                    L2.sort()
                    if L1 == L2:
                        retList.append(Lk[i] | Lk[j])
            return retList            

        C1 = createC1(data)
        data = map(set, data)
        L1, supD = scanData(C1)
        L = [L1]
        k = 2
        while (len(L[k-2]) > 0):            
            Ck = aprioriGen(L[k-2], k)
            Lk, supK = scanData(Ck)
            supD.update(supK)
            L.append(Lk)
            k += 1
        
        rules = []
        for i in range(1, len(L)):
            for freqSet in L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if (i > 1):
                    rulesFromConseq(freqSet, H1, supD, rules)
                else:
                    calcConf(freqSet, H1, supD, rules)
        
        self.freqSet = L
        self.supD = supD
        self.rules = rules

    def freq_set(self):
        return self.freqSet, self.supD

    def freq_rules(self):
        return self.rules
    
    def pprint_rules(self):
        from pprint import pprint
        for ele in self.rules:
            print ele[0], '--->', ele[1], 'conf:', ele[2]

if __name__ == '__main__':
    a = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    ap = Apriori(a, 0.5, 0.5)
    l, supD = ap.freq_set()
    from pprint import pprint
#    pprint(l)
    pprint(supD)
    ap.pprint_rules()
