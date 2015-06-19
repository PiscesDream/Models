class NaiveBayes(object):
   
    def __init__(self, x, y, 
        f = lambda x: x, f_mul = lambda x, y: x*y, g = lambda x: x, 
        m = None, p = None):        
        '''Train NavieBayes Classifier using  p = (n_ele + m * p)/(n + m) as pobability
            f(x) = ln(x) f_mul(x, y) = x+y, g(x) = exp(x)
            e.g.
                x = [['happy', 'sport', 'win', 'games'],
                     ['war', 'crisis', 'hunger', 'died'],
                     ['dollar', 'RMB', 'decrease', 'bad'],
                     ['benign', 'recover']]
                y = ['sports', 'international', 'finance','disease'] '''

        self.x = x
        self.y = y
        self.y_set = set(y)
        train_size = len(y)
        self.element_set = reduce(lambda x, y: x.union(y), map(set, x))
        if p == None:
            p = 1.0/len(self.element_set)
            m = len(self.element_set)
        self.p = p
        self.m = m
        self.f = f
        self.g = g
        self.f_mul = f_mul

        self.conditional_probability = {}
        self.probability_v = {}
        for v in set(y):
            self.conditional_probability[v] = {}
            self.probability_v[v] = f( y.count(v) * 1.0 / train_size )

            #blt-->belong to 
            ele_blt_v = reduce(lambda x, y: x+y, [x[ind] for ind, ele in enumerate(y) if ele == v])
            n = len(ele_blt_v)
            for element in self.element_set:
                n_ele = ele_blt_v.count(element)
                self.conditional_probability[v][element] = f( (n_ele + m * p)*1.0/(n + m) )

    def classify(self, x, show_all = False):
#        print self.conditional_probability

        Pros = {}
        sumPro = 0
        bestPro = 0
        bestPre = None
        for prediction in self.y_set:
            pro =self.f( self.probability_v[prediction] )
            for ele in x:
                if ele in self.element_set:
                    pro = self.f_mul(pro, self.conditional_probability[prediction][ele])
            if pro > bestPro:
                bestPro = pro
                bestPre = prediction
            Pros[prediction] = pro
            sumPro += self.g(pro)

        for v in Pros:
            Pros[v] = self.g( Pros[v]/sumPro )

        if show_all:
            return bestPre, self.g(bestPro)/sumPro, Pros
        else:
            return bestPre, self.g(bestPro)/sumPro
