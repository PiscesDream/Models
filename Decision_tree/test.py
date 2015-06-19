import id3
import treePlotter

if __name__ == '__main__':
    f = open('lenses.txt')
    names = f.readline().strip().split('\t')
    x = []
    y = []
    for ele in f.readlines():
        t = ele.strip().split('\t')
        x.append(t[:-1])
        y.append(t[-1])         
#    print x
#    print y

    print names
    print x
    print y

    Classifier = id3.ID3(names, x, y)
    ans = Classifier.result()
    print ans
    treePlotter.createPlot(ans)
