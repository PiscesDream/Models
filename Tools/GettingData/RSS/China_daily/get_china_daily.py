import pickle as pic
from pprint import pprint
from feedparser import parse

if __name__ == '__main__':
    d = {}

    f = open('china_daily.list', 'r')
    for line in f:
        line = line.strip()
        if line.startswith('http'):
            print 'getting %s' % name
            art = parse(line)
            d[name] = art
        else:
            name = line

    print 'saving'
    f = open('china_daily.dat', 'wb')
    pic.dump(d, f, True)
    f.close()

    pprint(d)
