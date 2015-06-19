import pickle as pic
from pprint import pprint

if __name__ == '__main__':
    d = pic.load(open('china_daily.dat', 'rb'))
    print 'loading done'

    new_d = {}
    for ele in d:
        if ele == 'Video News':
            continue

        arts = []
        for i in xrange(200):
#            print '=' * 80
#            pprint (d[ele]['entries'][0])
            try:
                arts.append((d[ele]['entries'][i]['title'],\
                             d[ele]['entries'][i]['summary'],\
                             d[ele]['entries'][i]['tags'][0]['term'],\
#                             d[ele]['entries'][i]['content'][0]['value'][:20]\
                             ))
            except:
                pass

        new_d[ele] = arts
        print ele, 'done'
        print '\tcounted:', len(arts)

    pprint(new_d)
    pic.dump(new_d, open('shrinked_200_3.dat','wb'), True)

