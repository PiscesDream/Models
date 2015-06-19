import feedparser

d = feedparser.parse("http://feedparser.org/docs/examples/atom10.xml")

print d['feed']['title']
print d.feed.title
print d.channel.title
print d.feed.link
print d.channel.description
print len(d['entries'])
print d['entries'][0]['title']
print d['items'][0].title

e = d.entries[0]
print e.link
print e.links[1].rel
print e.links[0].href
print e.author_detail.name
print e.update_parsed
