import urllib.request
import beautifulsoup4

## get html of fandom wiki
fp = urllib.request.urlopen("https://spongebob.fandom.com/wiki/List_of_transcripts")
mybytes = fp.read()
mystr = mybytes.decode("utf8")
fp.close()



print(mystr)