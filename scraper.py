import urllib.request
from bs4 import BeautifulSoup
import re

def main():
  get_episode_transcripts()

def get_episode_transcripts():
  ## get html of fandom wiki
  fp = urllib.request.urlopen("https://spongebob.fandom.com/wiki/List_of_transcripts")
  mybytes = fp.read()
  transcriptList = mybytes.decode("utf8")
  fp.close()

  transcriptSoup = BeautifulSoup(transcriptList, "html.parser")
  transcriptLinks = transcriptSoup.find_all(is_transcript_link)
  print(transcriptLinks)

def is_transcript_link(tag):
  isLink = tag.has_attr('href') and tag.has_attr('title')
  if not isLink:
    return False
  
  isTranscript = ('transcript' in tag['title']) and tag.parent.name == 'center'
  if not isTranscript:
    return False
  
  headerId = tag.parent.parent.parent.parent.previous_sibling.previous_sibling.contents[0]['id']
  return re.compile('Season_(?!3)').search(headerId)

main()