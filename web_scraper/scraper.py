import urllib.request
from bs4 import BeautifulSoup
import re

def main():
  print("main initialized. . .")
  create_document_from_episodes()

def create_document_from_episodes():
  print("opening document spongebob-transcripts.txt. . .")
  document = open('../web_scraper/spongebob-transcripts.txt', 'w')
  for ep in get_episodes():
    document.write(ep.title)
    document.write('\n\n=========================================================')
    document.write(ep.body)
    document.write('_________________________________________________________\n')
  
  print("closing document. . .")
  document.close()
  
def get_episodes():
  episodes = []
  for link in get_transcript_links():
    episodes.append(get_episode_from_link(link))
  return episodes

def get_episode_from_link(link):
  print("fetching episode from link. . .")
  html = html_from_url(link)
  soup = BeautifulSoup(html, "html.parser")

  #get body and title
  title = soup.select('.page-header__page-subtitle > a:nth-child(1)')[0]['title']
  body = ''
  for ul in soup.select('#mw-content-text > ul'):
    body += ul.prettify()
  
  # find and replace <b>, </b>, <i>, </i>, </li> with ''
  artifacts = ['<b>', '</b>', '<i>', '</i>', '</li>', '</ul>', '<li>', '<ul>']
  for art in artifacts:
    body = body.replace(art, '')

  return Episode(title, body)

def get_transcript_links():
  print("getting transcript links. . .")
  wikiURL = "https://spongebob.fandom.com/wiki/List_of_transcripts"
  transcriptSoup = BeautifulSoup(html_from_url(wikiURL), "html.parser")
  transcriptLinkElements = transcriptSoup.find_all(is_transcript_link)
  transcriptLinks = []

  for elem in transcriptLinkElements:
    transcriptLinks.append(make_link(elem))
  return transcriptLinks

def html_from_url(url):
  fp = urllib.request.urlopen(url)
  mybytes = fp.read()
  fp.close()
  return mybytes.decode("utf8")

def is_transcript_link(tag):
  isLink = tag.has_attr('href') and tag.has_attr('title')
  if not isLink:
    return False
  
  isTranscript = ('transcript' in tag['title']) and tag.parent.name == 'center'
  if not isTranscript:
    return False
  
  headerId = tag.parent.parent.parent.parent.previous_sibling.previous_sibling.contents[0]['id']
  return re.compile('Season_').search(headerId)

def make_link(aTag):
  return "https://spongebob.fandom.com" + aTag['href']

class Episode:
  def __init__(self, title, body):
    self.title = title
    self.body = body

main()