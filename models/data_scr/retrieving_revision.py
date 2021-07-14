import requests
import mwparserfromhell
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML, fromstring
import re

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

def get_revision(revision_ids):

    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "revids": revision_ids,
        "rvprop": "timestamp|user|comment|content",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    # The data retrieved in the json format 
    PAGES = DATA

    #checking the data structure       
    print(PAGES)
    for items in PAGES['query'].keys():
        print(items)
    print(len(PAGES['query']['pages']))

    #PAGES['query']['pages'] is the list consisting of a single dictionary containing the data . 

    for i in PAGES['query']['pages']:
        #There are 4 items in the dictionary 'i' --> pageid <'int'> , ns <'int'> , title<'str'> ,revisions <'list'> 
        for items in i :
            print(items , " " ,type(i[items]))

    # #storing the data  , to parse using mwparserfromhell 
    raw_text=PAGES['query']['pages'][0]['revisions'][0]['slots']['main']['content']
    
    wikicode=mwparserfromhell.parse(raw_text)
    # templates = wikicode.filter_templates()
    # for temp in templates:
    print(wikicode)
    print("="*30)
    print(len(wikicode))

list_of_revision_ids=[638181419]#616502017,651762922]629393521,655365754, 644933637
for revids in list_of_revision_ids:
    get_revision(revids)