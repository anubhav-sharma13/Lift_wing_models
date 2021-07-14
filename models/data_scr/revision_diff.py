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
        "action": "compare",
        "prop": "revisions",
        "fromrev":revision_ids,
        "torelative" : "prev", 
        # "titles": "API|Main Page",
        "prop": "timestamp|user|size|comment|diff",
        "slots": "main",
        "formatversion": "2",
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    # The data retrieved in the json format 
    PAGES = DATA

    #checking the data structure 
    for items in PAGES.keys() :
        print(PAGES[items])
        print("="*40)
        wikicode = mwparserfromhell.parse(PAGES[items]['bodies']['main'])
        wikicode = "<table>"+str(wikicode)+"</table>"
        #print(wikicode)

        root = ET.fromstring(wikicode)
        text_table=[]
        #print(root.tag)
        for child in root:
            for ch in child:
                for lh in ch:
                    lh.text = str(lh.text)
                    text_local = re.sub(r'https?://\S+|www\.\S+', '', lh.text)
                    text_local = re.sub (r'<span|</span',"",text_local)
                    text_local= re.sub("[<>#@\[\]\{\}\;\"]","",text_local)
                    text_table.append(lh.text)
                    print(text_local)
                    print("!"*30)
        
list_of_revision_ids=[638181419]#616502017,651762922]629393521,655365754, 644933637
for revids in list_of_revision_ids:
    get_revision(revids)