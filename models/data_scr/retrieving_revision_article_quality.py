import requests
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML, fromstring
import re
from tqdm import tqdm 
import time 
import sys
import csv
import multiprocessing
import json 
S = requests.Session()

#the final list of data_text and the label 
dataset_list = []
global_cnt=0
input_filename = sys.argv[1]
output_filename = sys.argv[2]
out_file_handler = open(output_filename, 'w')
csvwriter = csv.writer(out_file_handler)
csvwriter.writerow(['title','label'])
#the function to create dataset (takes revision_ids --> int and label --> int as argument)
def get_revision(revision_ids , label):

    #API to get the Introductory Paragraph 
    URL = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&revids={}".format(revision_ids)
    R = S.get(URL)

    DATA = R.json()
    try : 

        for i in DATA['query']['pages']:
            try : 
                QID = int(i)
                #print(DATA['query']['pages'][i]['extract'])
                if (DATA['query']['pages'][i]['extract'].strip() != ""):
                    
                    dataset_list.append([DATA['query']['pages'][i]['extract'].strip() , label])
                        
            except:
                print("INVALID rev_id")
    except:
        print("Invalid revid")
list_of_revision_ids=[]
list_of_labels = []
inp_file_handler = open(input_filename ,'r')
for files in inp_file_handler:
    json_obj=json.loads(files)
    # print(json_obj) 
    label_text = json_obj['wp10']
    if (label_text == 'FA'):
        list_of_labels.append(0)
    elif label_text == 'GA':
        list_of_labels.append(1)
    elif label_text == 'B':
        list_of_labels.append(2)
    list_of_exceptions = ["Start" ,"C" ,"Stub"]
    if label_text not in list_of_exceptions :
        list_of_revision_ids.append(json_obj['rev_id']) 
global_start_time = time.time()
for index in tqdm(range(len(list_of_revision_ids)), desc="Progress .. "):
    global_cnt+=1
    print(global_cnt)
    start_time = time.time()
    get_revision(list_of_revision_ids[index] , list_of_labels[index])
    if global_cnt%50 == 0 :
        out_file_handler = open(output_filename, 'w')
        csvwriter = csv.writer(out_file_handler)
        csvwriter.writerows(dataset_list)
    print("--- %s seconds ---" % (time.time() - start_time))





inp_file_handler.close()
out_file_handler.close()
print("--- Total Time - %s seconds ---" %(time.time() - global_start_time))
