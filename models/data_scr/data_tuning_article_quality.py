import csv
import sys
import spacy 
import time 

start_time = time.time()
sp = spacy.load('en_core_web_sm')


file_name_input = sys.argv[1]
file_handler_input = open(file_name_input , "r")
csv_reader = csv.reader(file_handler_input)


out_file = sys.argv[2]
out_file_handler= open(out_file , 'w')
csv_writer = csv.writer(out_file_handler)
csvwriter.writerows(['title','label'])
out_list=[]

cnt_items =0


for data_point in csv_reader:
    # print("in loop")
    print(cnt_items)    
    cnt_items +=1 
    label=data_point[1]
    list_sentence=[]
    sentence = sp(data_point[0])
    for words in sentence:
        list_sentence.append(str(words.text))
    list_sentence = list_sentence[:300]
    # print(len(list_sentence))
    proper_sentence = " ".join(list_sentence)
    # print(len(proper_sentence))
    out_list.append([proper_sentence,label])
csv_writer.writerows(out_list)

file_handler_input.close()
out_file_handler.close()
print("Total Datapoints - %s "%(cnt_items))
print("Total time -- %s seconds" %(time.time()- start_time))
