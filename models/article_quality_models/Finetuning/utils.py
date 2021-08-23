import os, json


def load_txt_file(filepath):
    res = []
    with open(filepath) as dfile:
        for line in dfile.readlines():
            res.append(line.strip())
    return res

def store_jsonl_file(res, filepath):
    with open(filepath, 'w', encoding='utf-8') as dfile:
        for x in res:
            json.dump(x, dfile, ensure_ascii=False)
            dfile.write("\n")

def load_jsonl_file(filepath, valid_field=None):
    res = []
    with open(filepath) as dfile:
        for line in dfile.readlines():
            temp = json.loads(line.strip())
            if valid_field:
                ptemp = {}
                for field in valid_field:
                    if field in temp:
                        ptemp[field] = temp[field]
                res.append(ptemp)
            else:
                res.append(temp)
    return res