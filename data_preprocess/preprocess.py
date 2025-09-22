from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

import glob
import re
import os
import numpy as np
import sys
import csv


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub('<.*?>', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


f = open("drug_target_formatted.txt")
lines = f.readlines()
print(type(lines))

drug_gene = {}
for line in lines:
    line = re.sub(r"\n", "", line)
    line = re.sub(r"\t", " ", line).lower()
    b = line.split(" ")
    drug_name = ' '.join(b[:-1])
    if drug_name not in drug_gene:
        drug_gene[drug_name] = [b[-1]]
    else:
        drug_gene[drug_name] += [b[-1]]
f.close()
drug_gene


labels = {}
features = {}
cnt_all = 0
for folder in glob.glob('*.xml'):
    for sub_fol in glob.glob(os.path.join(folder, '*')):
        for filename in glob.glob(os.path.join(sub_fol, '*')):
            cnt_all += 1
            print('=============')
            print(filename)
            
            soup = BeautifulSoup(open(filename, encoding="utf-8"))
            # abstract
            abstract = clean_str(str(soup.find('abstract'))).lower()
            abstract = re.sub(r"\\", "", abstract)
            if abstract=='none': 
                print('No abstract!')
                continue
            # title
            title = clean_str(str(soup.find('article-title'))).lower()
            # full text
            full_text = re.sub('<.*?>', '', str(soup.find_all('sec')))
            full_text = re.sub(r"\n", "", full_text)
            full_text = re.sub(r"\t", " ", full_text).lower()
            sentences = full_text.split('.') # split doc into sentences
            # features of the paper
            features[filename] = {'title':title, 'abstract': abstract, 'full_text':  clean_str(full_text)}
            
            for drug in drug_gene:
                
                # Step 1: Select a subset of papers
                if drug in abstract or drug in title:
                    
                    # Initialization
                    if drug not in labels:
                        labels[drug] = {filename: {'lab':0, 'comment': 'none'}}
                        
                    if filename not in labels[drug]:
                        labels[drug][filename] = {'lab':0, 'comment': 'none'}
                        
                    # If gene in title or in abstract, no need to see the whole sentence, label = 0    
                    flag = False    
                    for gene in drug_gene[drug]:
                        if gene[:-1] in abstract or gene[:-1] in title:
                            labels[drug][filename]['lab'] = 2
                            flag = True
                            break
                    if flag: continue
                    
                    # Step 2: Read the full text to get the label
                    flag = False    
                    for sentence in sentences:
                        for gene in drug_gene[drug]:
                            if drug in sentence and gene[:-1] in sentence:
                                sentence = clean_str(sentence)
                                print(sentence) 
                                labels[drug][filename]['lab'] = 1
                                labels[drug][filename]['comment'] = sentence
                                flag = True
                                break
                        if flag: break


with open('labels.csv', 'w', encoding="utf-8") as f:
    for drug in labels:
        for file in labels[drug]:
            f.write("%s,%s,%d\n"%(drug, file, labels[drug][file]['lab']))


with open('features.csv', 'w') as f:
    for file in features:
        f.write("%s,%s,%s,%s\n"%(file, features[file]['title'], features[file]['abstract'], features[file]['full_text']))

