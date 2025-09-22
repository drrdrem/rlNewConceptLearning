import re
import os
import random
import tarfile
from tqdm import tqdm
import pandas as pd
from torchtext import data

class CustomDataset(data.Dataset):
    """Create an dataset instance given paths and fields.

    Arguments:
        text_field: The field that will be used for text data.
        label_field: The field that will be used for label data.
        lab_path: labels csv file path.
        fea_path: features csv file path.
        drugs: a list of desired drugs (all in lowe case). Default: None
        seed (int): random seed, default 0.
        Remaining keyword arguments: Passed to the constructor of
            data.Dataset.
    """
    name = 'drug gene'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self,  text_field, label_field, lab_path, fea_path, examples=None, drugs=None, seed=0, **kwargs):
        random.seed(seed)
        
        fields = [('text', text_field), ('label', label_field)]
        if examples is None: 
            examples = []
            # lab_data = pd.read_csv(lab_path,header=None)
            # fea_data = pd.read_csv(fea_path, header=None)

            lab_data = lab_path.reset_index(drop=True)
            fea_data = fea_path.reset_index(drop=True)

            print("preparing examples...")
            for i in tqdm(range(len(lab_data))):
                sample = lab_data.loc[i]
                drug, file, label = sample['drug'], sample['file'], sample['lab']

                if label==2: continue
                if drugs:
                    if drug not in drugs: continue

                example = fea_data[fea_data['file']==file].values.tolist()
                title, abstract = example[0][1], example[0][2]

                if abstract=='none': continue
                examples.append(data.Example.fromlist([str(title)+" "+str(abstract)+" "+str(drug)+" "+"gene", label], fields))

        super(CustomDataset, self).__init__(examples, fields, **kwargs)
    
    @classmethod
    def splits(cls, text_field, label_field, lab_path, fea_path, drugs=None, dev_ratio=.1, shuffle=True, **kwargs):
        """Create dataset objects for splits of the dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            lab_path: labels csv file path.
            fea_path: features csv file path.
            shuffle: Whether to shuffle the data before split.
            drugs: a list of desired drugs (all in lowe case). Default: None
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(text_field, label_field, lab_path, fea_path, drugs, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))
        print("dev_index: ", dev_index)

        return (cls(text_field, label_field, lab_path, fea_path, examples=examples[:dev_index]),
                cls(text_field, label_field, lab_path, fea_path,examples=examples[dev_index:]))