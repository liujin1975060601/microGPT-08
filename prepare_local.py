import os
import glob
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import DatasetDict, Dataset


num_proc = 8

folder_path = "./data/openwebtext/"  # 文件夹路径
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))  # 获取文件夹下所有txt文件的路径

text_data=[]
for txt_file in txt_files:
    with open(txt_file, encoding='utf-8') as f:  # 打开文件
        text_data += [line.strip() for line in f]

train_dataset = Dataset.from_dict({'text': text_data})
dataset = DatasetDict({"train": train_dataset})

split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token) 
    out = {'ids': ids, 'len': len(ids)}
    return out

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():#split='train'和'val'两次循环,dset是对应的文本集合
        arr_len = np.sum(dset['len'])
        filename = os.path.join(folder_path, f'{split}.bin')#os.path.dirname(__file__)
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        print(f"writing {filename}...")
        #dset里面每个文本example汇集到arr里面，最后保存
        idx = 0
        for example in tqdm(dset):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
