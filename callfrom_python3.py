import os
import pandas as pd
import numpy as np
import pickle as pk
from collections import Counter
import sys
import swifter
import shlex
import subprocess


def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def new_wer(x):
    try:
        gt = x[-3].split(' ')
    except AttributeError:
        return 0
    try:
        lm = x[-1].strip().split(' ')
    except AttributeError:
        lm = []
    score = levenshtein(gt,lm)
    nor_score = score/len(gt)
    return nor_score


class Translitration():
    
    def __init__(self,path_csv,gt = 'gt', greedy = 'greedy',
                lm = 'lm'):
        self.df = pd.read_csv(path_csv)
        self.gt = gt
        self.lm = lm
        self.greedy = greedy
        self.EXPORT = "export LC_CTYPE=en_US.UTF-8"
        self.ENV_COMMAND = "conda activate cn27"


    def __get_command__(self,temp_input,temp_output):
        COMMAND = "python three_step_decoding.py --test-file {} --output-file {}".format(temp_input,temp_output)
        return COMMAND


    def run_command(self,path_to_savemapping, path_wrddist, csv_output,temp_input = 'temp_input.csv',
                    temp_output = 'temp_output.csv'):
        no_df = self.df.copy()
        no_df.dropna(inplace = True)
        full_gt_corpa = ' '.join(no_df[self.gt]) + ' '.join(no_df[self.lm])
        words = full_gt_corpa.split(' ')
        word_dist = Counter(words)
        wrd_set = list(set(words))
        lower_chars = set([chr(i) for i in range(97,97+26)])

        only_lower = [i for i in wrd_set if len(set(list(i)).difference(lower_chars))==0]
        with open(temp_input,'w') as f:
            for i in only_lower:
                if len(i) != 0:
                    f.write(i+'\n')
        FULL_COMMAND = "bash env_changer.sh {} {}".format(temp_input,temp_output)
        print("Running {}".format(FULL_COMMAND))
        pp = subprocess.Popen(shlex.split(FULL_COMMAND),shell=True,stdout=subprocess.DEVNULL)
        pp.wait()
        df = pd.read_csv(temp_output, header= None)
        mapping_one = {}
        for i,j in zip(df[0],df[1]):
            temp = mapping_one.get(j,set())
            temp.add(i)
            mapping_one[j] = temp
        max_reverse = {}
        for key in mapping_one.keys():
            val = list(mapping_one[key])
            temp_max = max([word_dist[i] for i in val])
            m_key = val[[word_dist[i] for i in val].index(temp_max)]
            max_reverse[m_key] = val
        actual_reverse = {}
        for i,j in max_reverse.items():
            for key in j:
                actual_reverse[key] = i
        def change_words(x):
            try:
                x = x.strip()
            except AttributeError:
                return ''
            words = x.split(' ')
            new_list = []
            for wrd in words:
                new_list.append(actual_reverse.get(wrd,wrd))
            return ' '.join(new_list)
        final_csv = pd.DataFrame()
        final_csv[self.gt] = self.df[self.gt]
        final_csv[self.lm] = self.df[self.lm]
        final_csv[self.greedy] = self.df[self.greedy]
        final_csv['sub_gt'] = self.df[self.gt].swifter.apply(change_words)
        final_csv['sub_greedy'] = self.df[self.greedy].swifter.apply(change_words)
        final_csv['sub_lm'] = self.df[self.lm].swifter.apply(change_words)
        final_csv['wer_lm'] = final_csv.swifter.apply(new_wer,axis =1) # some changes may be removed
        final_csv['wav_filename'] = self.df['wav_filename']
        with open(path_wrddist,'wb') as f:
            pk.dump(word_dist,f)
        print("Word Distribution saved at {}".format(path_wrddist))
        final_csv.to_csv(csv_output,index=False)
        print("Csv output at {}".format(csv_output))
        with open(path_to_savemapping,'wb') as f:
            pk.dump(actual_reverse,f)
        print("Saved the dictionary at {}".format(path_to_savemapping))

if __name__ == "__main__":
    obj = Translitration('/nfs/alldata/Airtel/Manifest/sub_clean_lm_output.csv', lm='transcript', gt='gt')
    obj.run_command('/nfs/alldata/Airtel/Manifest/sub_clean_lm_output-v1.pkl',
    '/nfs/alldata/Airtel/Manifest/sub_clean_lm_output-v1.pkl',
    '/nfs/alldata/Airtel/Manifest/sub_clean_lm_output_gt-v1.csv')