import sys
import traceback
import numpy as np
from ldacgs import *
import pandas as pd
import numpy as np
import os,glob
from matplotlib import pyplot as plt

# also make the file unique naming function

# maybe instead of doing it by topic just iterate through in groups of 200 hundreds

"""
INSTEAD OF GOING THROUGH TOPICS - JUST LEAVE IT RUNNING throughout the night, so maybe it should work backwards
"""
path = 'other_files/'
f = 'merged_summary_topics.json'

new = pd.read_json(path + f)
# my data is in data2 not data
new['File'] = [x[5:] for x in new.File]
new['File_num'] = [int(x[:-4]) for x in new.File]
new['tag_count'] = [len(x) for x in new.topic_lists]

chosen_topic = 'fasting'
repeat_lda_times = 5
params = [(0.001,1)
    ,(0.005,1)
    ,(0.001,0.17)]
attempt = 0
output_filename = f'{chosen_topic}_{repeat_lda_times}_lda_per_talk_attempt_{attempt}.json'
# raise NotImplementedError('pick alpha and beta values')


results = dict()
beginning = time.time()
errors = 0

counter = 0
files = new.loc[new[chosen_topic] == 1].File_num.values
new_set = set(files)
files = new_set - already_done_file_nums
already_done_file_nums = already_done_file_nums.union(new_set)

total_times = len(files) * repeat_lda_times
print(f'this is going to run {total_times} times')
for file_num in files:
    for i in range(repeat_lda_times):
        # avoid out of error with accessing betas
        j = (i % len(params))
        a,b = params[j]
        raise
        try:
            x = new.loc[new.File_num == file_num]
            start = time.time()
            #gs for gibbs sampler
            gs = LDACGS(n_topics=10,alpha=a,beta=b)
            # gs.buildCorpus(filename='reagan.txt')
            # gs.initialize()
            # gs._sweep()

            gs.sample(filename='data2/' + new.loc[new.File_num == file_num].File.values[0]
                    , stopwords_file='../3winter21/Gibbs_LDA/' + 'stopwords.txt')

            results[counter] = {'File_num':file_num
                 ,'Title':x.Title.values[0]
                 ,'Speaker':x.Speaker.values[0]
                 ,'Year':x.Year.values[0]
                 ,'Month':x.Month.values[0]
                 ,'alpha':a
                 ,'beta':b
                 ,'topic_lists':x.topic_lists.values[0]
                  #iter is number of times LDA was run on this talk
                 ,'iter':i
                 ,'time (sec)':time.time() - start
                 ,'log_probs':gs.logprobs
                 ,'fifty_words':np.array(gs.topterms(n_terms=5))
                 }
            counter += 1
        except:
            errors += 1
            message = str({'File_num':file_num
             ,'Title':x.Title
             ,'Speaker':x.Speaker
             ,'Year':x.Year
             ,'Month':x.Month })
            traceback.print_exc()
            print(message, 'failed')
    if (counter % 5) == 0:
        print(f'done with counter = {counter},or {np.round(counter / total_times * 100,1)}% done')
        print(f'its been running for {np.round((time.time() - beginning) / 60,2)} minutes')
print(f'there were {errors} errors')

pd.DataFrame(results).T.to_json(output_filename)
