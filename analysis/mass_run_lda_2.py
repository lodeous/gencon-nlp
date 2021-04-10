import sys
import traceback
import numpy as np
from ldacgs import *
import pandas as pd
import numpy as np
import os,glob
from matplotlib import pyplot as plt
import time

"""
INSTEAD OF GOING THROUGH TOPICS - JUST LEAVE IT RUNNING throughout the night, so maybe it should work backwards
"""
path = 'other_files/'
f = 'merged_summary_topics.json'
repeat_lda_times = 5
num_hours_to_run = 11
directory = '3mass_json'
# num_hours_to_run = 1 / 30 #for testing 60/30 = 2 minutes
params = [(0.001,1)
    ,(0.005,1)
    ,(0.001,0.17)]
STOP_WORDS_LOC = '../3winter21/Gibbs_LDA/' + 'stopwords.txt'
STOP_WORDS_LOC = 'stopwords.txt'


new = pd.read_json(path + f)
# my data is in data2 not data
new['File'] = [x[5:] for x in new.File]
new['File_num'] = [int(x[:-4]) for x in new.File]
new['tag_count'] = [len(x) for x in new.topic_lists]

files = (new['File_num'][::-1])[2050:]
# raise NotImplementedError('pick alpha and beta values')
try:
    os.mkdir(directory)
except:
    pass

results = dict()
beginning = time.time()
errors = 0
counter = 1
num_seconds_in_min = 60
num_min_in_hour = 60
output_file_count = 0
total_times = len(files) * repeat_lda_times
print('about to start loop')
for file_num in files:
    for i in range(repeat_lda_times):
        #so we don't have to iterate through the whole thing if we don't want to
        if (time.time() - beginning) / (num_seconds_in_min * num_min_in_hour ) < num_hours_to_run:
            # avoid out of error with accessing betas
            j = (i % len(params))
            a,b = params[j]
            try:
                x = new.loc[new.File_num == file_num]
                start = time.time()
                #gs for gibbs sampler
                gs = LDACGS(n_topics=10,alpha=a,beta=b)
                # gs.buildCorpus(filename='reagan.txt')
                # gs.initialize()
                # gs._sweep()

                gs.sample(filename='data2/' + new.loc[new.File_num == file_num].File.values[0]
                        , stopwords_file=STOP_WORDS_LOC)

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
            except:
                errors += 1
                message = str({'File_num':file_num
                 ,'Title':x.Title.values[0]
                 ,'Speaker':x.Speaker.values[0]
                 ,'Year':x.Year.values[0]
                 ,'Month':x.Month.values[0]})
                traceback.print_exc()
                print(message, 'failed')
            finally:
                counter += 1
        else:
            break
        if (counter % 7) == 0:
            print(f'done with counter = {counter},or {np.round(counter / total_times * 100,1)}% done')
            print(f'its been running for {np.round((time.time() - beginning) / (60 * 60),2)} hours')
        #if it outputs a file at each
        if (counter % 500) == 0:
            pd.DataFrame(results).T.to_json(f'{directory}/first_mass_lda_at_{output_file_count}.json')
            output_file_count += 1
            print(f'there were {errors} errors by counter = {counter}, or {np.round(errors / counter * 100,2)}%')

final_filename = f'{directory}/first_mass_lda_at_{output_file_count}.json'
pd.DataFrame(results).T.to_json(final_filename)
print('exported:',final_filename,'as final file')
print(f'there were {errors} errors by counter = {counter}, or {np.round(errors / counter * 100,2)}%')
print('done with complete program')
