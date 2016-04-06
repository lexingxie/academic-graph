
# coding: utf-8

# In[2]:

import os, sys
#import pandas as pd
from datetime import datetime 
import numpy as np
#import pickle
#import warnings

#data_dir = '/Users/xlx/Downloads/graph-data'
data_dir = '/home/xlx/d2/MicrosoftAcademicGraph'


# In[4]:

paper_file = os.path.join(data_dir, 'data_txt', 'Papers.txt')
out_file = os.path.join(data_dir, 'data_txt', 'Papers_pruned.txt')
print( '{} start reading {} ... '.format(datetime.now(), paper_file))

out_fh = open(out_file, 'wt')
line_cnt = 0
empty_venue = 0
dup_venue = 0
with open(paper_file, 'rt') as fh:
    for line in fh:
        row = line.strip().split('\t')
        """ paper table columns
            'PaperID', 'TitleOrig', 'TitleNorm', 'PubYear', 'PubDate', 
            'DOI', 'VenueOrig', 'VenueNorm', 'JournalID', 'ConfID', 'PaperRank'
        """
        line_cnt += 1
        if len(row[8])==0 and len(row[9])==0:
            out_str = "{},{},{}\n".format(row[0], row[3], '') 
            empty_venue += 1
        elif len(row[8])==0: # no journal ID, write conf
            out_str = "{},{},{}\n".format(row[0], row[3], row[9])
        elif len(row[9])==0: # no conf ID, write journal
            out_str = "{},{},{}\n".format(row[0], row[3], row[8]) 
        else:  # has both, write conf
            out_str = "{},{},{}\n".format(row[0], row[3], row[9]) 
            dup_venue += 1
        
        out_fh.write(out_str)
    
        if line_cnt % 1000000 == 0 : # 2000000
            print('{} {:9.0f} lines; venues - {} missing {} dups - '.format( 
                    datetime.now(), line_cnt, empty_venue, dup_venue) +out_str.strip() )
        if line_cnt >= 1e9: #1e9:
            break

out_fh.close()
print('{} {:9.0f} lines; venues - {} missing {} dups - DONE.'.format(datetime.now(), line_cnt, empty_venue, dup_venue) )


# In[ ]:



