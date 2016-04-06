import os, sys
import pandas as pd
from datetime import datetime 
import numpy as np
import pickle
import warnings

#data_dir = '/Users/xlx/Downloads/graph-data'
data_dir = '/home/xlx/d2/MicrosoftAcademicGraph'

load_ref = lambda fn: pd.read_table(fn, header=None, names=['PaperID', 'RefID'])

conf_file = os.path.join(data_dir, 'data_txt', 'Conferences.txt')
conf_df = pd.read_table(conf_file, header=None, names=['ConfID', 'Abbrv', 'FullName'])

conf_list = ['MM', 'CVPR', 'NIPS', 'ICML', 'IJCAI', 'PLDI']


paper_file = os.path.join(data_dir, 'data_txt', 'Papers.txt')
print( '{} start reading {} ... '.format(datetime.now(), paper_file))

with open(paper_file, 'rt') as fh:
    paper_buf = fh.read()
print( '{} load ref db {} bytes'.format(datetime.now(), sys.getsizeof(paper_buf)) )

#c = conf_list[0]
c = sys.argv[1]

row = conf_df.loc[conf_df['Abbrv'] == c]
conf_id = list(row['ConfID'])[0]

conf_paper_file = os.path.join(data_dir, 'papers.'+ c +'.txt')
df_paper = pd.read_table(conf_paper_file, header=None, 
                         names=['PaperID', 'TitleOrig', 'TitleNorm', 'PubYear', 'PubDate', 
                               'DOI', 'VenueOrig', 'VenueNorm', 'JournalID', 'ConfID', 'PaperRank' ])
df_paper.head()
set_paper = set(df_paper['PaperID'])

citing_file = os.path.join(data_dir, 'citing.'+c+'.txt')
df_citing = load_ref(citing_file)
cited_file = os.path.join(data_dir, 'cited.'+c+'.txt')
df_cited = load_ref(cited_file)
print ("{} conference {}: {} papers ({}-{})".format(datetime.now(), c, df_paper['PaperID'].count(), 
                                               df_paper['PubYear'].min(), df_paper['PubYear'].max()))
print ("\t citing {} papers, cited by {}".format(df_citing['PaperID'].count(), df_cited['PaperID'].count()))

# left joins for both the citing and cited
dfx_citing = df_citing.merge(df_paper[['PaperID', 'PubYear', 'ConfID']], on='PaperID', how='left') 
dfx_citing = dfx_citing.rename(columns = {'PubYear':'PaperPubYear', 'ConfID':"PaperConfID"})
dfx_citing['RefPubYear'] = 1000
dfx_citing['RefVenueID'] = 'AAAAaaaa'

dfx_cited = df_cited.merge(df_paper[['PaperID', 'PubYear', 'ConfID']], 
                           left_on="RefID", right_on='PaperID', how='left') 
dfx_cited.drop('PaperID_y', axis=1, inplace=True)
dfx_cited = dfx_cited.rename(columns = {'PubYear':'RefPubYear', 'ConfID':"RefConfID", "PaperID_x":"PaperID"})
dfx_cited['PaperPubYear'] = 1000
dfx_cited['PaperVenueID'] = 'AAAAaaaa'

ptr = 0 
line_cnt = 0
citing_cnt = [0, 0]
cited_cnt = [0, 0]
while ptr < len(paper_buf):        

    eol = paper_buf.find('\n', ptr)
    row = paper_buf[ptr:eol].split('\t')
    line_cnt += 1
    ptr = eol + 1
    """ paper table columns
        'PaperID', 'TitleOrig', 'TitleNorm', 'PubYear', 'PubDate', 
        'DOI', 'VenueOrig', 'VenueNorm', 'JournalID', 'ConfID', 'PaperRank'
    """

    cur_pid = row[0]
    r_ref = list(np.nonzero(dfx_citing['RefID'] == cur_pid)[0])
    r_paper  = list(np.nonzero(dfx_cited['PaperID'] == cur_pid)[0])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        if len(r_ref) >0:
            citing_cnt[0] += 1
            for rid in r_ref:
                # for each paper being cited by any paper in Conf
                dfx_citing['RefPubYear'][rid] = row[3]
                citing_cnt[1] += 1
                if row[9]: # conference
                    dfx_citing['RefVenueID'][rid] = row[9]
                elif row[8]: # journal
                    dfx_citing['RefVenueID'][rid] = row[8]

        if len(r_paper) >0:
            cited_cnt[0] += 1
            for rid in r_paper:
                cited_cnt[1] += 1
                # for each paper citing by any paper in Conf
                dfx_cited['PaperPubYear'][rid] = row[3]
                if row[9]: # conference
                    dfx_cited['PaperVenueID'][rid] = row[9]
                elif row[8]:
                    dfx_cited['PaperVenueID'][rid] = row[8]
        

    if line_cnt % 5000 == 0 : # 2000000
        print('{} {:9.0f} lines; citing {:6.0f}, {:6.0f} unique; {:6.0f} cited, {:6.0f} unique'.format(
                datetime.now(), line_cnt, citing_cnt[1], citing_cnt[0], cited_cnt[1], cited_cnt[0]) )
    if line_cnt >= 1e9: #1e9:
        break


pickle.dump({"name":c, 'citing':dfx_citing, "cited":dfx_cited, "paper":df_paper}, 
           os.path.join(data_dir, 'cite_records.'+c+".pkl"))
print('{} {:9.0f} lines; citing {:6.0f}, {:6.0f} unique; {:6.0f} cited, {:6.0f} unique\n\n'.format(
                datetime.now(), line_cnt, citing_cnt[1], citing_cnt[0], cited_cnt[1], cited_cnt[0]) )
