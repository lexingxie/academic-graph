CREATE TABLE paper_pruned(id TEXT PRIMARY KEY, year INTEGER, venueid TEXT);
.separator "," 
.import ./data_txt/Papers_pruned.txt paper_pruned