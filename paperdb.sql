CREATE TABLE paper_pruned(id TEXT, year INTEGER, venueid TEXT);
.separator "," 
.import ./data_txt/Papers_pruned.txt paper_pruned
CREATE UNIQUE INDEX id_index ON paper_pruned (id);
