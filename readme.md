

This repo contains the scripts to process Microsoft Academic Graph, 
in order to profile the citation influence and reference heritage of a publication venue (e.g. conferences). 


### developer workflow to analyze a new conference /venue

0) prep Paper.db (once for each new version of MAG data)
first run prune_papers.ipynb

then import the result to sqlite 

	sqlite> create table paper_pruned(id TEXT, year INTEGER, venueid TEXT);                

	sqlite> .separator ","                                                                   
	
	sqlite> .import ./data_txt/Papers_pruned.txt paper_pruned  

or 
sqlite3 Papers.db < paperdb.sql

note: 
75M+ papers with unknown venues among 120M in all   (jan 2016)
73M+ papers with unknown venues among 126M in all   (apr 2016)
                                              
1) get its citings and cited record (~30 mins)
python export_citations.py WSDM

prep-step: [_This is arleady been done in export_citations.py below_] get subset for its published papers:

xlx@braun:/data2/xlx/MicrosoftAcademicGraph$ grep WSDM data_txt/ConferenceSeries.txt    
42C7B402        WSDM    Web Search and Data Mining                                       
xlx@braun:/data2/xlx/MicrosoftAcademicGraph$ grep 42C7B4025 data_txt/Papers.txt > papers.WSDM.txt

2) do the necessary joins (can take a few hrs)
python construct_citation_table.py MM
