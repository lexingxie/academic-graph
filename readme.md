
workflow for a new conference 

0) prep Paper.db
first run prune_papers.ipynb
then import the result to sqlite 
sqlite> create table paper_pruned(id text, year integer, venueid string);                │xlx@braun:~/d2/MicrosoftAcademicGraph$ 
sqlite> .separator ","                                                                   │xlx@braun:~/d2/MicrosoftAcademicGraph$ 
sqlite> .import ./data_txt/Papers_pruned.txt paper_pruned     

1) get subset for its published papers:

xlx@braun:/data2/xlx/MicrosoftAcademicGraph$ grep WSDM data_txt/ConferenceSeries.txt    
42C7B402        WSDM    Web Search and Data Mining                                       
xlx@braun:/data2/xlx/MicrosoftAcademicGraph$ grep 42C7B4025 data_txt/Papers.txt > papers.WSDM.txt                                                

2) get its citings and cited record
python export_citations.py WSDM

3) do the necessary joins 
python construct_citation_table.py MM