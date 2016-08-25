#!/bin/bash

array=( OSDI CAV LICS SOSP POPL ASE FSE UbiComp IROS )
for i in "${array[@]}"
do
	python export_citations.py $i
	python construct_citation_table.py $i
done