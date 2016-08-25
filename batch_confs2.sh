#!/bin/bash

array=( CCS HPDC ICS MOBICOM MobiSys SenSys EuroSys )
for i in "${array[@]}"
do
	python export_citations.py $i
	python construct_citation_table.py $i
done