#!/bin/bash

# a few leftover conferences

array=( S&P HiPC )
for i in "${array[@]}"
do
	python export_citations.py $i
	python construct_citation_table.py $i
done