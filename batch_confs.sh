#!/bin/bash

array=( ASPLOS ISCA MICRO INFOCOM NSDI PODS)
for i in "${array[@]}"
do
	python export_citations.py $i
	python construct_citation_table.py $i
done