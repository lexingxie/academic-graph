
import os, sys
import subprocess

conf_list = ['PLDI', 'WSDM']

for c in conf_list:
    cmd = "./export_citations.py " + c
    subprocess.call(cmd, shell=True)

    cmd2 = "./construct_citation_table.py " + c
    subprocess.call(cmd2, shell=True)