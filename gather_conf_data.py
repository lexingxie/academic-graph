
import os, sys
import subprocess

conf_list = ['PLDI', 'WSDM']

for c in conf_list:
    cmd = "python export_citations.py " + c
    subprocess.call(cmd)

    cmd2 = "python construct_citation_table.py " + c
    subprocess.call(cmd2)