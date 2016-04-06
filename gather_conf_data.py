
import os, sys
import subprocess

conf_list = ['PLDI', 'WSDM']

for c in conf_list:
    cmd = "python export_citations.py " + c
    os.system(cmd)
    #subprocess.call(cmd, shell=True)

    cmd2 = "python construct_citation_table.py " + c
    os.system(cmd2)
    #subprocess.call(cmd2, shell=True)
