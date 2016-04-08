
import os, sys
import subprocess

conf_list = sys.argv[1].strip().split(',') #['PLDI', 'WSDM']

for c in conf_list:
    cmd = "python export_citations.py " + c
    print(cmd + "\n")
    os.system(cmd)
    #subprocess.call(cmd, shell=True)

    cmd2 = "python construct_citation_table.py " + c
    print(cmd2+"\n")
    os.system(cmd2)
    #subprocess.call(cmd2, shell=True)
