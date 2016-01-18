

import sqlite3
import os, sys

data_dir = '/Users/xlx/Downloads/graph-data'
db_file = os.path.join(data_dir, 'AcademicGraph.db')
schema_file = os.path.join(data_dir, "readme.txt")

def creat_table(table_name, db_file, schema_file):
	field_list = read_schema_file(schema_file, table_name)

	conn = sqlite3.connect(db_file)
	c = conn.cursor()

	conn.commit()
	conn.close()


def read_schema_file(schema_file, table_name):
	fh = open(schema_file, 'rt')
	while 
	fh.close()