import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

query = open("sql/queries.sql").read()

result = pd.read_sql_query(query, conn)

print(result)

conn.close()