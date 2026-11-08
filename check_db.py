import sqlite3

conn = sqlite3.connect("data/instamart.db")

tables = ["customers", "products", "orders"]

for table in tables:
    print("\nTABLE:", table)

    columns = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    for column in columns:
        print(column[1])

conn.close()