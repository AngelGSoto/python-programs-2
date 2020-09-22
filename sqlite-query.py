'''
Author: Luis A. Gutiérrez
11/09/2020
Testing query on sqlite3
Based on https://stuartsplace.com/computing/programming/python/python-and-sqlite-exporting-data
Please consult it
'''
import sqlite3
import os
import csv

# Create a SQL connection to our SQLite database
conn = sqlite3.connect('MCcatalog.db')

cur = conn.cursor()

#  query
qry = ("SELECT * FROM SMCcatalog WHERE R_aper_3 <= 20.5 AND e_R_aper_3 <= 0.2 AND e_F660_aper_3 <= 0.2 AND e_I_aper_3 <= 0.2 AND R_aper_3 - F660_aper_3 >= 0.43*(R_aper_3 - I_aper_3) + 0.65 AND R_aper_3 - F660_aper_3 <= -6.8*(R_aper_3 - I_aper_3) - 1.3 AND F515_aper_3 - F660_aper_3 >= 0.3 AND F515_aper_3 - F660_aper_3 >= 2.7*(F515_aper_3 - F861_aper_3) + 2.15 AND G_aper_3 - F515_aper_3 <= 0.12*(F660_aper_3 - R_aper_3) - 0.01 AND G_aper_3 - F515_aper_3 <= -1.1*(F660_aper_3 - R_aper_3) - 1.07 AND Z_aper_3 - F660_aper_3 >= 0.2319*(Z_aper_3 - G_aper_3) + 0.85 AND Z_aper_3 - F660_aper_3 >= -1.3*(Z_aper_3 - G_aper_3) + 1.7 AND F410_aper_3 - F660_aper_3 >= 8.0*(G_aper_3 - I_aper_3) + 4.5 AND F410_aper_3 - F660_aper_3 >= 0.8*(G_aper_3 - I_aper_3) + 0.55;")
#for row in cur.execute(qry):
    #print(row)
#PhotoFlag_F660 <= 3.0 AND R_aper_3 <= 19 AND e_R_aper_3 <= 0.2 AND e_F660_aper_3 <= 0.2 AND e_I_aper_3 <= 0.2

cur.execute(qry)
data = cur.fetchall()
#print(data)

# Extract the table headers
headers = [i[0] for i in cur.description]

# Open CSV file for writing
file_name = 'PneCand-smc-color-viirfilErrorR205.csv'
csv_file = csv.writer(open(file_name, 'w', newline=''),
                             delimiter=',', lineterminator='\r\n',
                             quoting=csv.QUOTE_ALL, escapechar='\\')

# Add the headers and data to the CSV file.
csv_file.writerow(headers)
csv_file.writerows(data)

# Message stating export successful.
print("Data export successful ans writting the file: {}.".format(file_name))

# Be sure to close the connection
conn.close()
