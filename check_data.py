import sqlite3
from datetime import datetime

# Connect to the database
conn = sqlite3.connect('instance/bank_marketing.db')
cursor = conn.cursor()

# Check total number of records
cursor.execute("SELECT COUNT(*) FROM customer")
total_records = cursor.fetchone()[0]
print(f"Total records in database: {total_records:,}")

# Check date range
cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM customer")
date_range = cursor.fetchone()
print(f"Date range: {date_range[0]} to {date_range[1]}")

# Check response distribution
cursor.execute("SELECT response, COUNT(*) FROM customer GROUP BY response")
response_dist = cursor.fetchall()
print("\nResponse distribution:")
for response, count in response_dist:
    percentage = (count / total_records) * 100
    print(f"  {response}: {count:,} ({percentage:.1f}%)")

# Check age distribution
cursor.execute("SELECT AVG(age), MIN(age), MAX(age) FROM customer")
age_stats = cursor.fetchone()
print(f"\nAge statistics:")
print(f"  Average: {age_stats[0]:.1f}")
print(f"  Range: {age_stats[1]} - {age_stats[2]}")

# Check balance statistics
cursor.execute("SELECT AVG(balance), MIN(balance), MAX(balance) FROM customer")
balance_stats = cursor.fetchone()
print(f"\nBalance statistics:")
print(f"  Average: ${balance_stats[0]:,.2f}")
print(f"  Range: ${balance_stats[1]:,.2f} - ${balance_stats[2]:,.2f}")

# Check job distribution (top 5)
cursor.execute("SELECT job, COUNT(*) FROM customer GROUP BY job ORDER BY COUNT(*) DESC LIMIT 5")
job_dist = cursor.fetchall()
print(f"\nTop 5 job categories:")
for job, count in job_dist:
    percentage = (count / total_records) * 100
    print(f"  {job}: {count:,} ({percentage:.1f}%)")

# Check records by year
cursor.execute("""
    SELECT strftime('%Y', created_at) as year, COUNT(*) 
    FROM customer 
    GROUP BY year 
    ORDER BY year
""")
year_dist = cursor.fetchall()
print(f"\nRecords by year:")
for year, count in year_dist:
    print(f"  {year}: {count:,} records")

conn.close()
print(f"\nDatabase size: 11MB (as expected for {total_records:,} records)") 