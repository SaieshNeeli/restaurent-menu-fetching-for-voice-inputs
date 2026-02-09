import sqlite3
from menu_items import MENU_ITEMS
# from menu_300_items import MENU_ITEMS

def create_database():
    connection = sqlite3.connect("menu_given_new1.db")
    cursor = connection.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS menu_items (
            item_id TEXT PRIMARY KEY,
            text TEXT,
            category TEXT,
            price TEXT,
            type TEXT
        )
    """)

    # Insert data
    for item in MENU_ITEMS:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO menu_items (item_id, text, category, price, type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                item["item_id"],
                item["text"].lower().strip(), 
                item["category"],
                item["price"],
                item["type"]
            ))
        except sqlite3.Error as e:
            print(f"Error inserting {item['item_id']}: {e}")

    connection.commit()
    print(f"Successfully inserted {len(MENU_ITEMS)} items into menu.db")
    
    # Verify count
    cursor.execute("SELECT COUNT(*) FROM menu_items")
    count = cursor.fetchone()[0]
    print(f"Total rows in database: {count}")

    connection.close()

if __name__ == "__main__":
    create_database()
