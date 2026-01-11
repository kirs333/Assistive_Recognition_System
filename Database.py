import sqlite3
from datetime import datetime

class MedicineDatabase:
    def __init__(self, db_name="medicine_db.sqlite"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self): #initializing the sqlite database with tables
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Creating the medicines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medicines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_name TEXT NOT NULL,
                dosage TEXT,
                form TEXT,
                frequency TEXT,
                notes TEXT,
                active_ingredients TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Createing the intake schedule table  
        #putting python string literals
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS intake_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                time_of_day TEXT NOT NULL,
                with_food TEXT,
                special_instructions TEXT,
                FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE CASCADE
            )
        ''')
        
        # Creating the intake history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intake_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medicine_id INTEGER NOT NULL,
                taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scheduled_time TEXT,
                status TEXT,
                FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()


    def add_medicine(self, name, dosage="", form="", frequency="", notes="", active_ingredients=""):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO medicines (medicine_name, dosage, form, frequency, notes, active_ingredients)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, dosage, form, frequency, notes, active_ingredients))
        
        medicine_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return medicine_id
    

    def delete_medicine(self, medicine_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM medicines WHERE id = ?', (medicine_id,))
        conn.commit()
        conn.close()

    # updating details of a medicine
    def update_medicine(self, medicine_id, name, dosage, form, frequency, notes, active_ingredients):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE medicines 
            SET medicine_name = ?, dosage = ?, form = ?, frequency = ?, 
                notes = ?, active_ingredients = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (name, dosage, form, frequency, notes, active_ingredients, medicine_id))
        
        conn.commit()
        conn.close()


    # taking the medicine with their id
    def get_medicine_by_id(self, medicine_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM medicines WHERE id = ?', (medicine_id,))
        medicine = cursor.fetchone()
        
        conn.close()
        return medicine

if __name__ == "__main__":
    db = MedicineDatabase()
    print("Database initialized successfully")