import sqlite3
from datetime import datetime

import tkinter as tk
from tkinter import ttk


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
    
    # taking all the medicine from the database
    def get_all_medicines(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM medicines ORDER BY medicine_name')
        medicines = cursor.fetchall()
        
        conn.close()
        return medicines
    

    # ---- 
    
    # schedule for medicine intake
    def add_schedule(self, medicine_id, time_of_day, with_food="No preference", special_instructions=""):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO intake_schedule (medicine_id, time_of_day, with_food, special_instructions)
            VALUES (?, ?, ?, ?)
        ''', (medicine_id, time_of_day, with_food, special_instructions))
        
        conn.commit()
        conn.close()


    # removing the schedule for a medicine
    def delete_schedule(self, schedule_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM intake_schedule WHERE id = ?', (schedule_id,))
        
        conn.commit()
        conn.close()

    # getting schedule for a medicine
    def get_schedules_for_medicine(self, medicine_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM intake_schedule WHERE medicine_id = ?', (medicine_id,))
        schedules = cursor.fetchall()
        
        conn.close()
        return schedules
    
    #getting current schedule of medicines based on time
    def get_current_schedule(self):
        current_time = datetime.now().strftime("%H:%M")
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.*, s.time_of_day, s.with_food, s.special_instructions
            FROM medicines m
            JOIN intake_schedule s ON m.id = s.medicine_id
            ORDER BY s.time_of_day
        ''')
        
        schedules = cursor.fetchall()
        conn.close()
        return schedules
    
    #function log to keep log about medicine intake
    def log_intake(self, medicine_id, scheduled_time, status="Taken"):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO intake_history (medicine_id, scheduled_time, status)
            VALUES (?, ?, ?)
        ''', (medicine_id, scheduled_time, status))
        
        conn.commit()
        conn.close()
    

    # --------

    #searching medicine by name or other ingredients 
    def search_medicine_by_name(self, search_term):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        search_pattern = f"%{search_term}%"
        cursor.execute('''
            SELECT * FROM medicines 
            WHERE medicine_name LIKE ? OR active_ingredients LIKE ?
            ORDER BY medicine_name
        ''', (search_pattern, search_pattern))
        
        medicines = cursor.fetchall()
        conn.close()
        return medicines
    

    # gui --------


class MedicineDatabaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Database Manager")
        self.root.geometry("1000x700")
        
        self.setup_ui()
        
        self.setup_list_tab()
    
    def setup_ui(self):

        # Main container
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Medicine List tab
        self.list_tab = tk.Frame(self.notebook)
        self.notebook.add(self.list_tab, text="Medicine List")
        
        # Adding or Editing the Medicine tab
        self.edit_tab = tk.Frame(self.notebook)
        self.notebook.add(self.edit_tab, text="Add or Edit Medicine")
        
        # Intake schedule managing tab
        self.schedule_tab = tk.Frame(self.notebook)
        self.notebook.add(self.schedule_tab, text="Intake Schedule")
        
        # The status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_list_tab(self):
        # Search frame
        search_frame = tk.Frame(self.list_tab)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_entry = tk.Entry(search_frame, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', lambda e: self.search_medicines())
        
        tk.Button(search_frame, text="Clear", command=self.clear_search).pack(side=tk.LEFT, padx=5)
        tk.Button(search_frame, text="Refresh", command=self.refresh_medicine_list).pack(side=tk.LEFT, padx=5)
        
        # Medicine list frame
        list_frame = tk.Frame(self.list_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tree view for medicine list
        columns = ("ID", "Name", "Dosage", "Form", "Frequency")
        self.medicine_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.medicine_tree.heading(col, text=col)
            if col == "ID":
                self.medicine_tree.column(col, width=50)
            elif col == "Name":
                self.medicine_tree.column(col, width=200)
            else:
                self.medicine_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.medicine_tree.yview)
        self.medicine_tree.configure(yscroll=scrollbar.set)
        
        self.medicine_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.medicine_tree.bind('<Double-1>', self.on_medicine_double_click)
        
        # All the action buttons
        button_frame = tk.Frame(self.list_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(button_frame, text="View Details", command=self.view_medicine_details, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Edit", command=self.edit_selected_medicine,bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Delete", command=self.delete_selected_medicine,bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Displaying the medicine details
        details_frame = tk.LabelFrame(self.list_tab, text="Medicine Details", padx=10, pady=10)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        from tkinter import scrolledtext
        self.details_text = scrolledtext.ScrolledText(details_frame, height=8, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)

    #Buttons functionalities will add how they work but for now just for asthetics and design confirmation
    def refresh_medicine_list(self):
        pass
    
    def search_medicines(self):
        pass
    
    def clear_search(self):
        pass
    
    def on_medicine_double_click(self, event):
        pass
    
    def view_medicine_details(self):
        pass
    
    def edit_selected_medicine(self):
        pass
    
    def delete_selected_medicine(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicineDatabaseGUI(root)
    root.mainloop()