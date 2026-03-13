import sqlite3
from datetime import datetime

DB_NAME = "radiology.db"

def connect():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_tables():

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors(
        doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients(
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans(
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        scan_path TEXT,
        prediction TEXT,
        confidence REAL,
        scan_date TEXT
    )
    """)

    conn.commit()

    cursor.execute("SELECT * FROM doctors")

    if not cursor.fetchall():
        cursor.execute(
            "INSERT INTO doctors(username,password) VALUES(?,?)",
            ("doctor","123")
        )

    conn.commit()
    conn.close()

def verify_doctor(username,password):

    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM doctors WHERE username=? AND password=?",
        (username,password)
    )

    doctor = cursor.fetchone()

    conn.close()

    return doctor

def add_patient(name,age,gender):

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patients(name,age,gender,created_at)
    VALUES(?,?,?,?)
    """,(name,age,gender,str(datetime.now())))

    conn.commit()
    conn.close()

def get_patients():

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients")

    patients = cursor.fetchall()

    conn.close()

    return patients

def add_scan(patient_id,path,prediction,confidence):

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO scans(patient_id,scan_path,prediction,confidence,scan_date)
    VALUES(?,?,?,?,?)
    """,(patient_id,path,prediction,confidence,str(datetime.now())))

    conn.commit()
    conn.close()

def get_scans(patient_id):

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT scan_path,prediction,confidence,scan_date
    FROM scans
    WHERE patient_id=?
    ORDER BY scan_date DESC
    """,(patient_id,))

    scans = cursor.fetchall()

    conn.close()

    return scans

def get_last_two_scans(patient_id):

    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT scan_path
    FROM scans
    WHERE patient_id=?
    ORDER BY scan_date DESC
    LIMIT 2
    """,(patient_id,))

    scans = cursor.fetchall()

    conn.close()

    return scans

def reset_password(username, new_password):

    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE doctors SET password=? WHERE username=?",
        (new_password, username)
    )

    conn.commit()
    conn.close()