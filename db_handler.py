import sqlite3
import json
from typing import Dict, Any, List
from datetime import datetime

class DatabaseHandler:
    def __init__(self, db_path: str = "resume_screening.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS screening_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resume_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                filename TEXT,
                final_score REAL,
                llm_score REAL,
                similarity_score REAL,
                skills_count INTEGER,
                recommendation TEXT,
                analysis_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES screening_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_screening_session(self, job_description: str, resumes: List[Dict[str, Any]]) -> int:
        """Save screening session to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO screening_sessions (job_description) VALUES (?)",
            (job_description,)
        )
        session_id = cursor.lastrowid
        
        for resume in resumes:
            cursor.execute('''
                INSERT INTO resume_results 
                (session_id, filename, final_score, llm_score, similarity_score, skills_count, recommendation, analysis_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                resume['filename'],
                resume['final_score'],
                resume['analysis']['score'],
                resume['similarity_score'],
                len(resume['processed_data']['skills']),
                resume['analysis']['recommendation'],
                json.dumps(resume['analysis'])
            ))
        
        conn.commit()
        conn.close()
        return session_id