#!/usr/bin/env python3
"""
Main entry point for the Resume Screening Agent
"""

import os
import uvicorn
from dotenv import load_dotenv

def main():
    """Main function to run the application"""
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the application.")
    
    print("Starting Resume Screening Agent...")
    print("Backend API: http://localhost:8000")
    print("Frontend UI: Run 'streamlit run frontend/streamlit_app.py'")
    print("--" * 30)
    
    # Start FastAPI server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()