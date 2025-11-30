from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import tempfile
from agent.resume_processor import ResumeProcessor
from agent.jd_matcher import JDMatcher
from agent.ranking_engine import RankingEngine

app = FastAPI(title="Resume Screening API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
current_jd = ""
jd_matcher = None
ranking_engine = None

@app.get("/")
async def root():
    return {"message": "Resume Screening API", "status": "active"}

@app.post("/api/set-job-description")
async def set_job_description(job_description: dict):
    global current_jd, jd_matcher, ranking_engine
    
    try:
        jd_text = job_description.get("text", "")
        if not jd_text:
            raise HTTPException(status_code=400, detail="Job description text is required")
        
        current_jd = jd_text
        jd_matcher = JDMatcher(os.getenv("OPENAI_API_KEY"))
        jd_matcher.create_jd_embedding(jd_text)
        ranking_engine = RankingEngine(jd_matcher)
        
        return {"message": "Job description set successfully", "status": "success"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting job description: {str(e)}")

@app.post("/api/process-resumes")
async def process_resumes(files: List[UploadFile] = File(...)):
    global jd_matcher, ranking_engine, current_jd
    
    if not jd_matcher or not current_jd:
        raise HTTPException(status_code=400, detail="Job description not set. Please set JD first.")
    
    try:
        processor = ResumeProcessor()
        processed_resumes = []
        
        for file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Process resume
                text = processor.extract_text(tmp_path)
                parsed_data = processor.parse_resume(text)
                
                resume_data = {
                    'filename': file.filename,
                    'raw_text': text,
                    'processed_data': parsed_data,
                    'file_size': len(content)
                }
                
                processed_resumes.append(resume_data)
                
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
            finally:
                # Clean up
                os.unlink(tmp_path)
        
        # Rank resumes
        ranked_resumes = ranking_engine.rank_resumes(processed_resumes, current_jd)
        
        return {
            "status": "success",
            "processed_count": len(ranked_resumes),
            "ranked_resumes": ranked_resumes
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resumes: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "resume_screening_api"}