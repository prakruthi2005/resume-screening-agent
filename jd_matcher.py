import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import numpy as np
import re

class JDMatcher:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        self.vector_store = None
    
    def create_jd_embedding(self, job_description: str):
        """Create embedding for job description"""
        documents = [Document(page_content=job_description)]
        self.vector_store = Chroma.from_documents(
            documents, 
            self.embeddings,
            collection_name="job_description"
        )
    
    def calculate_similarity(self, resume_text: str) -> float:
        """Calculate similarity between resume and JD"""
        if not self.vector_store:
            raise ValueError("Job description not set. Call create_jd_embedding first.")
        
        resume_embedding = self.embeddings.embed_query(resume_text)
        jd_embedding = self.vector_store._collection.get(include=['embeddings'])['embeddings'][0]
        
        similarity = self._cosine_similarity(resume_embedding, jd_embedding)
        return similarity
    
    def analyze_fit(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Comprehensive analysis of resume fit using LLM"""
        
        prompt_template = ChatPromptTemplate.from_template(
            """
            Analyze how well this resume matches the job description.
            
            JOB DESCRIPTION:
            {job_description}
            
            RESUME EXTRACT:
            {resume_text}
            
            RESUME SKILLS: {skills}
            EXPERIENCE: {experience}
            EDUCATION: {education}
            
            Please provide:
            1. Overall match score (0-100)
            2. Key strengths (3-4 bullet points)
            3. Missing qualifications (2-3 bullet points)
            4. Recommendation (Strong Yes/Yes/Maybe/No)
            
            Format your response as:
            Score: [number]
            Strengths: [bullet points]
            Missing: [bullet points]
            Recommendation: [Strong Yes/Yes/Maybe/No]
            """
        )
        
        prompt = prompt_template.format(
            job_description=job_description,
            resume_text=resume_data['cleaned_text'][:2000],
            skills=", ".join(resume_data['skills']),
            experience=resume_data['experience'],
            education=", ".join(resume_data['education'])
        )
        
        response = self.llm.predict(prompt)
        return self._parse_llm_response(response)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        lines = response.split('\n')
        result = {
            'score': 0,
            'strengths': [],
            'missing': [],
            'recommendation': 'Maybe'
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Score:'):
                try:
                    score_match = re.search(r'(\d+)', line)
                    if score_match:
                        result['score'] = int(score_match.group(1))
                except:
                    pass
            elif line.startswith('Strengths:'):
                result['strengths'] = self._extract_bullet_points(lines, lines.index(line))
            elif line.startswith('Missing:'):
                result['missing'] = self._extract_bullet_points(lines, lines.index(line))
            elif line.startswith('Recommendation:'):
                result['recommendation'] = line.replace('Recommendation:', '').strip()
        
        return result
    
    def _extract_bullet_points(self, lines: List[str], start_index: int) -> List[str]:
        """Extract bullet points from LLM response"""
        bullets = []
        for i in range(start_index + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            if line.startswith(('Score:', 'Strengths:', 'Missing:', 'Recommendation:')):
                break
            if line.startswith(('-', '•', '*')):
                bullets.append(line[1:].strip())
            else:
                bullets.append(line)
        return bullets
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))