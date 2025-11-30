import os
import PyPDF2
from docx import Document
from typing import Dict, List, Any
import re

class ResumeProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from resume files"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_ext == '.docx':
            return self._extract_from_docx(file_path)
        elif file_ext == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume text into structured data"""
        return {
            'raw_text': text,
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'cleaned_text': self._clean_text(text)
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        common_skills = [
            'python', 'java', 'javascript', 'sql', 'machine learning', 'ai',
            'deep learning', 'tensorflow', 'pytorch', 'react', 'node.js',
            'aws', 'docker', 'kubernetes', 'git', 'rest api', 'mongodb',
            'postgresql', 'mysql', 'html', 'css', 'typescript', 'angular',
            'vue', 'django', 'flask', 'fastapi', 'spring', 'hibernate',
            'jenkins', 'ansible', 'terraform', 'gcp', 'azure', 'linux',
            'unix', 'bash', 'shell', 'power bi', 'tableau', 'excel',
            'agile', 'scrum', 'jira', 'confluence', 'devops', 'ci/cd'
        ]
        
        found_skills = []
        text_lower = text.lower()
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_experience(self, text: str) -> str:
        """Extract experience information"""
        experience_patterns = [
            r'(\d+)\s*years?[\s\w]*experience',
            r'experience[\s\w]*(\d+)\s*years?',
            r'(\d+)\s*\+?\s*years?'
        ]
        
        for pattern in experience_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return "Experience not specified"
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'b\.tech', 'm\.tech', 'be', 'me',
            'bsc', 'msc', 'mbbs', 'bca', 'mca', 'associate', 'diploma'
        ]
        
        education = []
        for keyword in education_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                education.append(keyword)
        
        return education
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()