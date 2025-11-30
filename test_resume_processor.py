import unittest
import os
import tempfile
from app.agent.resume_processor import ResumeProcessor

class TestResumeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ResumeProcessor()
    
    def test_skill_extraction(self):
        sample_text = "I have experience with Python, JavaScript, and machine learning."
        skills = self.processor._extract_skills(sample_text)
        self.assertIn('python', skills)
        self.assertIn('javascript', skills)
        self.assertIn('machine learning', skills)
    
    def test_experience_extraction(self):
        sample_text = "I have 5 years of experience in software development."
        experience = self.processor._extract_experience(sample_text)
        self.assertIn('5 years', experience.lower())

if __name__ == '__main__':
    unittest.main()