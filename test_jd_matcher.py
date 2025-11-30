import unittest
import os
from app.agent.jd_matcher import JDMatcher

class TestJDMatcher(unittest.TestCase):
    def setUp(self):
        # Note: This test requires OPENAI_API_KEY to be set
        api_key = os.getenv('OPENAI_API_KEY', 'test-key')
        self.matcher = JDMatcher(api_key)
    
    def test_response_parsing(self):
        sample_response = """
        Score: 85
        Strengths: - Strong Python experience
        - Good communication skills
        Missing: - Cloud experience
        - Team leadership
        Recommendation: Yes
        """
        
        parsed = self.matcher._parse_llm_response(sample_response)
        self.assertEqual(parsed['score'], 85)
        self.assertEqual(parsed['recommendation'], 'Yes')
        self.assertIn('Strong Python experience', parsed['strengths'])

if __name__ == '__main__':
    unittest.main()