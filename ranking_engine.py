from typing import List, Dict, Any
import numpy as np
from .jd_matcher import JDMatcher

class RankingEngine:
    def __init__(self, jd_matcher: JDMatcher):
        self.jd_matcher = jd_matcher
    
    def rank_resumes(self, resumes: List[Dict[str, Any]], job_description: str) -> List[Dict[str, Any]]:
        """Rank resumes based on multiple criteria"""
        ranked_resumes = []
        
        for resume in resumes:
            similarity_score = self.jd_matcher.calculate_similarity(
                resume['processed_data']['cleaned_text']
            )
            
            analysis = self.jd_matcher.analyze_fit(
                resume['processed_data'], 
                job_description
            )
            
            final_score = self._calculate_final_score(
                analysis['score'], 
                similarity_score,
                len(resume['processed_data']['skills']),
                resume['processed_data']
            )
            
            ranked_resume = {
                **resume,
                'similarity_score': round(similarity_score * 100, 2),
                'analysis': analysis,
                'final_score': round(final_score, 2),
                'ranking_factors': {
                    'llm_score': analysis['score'],
                    'embedding_similarity': similarity_score * 100,
                    'skills_count': len(resume['processed_data']['skills'])
                }
            }
            
            ranked_resumes.append(ranked_resume)
        
        ranked_resumes.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_resumes
    
    def _calculate_final_score(self, llm_score: float, similarity: float, 
                             skills_count: int, resume_data: Dict[str, Any]) -> float:
        """Calculate weighted final score"""
        weights = {
            'llm_score': 0.5,
            'similarity': 0.3,
            'skills': 0.2
        }
        
        normalized_skills = min(skills_count / 20, 1.0) * 100
        
        final_score = (
            weights['llm_score'] * llm_score +
            weights['similarity'] * similarity * 100 +
            weights['skills'] * normalized_skills
        )
        
        return final_score