import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.resume_processor import ResumeProcessor
from app.agent.jd_matcher import JDMatcher
from app.agent.ranking_engine import RankingEngine

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-strong-yes {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .recommendation-yes {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .recommendation-maybe {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .recommendation-no {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'resumes' not in st.session_state:
        st.session_state.resumes = []
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'ranked_resumes' not in st.session_state:
        st.session_state.ranked_resumes = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">🤖 AI Resume Screening Agent</h1>', unsafe_allow_html=True)
    st.markdown("Upload resumes and a job description to automatically rank candidates based on fit.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI analysis",
            value=st.session_state.get('api_key', '')
        )
        
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured")
        
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=250,
            help="Enter the complete job description for matching",
            value=st.session_state.get('job_description', '')
        )
        
        if st.button("💾 Set Job Description", use_container_width=True):
            if job_description.strip():
                st.session_state.job_description = job_description.strip()
                st.success("✅ Job description saved!")
            else:
                st.error("❌ Please enter a job description")
        
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        
        if st.session_state.ranked_resumes:
            total = len(st.session_state.ranked_resumes)
            recommended = len([r for r in st.session_state.ranked_resumes 
                             if r['analysis']['recommendation'] in ['Strong Yes', 'Yes']])
            avg_score = sum(r['final_score'] for r in st.session_state.ranked_resumes) / total
            
            st.metric("Total Resumes", total)
            st.metric("Recommended", recommended)
            st.metric("Avg Score", f"{avg_score:.1f}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Resumes", "🎯 Screening Results", "📊 Analytics", "🚀 Deployment Guide"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_deployment_guide()

def render_upload_tab():
    st.header("📤 Upload Resume Files")
    
    if not st.session_state.api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    if not st.session_state.job_description:
        st.warning("⚠️ Please set a job description in the sidebar to continue.")
        return
    
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) selected for processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Resumes", use_container_width=True):
                process_resumes(uploaded_files)
        
        with col2:
            if st.button("🔄 Clear Results", use_container_width=True):
                st.session_state.resumes = []
                st.session_state.ranked_resumes = []
                st.session_state.processing_complete = False
                st.rerun()

def process_resumes(uploaded_files):
    """Process uploaded resume files"""
    try:
        processor = ResumeProcessor()
        jd_matcher = JDMatcher(st.session_state.api_key)
        ranking_engine = RankingEngine(jd_matcher)
        
        # Set job description
        jd_matcher.create_jd_embedding(st.session_state.job_description)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state.resumes = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                text = processor.extract_text(tmp_path)
                parsed_data = processor.parse_resume(text)
                
                resume_data = {
                    'filename': uploaded_file.name,
                    'raw_text': text,
                    'processed_data': parsed_data,
                    'file_size': uploaded_file.size
                }
                
                st.session_state.resumes.append(resume_data)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Rank resumes
        if st.session_state.resumes:
            st.session_state.ranked_resumes = ranking_engine.rank_resumes(
                st.session_state.resumes, 
                st.session_state.job_description
            )
            st.session_state.processing_complete = True
            st.success(f"✅ Successfully processed and ranked {len(st.session_state.resumes)} resumes!")
        
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

def render_results_tab():
    st.header("🎯 Screening Results")
    
    if not st.session_state.ranked_resumes:
        st.info("ℹ️ Upload resumes and set a job description to see results")
        return
    
    st.subheader(f"📋 Ranked Candidates (Total: {len(st.session_state.ranked_resumes)})")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Minimum Score", 0, 100, 0, key="min_score_filter")
    
    with col2:
        recommendation_filter = st.selectbox(
            "Recommendation",
            ["All", "Strong Yes", "Yes", "Maybe", "No"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Final Score", "Similarity Score", "LLM Score", "Skills Count"]
        )
    
    # Filter resumes
    filtered_resumes = st.session_state.ranked_resumes.copy()
    filtered_resumes = [r for r in filtered_resumes if r['final_score'] >= min_score]
    
    if recommendation_filter != "All":
        filtered_resumes = [r for r in filtered_resumes if r['analysis']['recommendation'] == recommendation_filter]
    
    # Sort resumes
    sort_keys = {
        "Final Score": "final_score",
        "Similarity Score": "similarity_score", 
        "LLM Score": lambda x: x['analysis']['score'],
        "Skills Count": lambda x: len(x['processed_data']['skills'])
    }
    
    filtered_resumes.sort(key=lambda x: x[sort_keys[sort_by]] if isinstance(sort_keys[sort_by], str) 
                         else sort_keys[sort_by](x), reverse=True)
    
    # Display results
    for i, resume in enumerate(filtered_resumes, 1):
        rec_class = f"recommendation-{resume['analysis']['recommendation'].lower().replace(' ', '-')}"
        
        with st.expander(f"#{i} - {resume['filename']} - Score: {resume['final_score']}/100", expanded=i <= 3):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Recommendation:** <span class='{rec_class}'>{resume['analysis']['recommendation']}</span>", 
                           unsafe_allow_html=True)
                
                # Scores
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("LLM Score", f"{resume['analysis']['score']}")
                with score_col2:
                    st.metric("Similarity", f"{resume['similarity_score']}%")
                with score_col3:
                    st.metric("Skills", len(resume['processed_data']['skills']))
                
                # Strengths
                if resume['analysis']['strengths']:
                    st.write("**✅ Strengths:**")
                    for strength in resume['analysis']['strengths'][:3]:
                        st.write(f"• {strength}")
                
                # Missing
                if resume['analysis']['missing']:
                    st.write("**❌ Areas for Improvement:**")
                    for missing in resume['analysis']['missing'][:2]:
                        st.write(f"• {missing}")
            
            with col2:
                st.write("**📊 Extracted Info:**")
                st.write(f"**Experience:** {resume['processed_data']['experience']}")
                st.write(f"**Education:** {', '.join(resume['processed_data']['education']) if resume['processed_data']['education'] else 'Not specified'}")
                st.write(f"**Skills Found:** {len(resume['processed_data']['skills'])}")
                
                # Show top skills
                if resume['processed_data']['skills']:
                    st.write("**Top Skills:**")
                    for skill in resume['processed_data']['skills'][:5]:
                        st.write(f"• {skill.title()}")

def render_analytics_tab():
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.ranked_resumes:
        st.info("ℹ️ Process some resumes to see analytics")
        return
    
    df = pd.DataFrame([{
        'Filename': r['filename'],
        'Final_Score': r['final_score'],
        'LLM_Score': r['analysis']['score'],
        'Similarity_Score': r['similarity_score'],
        'Skills_Count': len(r['processed_data']['skills']),
        'Recommendation': r['analysis']['recommendation'],
        'Experience': r['processed_data']['experience']
    } for r in st.session_state.ranked_resumes])
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", len(df))
    with col2:
        st.metric("Average Score", f"{df['Final_Score'].mean():.1f}")
    with col3:
        strong_candidates = len(df[df['Recommendation'].str.contains('Strong Yes|Yes')])
        st.metric("Recommended Candidates", strong_candidates)
    with col4:
        st.metric("Top Score", f"{df['Final_Score'].max():.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(df, x='Final_Score', nbins=20, 
                          title="Distribution of Final Scores")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recommendations Breakdown")
        rec_counts = df['Recommendation'].value_counts()
        fig = px.pie(values=rec_counts.values, names=rec_counts.index,
                    title="Recommendation Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Skills analysis
    st.subheader("🛠️ Skills Analysis")
    all_skills = []
    for resume in st.session_state.ranked_resumes:
        all_skills.extend(resume['processed_data']['skills'])
    
    if all_skills:
        skills_df = pd.DataFrame({'Skill': all_skills})
        skill_counts = skills_df['Skill'].value_counts().head(10)
        
        fig = px.bar(x=skill_counts.values, y=skill_counts.index, 
                    orientation='h', title="Top 10 Most Common Skills")
        st.plotly_chart(fig, use_container_width=True)

def render_deployment_guide():
    st.header("🚀 Deployment Guide")
    
    st.markdown("""
    ## Quick Deployment Options
    
    ### Option 1: Streamlit Cloud (Recommended)
    1. **Fork/Push** this code to a GitHub repository
    2. **Go to** [share.streamlit.io](https://share.streamlit.io)
    3. **Connect** your GitHub repository
    4. **Set** `OPENAI_API_KEY` in secrets
    5. **Deploy!** 🎉
    
    ### Option 2: Local Production
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # Set environment variable
    export OPENAI_API_KEY="your-api-key-here"
    
    # Run the application
    streamlit run frontend/streamlit_app.py
    ```
    
    ### Option 3: Docker Deployment
    ```dockerfile
    FROM python:3.9-slim
    
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY . .
    EXPOSE 8501
    
    CMD ["streamlit", "run", "frontend/streamlit_app.py"]
    ```
    
    ## Environment Variables
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `DATABASE_URL`: Database connection string (optional)
    
    ## API Endpoints
    The backend API runs on FastAPI with these endpoints:
    - `POST /api/set-job-description` - Set job description
    - `POST /api/process-resumes` - Process and rank resumes
    - `GET /api/health` - Health check
    
    ## Support
    For issues or questions, check the README.md file or contact the development team.
    """)

if __name__ == "__main__":
    main()