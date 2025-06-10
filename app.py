import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('wordnet')

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-letters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def ats_check(resume_text, required_skills, min_years_exp, experience_phrases):
    score = 0
    resume_lower = resume_text.lower()
    
    # Skills found
    skills_found = [skill.strip() for skill in required_skills if skill.strip().lower() in resume_lower]
    skill_score = len(skills_found) / len(required_skills) if required_skills else 0
    
    # Experience extraction
    experience_years_found = 0
    matches = re.findall(r'(\d+)\s*\+?\s*years', resume_lower)
    if matches:
        experience_years_found = max(int(year) for year in matches)
    experience_score = 1.0 if experience_years_found >= min_years_exp else 0.0
    
    # Weighted ATS score
    score = 0.7 * skill_score + 0.3 * experience_score
    return score, skills_found, experience_years_found

@st.cache_data
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['Resume_str'])
    df['Resume_Clean'] = df['Resume_str'].apply(preprocess_text)
    return df

def rank_resumes(job_desc, resumes_clean):
    documents = resumes_clean.tolist() + [job_desc]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    scores = cosine_sim.flatten()
    return scores

st.title("📝 Advanced Resume Screening with ATS Checker & Suggestions")

st.markdown("""
Upload a CSV file containing resumes with a column named **`Resume_str`**.  
Enter or upload the job description, then see top matching resumes with ATS scoring and improvement suggestions.
""")

csv_file = st.file_uploader("Upload Resume Dataset CSV", type=['csv'])

if csv_file:
    data = load_dataset(csv_file)
    st.success(f"Loaded {len(data)} resumes.")

    job_desc_text = st.text_area("Enter Job Description Text", height=200)
    uploaded_jd = st.file_uploader("Or Upload Job Description Text File", type=['txt'])
    if uploaded_jd:
        job_desc_text = uploaded_jd.read().decode('utf-8')
        st.text_area("Job Description Text", value=job_desc_text, height=200)

    if job_desc_text.strip() != "":
        job_desc_clean = preprocess_text(job_desc_text)

        scores = rank_resumes(job_desc_clean, data['Resume_Clean'])
        data['Similarity_Score'] = scores

        st.sidebar.header("ATS Configuration")
        required_skills_str = st.sidebar.text_input("Required Skills (comma separated)", 
                                                   "python, machine learning, nlp, sql")
        required_skills = [skill.strip() for skill in required_skills_str.lower().split(',') if skill.strip()]
        min_experience = st.sidebar.number_input("Minimum Experience (years)", min_value=0, max_value=50, value=3)

        exp_phrases = ['years', 'year experience', 'yrs', 'yr experience']

        ats_scores = []
        skills_found_list = []
        exp_years_list = []
        for resume in data['Resume_str']:
            ats_score, skills_found, exp_years = ats_check(resume, required_skills, min_experience, exp_phrases)
            ats_scores.append(ats_score)
            skills_found_list.append(skills_found)
            exp_years_list.append(exp_years)

        data['ATS_Score'] = ats_scores
        data['Skills_Found'] = skills_found_list
        data['Experience_Years_Found'] = exp_years_list

        data['Final_Score'] = 0.6 * data['Similarity_Score'] + 0.4 * data['ATS_Score']

        data = data.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)

        SHORTLIST_THRESHOLD = 0.5  # Threshold below which suggestions show

        st.subheader("Top Matching Resumes")

        for i, row in data.head(10).iterrows():
            st.markdown(f"### Candidate #{i+1} (ID: {row['ID']})")
            if 'Category' in row:
                st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"**Similarity Score:** {row['Similarity_Score']:.2f}")
            st.markdown(f"**ATS Score:** {row['ATS_Score']:.2f}")
            st.markdown(f"**Final Score:** {row['Final_Score']:.2f}")
            st.markdown(f"**Skills Matched:** {', '.join(row['Skills_Found']) if row['Skills_Found'] else 'None'}")
            st.markdown(f"**Experience Found:** {row['Experience_Years_Found']} years")

            with st.expander("Show Resume Text"):
                st.text(row['Resume_str'][:1000])  # Show first 1000 chars

            # Suggestion system for resumes NOT shortlisted
            if row['Final_Score'] < SHORTLIST_THRESHOLD:
                st.warning("### Suggestions to Improve Your Resume:")

                # Missing skills
                skills_found_set = set([s.lower() for s in row['Skills_Found']])
                missing_skills = set(required_skills) - skills_found_set
                if missing_skills:
                    st.write("- Add or highlight these key skills:", ", ".join(missing_skills))

                # Experience suggestion
                if row['Experience_Years_Found'] < min_experience:
                    st.write(f"- You appear to have {row['Experience_Years_Found']} years of experience, which is below the minimum required {min_experience} years. Consider highlighting relevant experience.")

                # Suggest keywords from job description missing in resume
                job_desc_tokens = set(job_desc_clean.split())
                resume_tokens = set(row['Resume_Clean'].split())
                missing_keywords = job_desc_tokens - resume_tokens
                common_suggestions = [kw for kw in missing_keywords if len(kw) > 4]  # Filter out short words

                if common_suggestions:
                    st.write("- Consider including relevant keywords such as:", ", ".join(list(common_suggestions)[:10]))

else:
    st.info("Please upload your Resume Dataset CSV file.")
