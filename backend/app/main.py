from fastapi import FastAPI, UploadFile, File, HTTPException
from parser import parse_resume, parse_job_description,match_resume_to_job
from parser import format_match_results,format_missing_results

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Resume Matcher API is running"}

@app.post("/analyze")
async def analyze_resume(resume: UploadFile = File(...), job_desc: UploadFile = File(...)):
  try:
    resume_text = await resume.read()
    job_text = await job_desc.read()

    parsed_resume = parse_resume(resume_text)
    parsed_job = parse_job_description(job_text)

    match_score, missing_skills, missing_experience, missing_education, matched_skills, resume_data, job_data = match_resume_to_job(parsed_resume, parsed_job)

    formatted_match = format_match_results(resume_data["skills"], title="Matched Skills")
    formatted_missing = format_missing_results(job_data["required_skills"], title="Missing Skills")

    return {
        "match_score": match_score,
        "matched_skills_detailed": formatted_match,
        "missing_skills_detailed": formatted_missing,
        "missing_experience": missing_experience,
        "missing_education": missing_education,
        "raw_matched_skills": matched_skills,
        "raw_missing_skills": missing_skills
        }
  except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

