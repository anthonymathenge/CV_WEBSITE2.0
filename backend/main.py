from fastapi import FastAPI, UploadFile, File, HTTPException
from parser import parse_resume, parse_job_description,match_resume_to_job

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

    match_score, missing_skills, missing_experience, missing_education= match_resume_to_job(parsed_resume, parsed_job)

    return {
        "match_score": match_score,
        "missing_skills": missing_skills,
        "missing_experience": missing_experience,
        "missing_education": missing_education
    }
  except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

