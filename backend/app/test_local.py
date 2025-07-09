from parser import parse_resume, parse_job_description, match_resume_to_job
from parser import format_match_results,format_missing_results, format_missing_experience, format_missing_education

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

if __name__ == "__main__":
    # Use your own test resume and job description paths
    resume_text = read_file("resume_test.pdf")
    job_desc_text = read_file("job_desc.docx")

    parsed_resume = parse_resume(resume_text)
    parsed_job = parse_job_description(job_desc_text)

    match_score, missing_skills, missing_experience, missing_education, matched_skills, resume_data, job_data = match_resume_to_job(parsed_resume, parsed_job)
    formatted_match = format_match_results(resume_data["skills"], title="Matched Skills")
    formatted_missing = format_missing_results(job_data["required_skills"], title="Missing Skills")
    format_experience = format_missing_experience(missing_experience)
    format_education = format_missing_education(missing_education)

    print(f"Match Score: {match_score}")
    print(f"{formatted_match}")
    print(f"{formatted_missing}")
    print(f"{format_experience}")
    print(f"{format_education}")
    print(f"Matched Skills: {matched_skills}")
    print(f"raw_missing_skills: {missing_skills}")

