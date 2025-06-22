
import re
import openai
import os
import json

#openai.api_key = 
MODEL = "gpt-4o"  
# Resume extractor
def extract_resume_info(resume_text):
    system_prompt = """
    You are an advanced resume parser designed for talent matching. Extract detailed information from resumes by reading and reasoning like a recruiter.

    When extracting, apply the following logic:

    - Extract skills even if they are implied or indirectly stated. Use your reasoning to infer both technical and soft skills.
    - Include programming languages, frameworks, libraries, tools, platforms, cloud providers, as well as soft skills such as teamwork, problem-solving, and leadership.
    - If certifications are present, include them under skills.
    - Normalize and unify degree names: for example, "Computer Science", "Software Engineering", "Information Technology", and "STEM fields" should all map under "degree in computer science".
    - If the person has exposure to specialized tools like CAD, simulation software, or laboratory instruments, infer related technical skills.
    - Include both current and previous job experience, including job title, employer name, and duration if available.

    Output strictly valid JSON in the following format:

    {
    "name": "",
    "email": "",
    "phone": "",
    "skills": ["Skill A", "Skill B", ...],
    "education": ["education A", "education B", ...],
    "experience": ["experience A", "experience B", ...]
    }
    """

    user_prompt = f"Resume Text:\n\"\"\"\n{resume_text}\n\"\"\""

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    output_text = response.choices[0].message.content.strip()
    extracted_data = safe_json_load(output_text)
    return extracted_data


# Job Description extractor
def extract_job_description_info(job_text):
    system_prompt = """
    You are an advanced job description parser designed for resume matching. Extract required qualifications using reasoning and inference.

    When extracting, apply the following logic:

    - Extract both explicitly stated and implied required skills.
    - Include technical skills (programming languages, tools, frameworks, cloud platforms, libraries) as well as soft skills (communication, leadership, problem-solving).
    - Include certifications or licenses if mentioned.
    - Normalize equivalent degree fields: for example, "Computer Science", "Software Engineering", "Information Technology", or "related STEM fields" should all map to "degree in computer science".
    - Infer missing details where possible (e.g. "object-oriented programming" → infer languages like Java, C++, Python).
    - Normalize phrasing for consistent matching.

    Output strictly valid JSON in the following format:

    {
    "required_skills": [],
    "required_experience": [],
    "required_education": []
    }
    """

    user_prompt = f"Job Description Text:\n\"\"\"\n{job_text}\n\"\"\""

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    output_text = response.choices[0].message.content.strip()
    extracted_data = safe_json_load(output_text)
    return extracted_data

def safe_json_load(gpt_response):
    # Remove Markdown-style code fences
    gpt_response = re.sub(r"```(json)?", "", gpt_response, flags=re.IGNORECASE).strip()
    return json.loads(gpt_response)