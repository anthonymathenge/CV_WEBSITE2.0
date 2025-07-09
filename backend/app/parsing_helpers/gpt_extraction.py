
import re
import openai
import os
import json
from openai import OpenAI

#client = OpenAI(api_key = )
MODEL = "gpt-4o"  
# Resume extractor
def extract_resume_info(resume_text):
    system_prompt = """
    You are an advanced resume parser designed for talent matching. Extract detailed information from resumes by reading and reasoning like a recruiter.

    When extracting, apply the following logic:

    - Extract skills only if they are clearly and explicitly mentioned in the resume.
    - Include technical skills such as programming languages, tools, frameworks, cloud platforms, libraries as well as soft skills such as communication, leadership, problem-solving.
    - If certifications are present, include them under skills.
    - If the person has exposure to specialized tools like CAD, simulation software, or laboratory instruments, include that in technical skills.
    - Include both current and previous job experience, including job title, employer name, and duration if available.

    Output strictly valid JSON in this enhanced format:

    {
    "name": "",
    "email": "",
    "phone": "",
    "skills": [
        {
        "category": "Technical & Software Skills",
        "items": ["Java", "Unix", "SQL"],
        },
        {
        "category": "Communication & Collaboration",
        "items": ["Mentoring", "Team Leadership"],
        }
    ],
    "education": ["BSc in Computer Science"],
    "experience": ["Technical Support Intern at ABC Ltd (2022-2023)"]
    }

    """

    user_prompt = f"Resume Text:\n\"\"\"\n{resume_text}\n\"\"\""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    output_text = response.choices[0].message.content.strip()
    return safe_json_load(output_text)


# Job Description extractor
def extract_job_description_info(job_text):
    system_prompt = """
    You are an advanced job description parser designed for resume matching. Extract required qualifications using reasoning and inference.

    When extracting, apply the following logic:

    - Extract both explicitly stated and implied required skills.
    - Include technical skills (programming languages, tools, frameworks, cloud platforms, libraries) as well as soft skills (communication, leadership, problem-solving).
    - Include certifications or licenses if mentioned.
    - Infer missing details where possible (e.g. "object-oriented programming" → infer languages like Java, C++, Python).
    - Normalize phrasing for consistent matching.

    Output strictly valid JSON in the following format:

    {
    "required_skills": [
        {
        "category": "Customer Support Tools",
        "items": ["ticketing systems", "live chat"],
        },
        {
        "category": "Testing and QA",
        "items": ["bug reporting", "feature testing"],
        }
    ],
    "required_experience": ["1+ year in technical support roles"],
    "required_education": ["Degree in Computer Science or related field"]
    }
    """

    user_prompt = f"Job Description Text:\n\"\"\"\n{job_text}\n\"\"\""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    output_text = response.choices[0].message.content.strip()
    return safe_json_load(output_text)
 
def safe_json_load(gpt_response):
    # Remove Markdown-style code fences
    gpt_response = re.sub(r"```(json)?", "", gpt_response, flags=re.IGNORECASE).strip()
    return json.loads(gpt_response)