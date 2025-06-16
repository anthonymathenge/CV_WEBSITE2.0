# resume_parser/gpt_extraction.py
import re
import openai
import os
import json

# Load OpenAI key from env
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"  # Or "gpt-4-turbo" if you don't have access to GPT-4o

# Resume extractor
def extract_resume_info(resume_text):
    system_prompt = """
    You are a resume parser. Extract the following information from the resume text provided.

    Output format (strictly valid JSON):
    {
      "name": "",
      "email": "",
      "phone": "",
      "skills": [],
      "education": [],
      "experience": []
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
    You are a job description parser. Extract the following information from the job description text provided.

    Output format (strictly valid JSON):
    {
      "required_skills": [],
      "required_experience": [],
      "required_education": []
    }

    Only include skills, experience, and education explicitly mentioned as required in the job description.
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