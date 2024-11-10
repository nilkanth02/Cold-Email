import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name ='llama-3.1-70b-versatile')

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res=chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("context too big. Unable to parse job.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB Description:
             {job_description}

            ### Instruction:
            Write a professional cold email from the perspective of a final-year engineering student, 
            Nilkanth Ahire, applying for a role at a company. Highlight their background in AI and Data
            Science, mention relevant projects (like a Spam Classifier and Student Management System), 
            and express eagerness to contribute to the company. The email should convey enthusiasm,
            technical expertise, and a willingness to learn. Keep it concise yet impactful, 
            making it clear that Nilkanth is ready to bring value to the company through his skills 
            in Python, machine learning, and data analytics.
            Do not provide the preamble.
            ### Email(Nopreamble):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description":str(job), "link_list":links})
        return res.content

    if __name__ == "__main__":
        print(os.getenv("GROQ_API_KEY"))