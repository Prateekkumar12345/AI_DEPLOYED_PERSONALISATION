# """
# AI Resume Analyzer with Comprehensive Features & Three-Layer Validation
# This version combines robust resume analysis with extensive feature tracking
# """

# from fastapi import FastAPI, HTTPException, File, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# import os
# import json
# import logging
# from dotenv import load_dotenv
# from typing import Optional, Dict, Any, List
# import asyncio
# import io
# import concurrent.futures
# import re
# from datetime import datetime
# import hashlib
# import uuid

# # Updated LangChain imports
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import HumanMessage, SystemMessage
# from pydantic import BaseModel, Field, validator

# # Import shared database
# from shared_database import SharedDatabase, EXTERNAL_DB_API_URL

# # Load environment variables
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI
# app = FastAPI(title="AI Resume Analyzer API with Comprehensive Features")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize OpenAI client
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     logger.warning("OPENAI_API_KEY not found in environment variables")

# # Initialize shared database
# shared_db = SharedDatabase()

# # ===== CACHE FOR CONSISTENT RESULTS =====
# analysis_cache = {}

# def get_content_hash(resume_text: str, target_role: str) -> str:
#     """Generate consistent hash for caching"""
#     content = f"{resume_text[:1000]}_{target_role}"
#     return hashlib.md5(content.encode()).hexdigest()

# # ===== REQUEST/RESPONSE MODELS =====

# class AnalyzeResumeRequest(BaseModel):
#     """Request model for resume analysis"""
#     username: str = Field(..., description="Username for whom the analysis is being done")
#     target_role: Optional[str] = Field(None, description="Target job position/role")
#     search_jobs: bool = Field(True, description="Whether to search for relevant jobs")
#     location: str = Field("India", description="Location for job search")

#     class Config:
#         schema_extra = {
#             "example": {
#                 "username": "john_doe",
#                 "target_role": "Senior Software Engineer",
#                 "search_jobs": True,
#                 "location": "India"
#             }
#         }

# # ===== PYDANTIC MODELS =====

# class ProfessionalProfile(BaseModel):
#     experience_level: str = Field(description="Years of experience and seniority level")
#     technical_skills_count: int = Field(description="Number of technical skills identified")
#     project_portfolio_size: str = Field(description="Size and quality of project portfolio")
#     achievement_metrics: str = Field(description="Quality of quantified achievements")
#     technical_sophistication: str = Field(description="Level of technical expertise")

# class ContactPresentation(BaseModel):
#     email_address: str = Field(description="Email presence and quality")
#     phone_number: str = Field(description="Phone number presence")
#     education: str = Field(description="Education background quality")
#     resume_length: str = Field(description="Resume length assessment")
#     action_verbs: str = Field(description="Use of strong action verbs")

# class OverallAssessment(BaseModel):
#     score_percentage: int = Field(description="Overall score percentage")
#     level: str = Field(description="Assessment level")
#     description: str = Field(description="Score description")
#     recommendation: str = Field(description="Overall recommendation")

# class ExecutiveSummary(BaseModel):
#     professional_profile: ProfessionalProfile
#     contact_presentation: ContactPresentation
#     overall_assessment: OverallAssessment

# class ScoringDetail(BaseModel):
#     score: int = Field(description="Score out of max points")
#     max_score: int = Field(description="Maximum possible score")
#     percentage: float = Field(description="Percentage score")
#     details: List[str] = Field(description="Detailed breakdown of scoring")

# class StrengthAnalysis(BaseModel):
#     strength: str = Field(description="Main strength identified")
#     why_its_strong: str = Field(description="Explanation of why it's a strength")
#     ats_benefit: str = Field(description="How it helps with ATS systems")
#     competitive_advantage: str = Field(description="Competitive advantage provided")
#     evidence: str = Field(description="Supporting evidence from resume")

# class WeaknessAnalysis(BaseModel):
#     weakness: str = Field(description="Main weakness identified")
#     why_problematic: str = Field(description="Why this is problematic")
#     ats_impact: str = Field(description="Impact on ATS systems")
#     how_it_hurts: str = Field(description="How it hurts candidacy")
#     fix_priority: str = Field(description="Priority level: CRITICAL/HIGH/MEDIUM")
#     specific_fix: str = Field(description="Specific steps to fix")
#     timeline: str = Field(description="Timeline for implementation")

# class ImprovementPlan(BaseModel):
#     critical: List[str] = Field(default_factory=list, description="Critical improvements")
#     high: List[str] = Field(default_factory=list, description="High priority improvements")
#     medium: List[str] = Field(default_factory=list, description="Medium priority improvements")

# class JobMarketAnalysis(BaseModel):
#     role_compatibility: str = Field(description="Compatibility with target role")
#     market_positioning: str = Field(description="Position in job market")
#     career_advancement: str = Field(description="Career advancement opportunities")
#     skill_development: str = Field(description="Skill development recommendations")

# class AIInsights(BaseModel):
#     overall_score: int = Field(description="Overall AI-determined score")
#     recommendation_level: str = Field(description="Recommendation level")
#     key_strengths_count: int = Field(description="Number of key strengths")
#     improvement_areas_count: int = Field(description="Number of improvement areas")

# class ResumeAnalysis(BaseModel):
#     """Main analysis model matching standard JSON structure"""
#     professional_profile: ProfessionalProfile
#     contact_presentation: ContactPresentation
#     detailed_scoring: Dict[str, ScoringDetail]
#     strengths_analysis: List[StrengthAnalysis] = Field(min_items=5)
#     weaknesses_analysis: List[WeaknessAnalysis] = Field(min_items=5)
#     improvement_plan: ImprovementPlan
#     job_market_analysis: JobMarketAnalysis
#     overall_score: int = Field(ge=0, le=100, description="Overall resume score out of 100")
#     recommendation_level: str = Field(description="Overall recommendation level")

# class JobListing(BaseModel):
#     company_name: str = Field(description="Name of the hiring company")
#     position: str = Field(description="Job position/title")
#     location: str = Field(description="Job location")
#     ctc: str = Field(description="Compensation/Salary range")
#     experience_required: str = Field(description="Required years of experience")
#     last_date_to_apply: str = Field(description="Application deadline")
#     about_job: str = Field(description="Brief description about the job")
#     job_description: str = Field(description="Detailed job description")
#     job_requirements: str = Field(description="Required skills and qualifications")
#     application_url: Optional[str] = Field(description="Link to apply")

# # ===== DOCUMENT CLASSIFICATION =====

# class DocumentClassificationResult(BaseModel):
#     """Result of initial document classification"""
#     label: str              # "resume" | "non_resume"
#     confidence: float       # 0.0 to 1.0
#     reason: str            # Explanation from classifier

# class DocumentClassifier:
#     """
#     Initial LLM-based document classifier.
#     Runs BEFORE the heuristic validator as a quick first pass.
#     """
    
#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm
    
#     async def classify(self, text: str, max_chars: int = 6000) -> DocumentClassificationResult:
#         """Quick classification using GPT to determine if document is a resume"""
        
#         system_prompt = """You are an expert HR document classifier.

# Task: Classify whether the given document is a Resume/CV or NOT.

# Guidelines:
# - Resume/CV includes: education, work experience, skills, certifications, projects, personal information
# - Resumes may vary in format (tables, bullet points, paragraphs, columns)
# - Job descriptions, technical documentation, invoices, letters, articles, forms are NOT resumes
# - Developer resumes will include technical skills like Docker, Kubernetes, architecture - this is NORMAL

# Output rules:
# - Respond ONLY with valid JSON
# - No extra text before or after JSON"""

#         user_prompt = f"""Classify the following document:

# {text[:max_chars]}

# Return JSON exactly in this format:
# {{
#   "label": "resume" or "non_resume",
#   "confidence": number between 0 and 1,
#   "reason": "short explanation"
# }}"""

#         try:
#             response = await self.llm.ainvoke(
#                 f"{system_prompt}\n\n{user_prompt}"
#             )
            
#             response_text = response.content if hasattr(response, 'content') else str(response)
            
#             # Extract JSON from response
#             json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
#             if json_match:
#                 parsed = json.loads(json_match.group())
#                 return DocumentClassificationResult(
#                     label=parsed.get("label", "non_resume"),
#                     confidence=float(parsed.get("confidence", 0.0)),
#                     reason=parsed.get("reason", "Classification completed")
#                 )
#             else:
#                 logger.warning(f"Could not parse classifier JSON: {response_text}")
#                 return DocumentClassificationResult(
#                     label="non_resume",
#                     confidence=0.5,
#                     reason="Could not parse classifier response"
#                 )
                
#         except Exception as e:
#             logger.error(f"Document classification error: {e}")
#             return DocumentClassificationResult(
#                 label="non_resume",
#                 confidence=0.0,
#                 reason=f"Classification failed: {e}"
#             )

# # ===== RESUME VALIDATION =====

# class ResumeValidationResult(BaseModel):
#     """Result of resume validation"""
#     is_resume: bool
#     confidence: str        # "high" | "medium" | "low"
#     method: str            # "heuristic" | "llm" | "heuristic+llm"
#     reason: str            # Human-readable explanation

# class ResumeValidator:
#     """
#     Two-layer resume validator with comprehensive keyword sets.
#     """
    
#     # Resume signals with weights
#     RESUME_SIGNALS: List[tuple] = [
#         # Identity / contact block (very strong resume signals)
#         ("linkedin.com",           3),
#         ("github.com",             3),
#         # Education statements
#         ("bachelor",               2),
#         ("master",                 2),
#         ("b.tech",                 2),
#         ("m.tech",                 2),
#         ("b.sc",                   2),
#         ("m.sc",                   2),
#         ("mba",                    2),
#         ("university",             2),
#         ("degree",                 2),
#         # Classic resume section headers
#         ("work experience",        2),
#         ("professional experience",2),
#         ("employment history",     2),
#         ("education",              2),
#         ("certifications",         2),
#         ("technical skills",       2),
#         ("skills",                 1),
#         ("objective",              1),
#         ("summary",                1),
#         ("achievements",           2),
#         ("projects",               1),
#         ("personal statement",     2),
#         # Developer-CV specific headers / phrases
#         ("full stack developer",   2),
#         ("software developer",     2),
#         ("software engineer",      2),
#         ("frontend developer",     2),
#         ("backend developer",      2),
#         ("freelancer",             2),
#         # Action phrases common in experience bullets
#         ("responsible for",        1),
#         ("managed",                1),
#         ("developed",              1),
#         ("led a team",             1),
#         ("experience in",          1),
#         ("proficient in",          1),
#         # Percentage / score (common in Indian CVs for marks)
#         ("percentage:",            2),
#     ]

#     # Non-resume signals with weights
#     NON_RESUME_SIGNALS: List[tuple] = [
#         # Multi-word phrases that only appear in doc/spec writing
#         ("technical documentation", 3),
#         ("system design",           2),
#         ("requirements document",   3),
#         ("data model",              2),
#         ("database schema",         2),
#         ("flow lifecycle",          3),
#         ("api endpoint",            2),
#         # Actual code syntax (with trailing space / brace to reduce false positives)
#         ("def ",                    2),
#         ("import ",                 1),
#         ("class {",                 3),
#         ("extends model",           3),
#         ("enum ",                   2),
#         # Project-management / agile docs
#         ("user story",              2),
#         ("sprint",                  2),
#         ("backlog",                 2),
#         ("readme",                  2),
#         ("changelog",               2),
#         # Academic / research
#         ("abstract",                2),
#         ("methodology",             2),
#         ("bibliography",            3),
#         ("hypothesis",              3),
#         ("literature review",       3),
#     ]

#     RESUME_NET_THRESHOLD     =  2   # net >= this → resume
#     NON_RESUME_NET_THRESHOLD = -4   # net <= this → not a resume

#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm

#     def _heuristic_check(self, text: str) -> ResumeValidationResult:
#         """Compute weighted resume and non-resume scores"""
#         lower_text = text.lower()

#         resume_score = 0
#         resume_hits: List[str] = []
#         for phrase, weight in self.RESUME_SIGNALS:
#             if phrase in lower_text:
#                 resume_score += weight
#                 resume_hits.append(phrase)

#         non_resume_score = 0
#         non_resume_hits: List[str] = []
#         for phrase, weight in self.NON_RESUME_SIGNALS:
#             if phrase in lower_text:
#                 non_resume_score += weight
#                 non_resume_hits.append(phrase)

#         net = resume_score - non_resume_score

#         logger.info(
#             f"Heuristic — resume_score: {resume_score} (hits: {resume_hits}), "
#             f"non_resume_score: {non_resume_score} (hits: {non_resume_hits}), "
#             f"net: {net}"
#         )

#         if net >= self.RESUME_NET_THRESHOLD:
#             return ResumeValidationResult(
#                 is_resume=True,
#                 confidence="high",
#                 method="heuristic",
#                 reason=(
#                     f"Document matched resume indicators with a weighted score of "
#                     f"{resume_score} vs {non_resume_score} for non-resume indicators "
#                     f"(net: +{net})."
#                 ),
#             )

#         if net <= self.NON_RESUME_NET_THRESHOLD:
#             return ResumeValidationResult(
#                 is_resume=False,
#                 confidence="high",
#                 method="heuristic",
#                 reason=(
#                     f"Document matched non-resume indicators with a weighted score of "
#                     f"{non_resume_score} vs {resume_score} for resume indicators "
#                     f"(net: {net})."
#                 ),
#             )

#         return ResumeValidationResult(
#             is_resume=False,
#             confidence="low",
#             method="heuristic",
#             reason=f"Ambiguous signal (net: {net}) — escalating to LLM classification.",
#         )

#     async def _llm_check(self, text: str) -> ResumeValidationResult:
#         """Send a lightweight classification prompt to the LLM"""
#         max_chars = 2000
#         if len(text) > max_chars:
#             half = max_chars // 2
#             snippet = text[:half] + "\n\n[... middle section omitted ...]\n\n" + text[-half:]
#         else:
#             snippet = text

#         prompt = (
#             "You are a document classifier. Read the following document excerpt and decide "
#             "whether it is a RESUME (also called a CV) or NOT a resume.\n\n"
#             "A resume/CV is a personal document that lists an individual's education, "
#             "work experience, skills, and qualifications for the purpose of job applications.\n\n"
#             "Respond with EXACTLY one of these two JSON objects and nothing else:\n"
#             '  {"is_resume": true, "reason": "<brief explanation>"}\n'
#             '  {"is_resume": false, "reason": "<brief explanation of what the document actually is>"}\n\n'
#             "Document excerpt:\n"
#             "---\n"
#             f"{snippet}\n"
#             "---\n"
#         )

#         try:
#             response = await self.llm.ainvoke(prompt)
#             response_text = response.content if hasattr(response, 'content') else str(response)
            
#             json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
#             if json_match:
#                 parsed = json.loads(json_match.group())
#                 is_resume = bool(parsed.get("is_resume", False))
#                 reason = parsed.get("reason", "LLM classification completed.")
#                 return ResumeValidationResult(
#                     is_resume=is_resume,
#                     confidence="high",
#                     method="llm",
#                     reason=reason,
#                 )
#             else:
#                 logger.warning(f"LLM validation: could not parse JSON from response: {response_text}")
#                 return ResumeValidationResult(
#                     is_resume=False,
#                     confidence="medium",
#                     method="llm",
#                     reason="LLM response could not be parsed reliably.",
#                 )
#         except Exception as e:
#             logger.error(f"LLM validation error: {e}")
#             return ResumeValidationResult(
#                 is_resume=False,
#                 confidence="low",
#                 method="llm",
#                 reason=f"LLM classification failed ({e}).",
#             )

#     async def validate(self, text: str) -> ResumeValidationResult:
#         """Run Layer 1. If ambiguous, run Layer 2."""
#         heuristic_result = self._heuristic_check(text)

#         if heuristic_result.confidence == "high":
#             logger.info(f"Validation decided by heuristic: is_resume={heuristic_result.is_resume}")
#             return heuristic_result

#         logger.info("Heuristic ambiguous — running LLM classification.")
#         llm_result = await self._llm_check(text)
#         llm_result.method = "heuristic+llm"
#         logger.info(f"Validation decided by LLM: is_resume={llm_result.is_resume}")
#         return llm_result

# # ===== PDF EXTRACTION =====

# class OptimizedPDFExtractor:
#     """Optimized PDF text extraction"""
    
#     @staticmethod
#     async def extract_text_from_pdf(uploaded_file) -> Optional[str]:
#         try:
#             uploaded_file.seek(0)
#             content = await uploaded_file.read()
            
#             def process_pdf(content_bytes):
#                 pdf_file = io.BytesIO(content_bytes)
#                 pdf_reader = PdfReader(pdf_file)
                
#                 extracted_text = ""
#                 for page_num, page in enumerate(pdf_reader.pages):
#                     try:
#                         page_text = page.extract_text()
#                         if page_text and page_text.strip():
#                             extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
#                     except Exception as page_error:
#                         logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
#                         continue
                
#                 return extracted_text.strip()
            
#             loop = asyncio.get_event_loop()
#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 extracted_text = await loop.run_in_executor(pool, process_pdf, content)
            
#             return extracted_text if extracted_text else None
            
#         except Exception as e:
#             logger.error(f"PDF extraction error: {str(e)}")
#             return None

# # ===== JOB SEARCH =====

# class JobSearchService:
#     """Service to search and parse job listings"""
    
#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm
    
#     async def search_jobs(self, target_role: str, location: str = "India") -> List[Dict[str, Any]]:
#         """Search for jobs and extract structured information"""
#         try:
#             job_extraction_prompt = f"""
#             Generate 5-10 realistic current job listings for the position: {target_role} in {location}.
            
#             For each job listing, provide EXACTLY these fields in JSON format:
#             {{
#                 "company_name": "Company name",
#                 "position": "Exact job title",
#                 "location": "City/region in {location}",
#                 "ctc": "Salary range with currency",
#                 "experience_required": "X-Y years",
#                 "last_date_to_apply": "YYYY-MM-DD format",
#                 "about_job": "2-3 sentence summary",
#                 "job_description": "Detailed responsibilities and duties",
#                 "job_requirements": "Required skills, qualifications, and education",
#                 "application_url": "https://company-careers.com/job-id"
#             }}
            
#             Return ONLY a valid JSON array with no additional text. Make the data realistic and relevant to the current job market in 2025.
#             """
            
#             response = await self.llm.ainvoke(job_extraction_prompt)
#             response_text = response.content if hasattr(response, 'content') else str(response)
            
#             # Parse the JSON response
#             try:
#                 json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
#                 if json_match:
#                     jobs_data = json.loads(json_match.group())
#                 else:
#                     jobs_data = json.loads(response_text)
                
#                 return jobs_data
#             except json.JSONDecodeError as e:
#                 logger.error(f"Failed to parse job listings JSON: {e}")
#                 return []
                
#         except Exception as e:
#             logger.error(f"Job search error: {str(e)}")
#             return []

# # ===== RESUME ANALYZER =====

# class HighPerformanceLangChainAnalyzer:
#     """High-performance AI analyzer with guaranteed standard JSON output"""
    
#     def __init__(self, openai_api_key: str):
#         self.llm = ChatOpenAI(
#             api_key=openai_api_key,
#             model_name="gpt-3.5-turbo-16k",
#             temperature=0.0,
#             max_tokens=4000,
#             request_timeout=30
#         )
        
#         self.output_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
#         self.document_classifier = DocumentClassifier(llm=self.llm)
#         self.resume_validator = ResumeValidator(llm=self.llm)
#         self.job_search = JobSearchService(self.llm)
#         self._setup_analysis_chain()
    
#     def _setup_analysis_chain(self):
#         """Setup the analysis chain using LCEL"""
        
#         analysis_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are an expert resume analyzer. Analyze the resume comprehensively for the target role.

# YOU MUST provide a complete JSON response with ALL of the following sections:

# 1. PROFESSIONAL PROFILE (experience_level, technical_skills_count, project_portfolio_size, achievement_metrics, technical_sophistication)
# 2. CONTACT PRESENTATION (email_address, phone_number, education, resume_length, action_verbs)
# 3. DETAILED SCORING with these exact sections (use snake_case keys):
#    - "contact_information" (score, max_score, percentage, details)
#    - "technical_skills" (score, max_score, percentage, details)
#    - "experience_quality" (score, max_score, percentage, details)
#    - "quantified_achievements" (score, max_score, percentage, details)
#    - "content_optimization" (score, max_score, percentage, details)
# 4. STRENGTHS ANALYSIS - Provide at least 5 strengths
# 5. WEAKNESSES ANALYSIS - Provide at least 5 weaknesses
# 6. IMPROVEMENT PLAN (critical, high, medium lists)
# 7. JOB MARKET ANALYSIS (role_compatibility, market_positioning, career_advancement, skill_development)
# 8. overall_score (0-100)
# 9. recommendation_level

# {format_instructions}

# CRITICAL: Return ONLY valid JSON matching the exact structure specified. No additional text."""),
#             ("human", "Target Role: {target_role}\n\nResume Content:\n{resume_text}")
#         ]).partial(format_instructions=self.output_parser.get_format_instructions())
        
#         self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()
    
#     def _get_standard_response_template(self, target_role: str, word_count: int) -> Dict[str, Any]:
#         """Returns the standard response structure"""
#         return {
#             "success": True,
#             "analysis_method": "AI-Powered LangChain Analysis with Three-Layer Validation",
#             "resume_metadata": {
#                 "word_count": word_count,
#                 "validation_message": "Comprehensive AI analysis completed",
#                 "target_role": target_role or "general position"
#             },
#             "executive_summary": {
#                 "professional_profile": {},
#                 "contact_presentation": {},
#                 "overall_assessment": {}
#             },
#             "detailed_scoring": {},
#             "strengths_analysis": [],
#             "weaknesses_analysis": [],
#             "improvement_plan": {
#                 "critical": [],
#                 "high": [],
#                 "medium": []
#             },
#             "job_market_analysis": {},
#             "ai_insights": {}
#         }
    
#     def _convert_to_snake_case(self, key: str) -> str:
#         """Convert title case to snake_case"""
#         mapping = {
#             "Contact Information": "contact_information",
#             "Technical Skills": "technical_skills",
#             "Experience Quality": "experience_quality",
#             "Quantified Achievements": "quantified_achievements",
#             "Content Optimization": "content_optimization"
#         }
#         return mapping.get(key, key.lower().replace(" ", "_"))
    
#     async def analyze_resume_with_jobs(
#         self, 
#         resume_text: str, 
#         username: str,
#         target_role: Optional[str] = None,
#         search_jobs: bool = True,
#         location: str = "India"
#     ) -> Dict[str, Any]:
#         """Analyze resume with guaranteed standard JSON format and optional job search"""
#         try:
#             role_context = target_role or "general position"
#             word_count = len(resume_text.split())
            
#             # Check cache first
#             cache_key = get_content_hash(resume_text, role_context)
#             if cache_key in analysis_cache:
#                 logger.info("Returning cached analysis result")
#                 return analysis_cache[cache_key]
            
#             # Initialize response with standard structure
#             response = self._get_standard_response_template(role_context, word_count)
            
#             # Run resume analysis and job search in parallel if needed
#             if search_jobs and target_role:
#                 analysis_task = self.analysis_chain.ainvoke({
#                     "resume_text": resume_text,
#                     "target_role": role_context
#                 })
#                 jobs_task = self.job_search.search_jobs(target_role, location)
                
#                 analysis_result, job_listings = await asyncio.gather(
#                     analysis_task,
#                     jobs_task,
#                     return_exceptions=True
#                 )
                
#                 if isinstance(analysis_result, Exception):
#                     raise analysis_result
#                 if isinstance(job_listings, Exception):
#                     logger.error(f"Job search failed: {job_listings}")
#                     job_listings = []
#             else:
#                 analysis_result = await self.analysis_chain.ainvoke({
#                     "resume_text": resume_text,
#                     "target_role": role_context
#                 })
#                 job_listings = []
            
#             # Parse and populate response
#             try:
#                 parsed_analysis = self.output_parser.parse(analysis_result)
#                 self._populate_response(response, parsed_analysis, word_count, role_context)
                
#             except Exception as parse_error:
#                 logger.warning(f"Structured parsing failed, using fallback: {parse_error}")
#                 self._populate_fallback_response(response, analysis_result, word_count, role_context)
            
#             # Add job listings if available
#             if job_listings:
#                 response["job_listings"] = {
#                     "total_jobs_found": len(job_listings),
#                     "search_query": f"{target_role} in {location}",
#                     "jobs": job_listings
#                 }
            
#             # Add username to response
#             response["username"] = username
            
#             # Cache the result
#             analysis_cache[cache_key] = response
#             logger.info(f"Cached analysis result for key: {cache_key}")
            
#             return response
                
#         except Exception as e:
#             logger.error(f"Analysis error: {str(e)}")
#             return self._generate_error_response(str(e), target_role, word_count, username)
    
#     def _populate_response(self, response: Dict, analysis: ResumeAnalysis, word_count: int, target_role: str):
#         """Populate response with parsed analysis data"""
        
#         response["executive_summary"] = {
#             "professional_profile": {
#                 "experience_level": analysis.professional_profile.experience_level,
#                 "technical_skills_count": analysis.professional_profile.technical_skills_count,
#                 "project_portfolio_size": analysis.professional_profile.project_portfolio_size,
#                 "achievement_metrics": analysis.professional_profile.achievement_metrics,
#                 "technical_sophistication": analysis.professional_profile.technical_sophistication
#             },
#             "contact_presentation": {
#                 "email_address": analysis.contact_presentation.email_address,
#                 "phone_number": analysis.contact_presentation.phone_number,
#                 "education": analysis.contact_presentation.education,
#                 "resume_length": analysis.contact_presentation.resume_length,
#                 "action_verbs": analysis.contact_presentation.action_verbs
#             },
#             "overall_assessment": {
#                 "score_percentage": analysis.overall_score,
#                 "level": analysis.recommendation_level,
#                 "description": f"AI-determined resume quality: {analysis.overall_score}%",
#                 "recommendation": analysis.recommendation_level
#             }
#         }
        
#         response["detailed_scoring"] = {}
#         for key, detail in analysis.detailed_scoring.items():
#             snake_case_key = self._convert_to_snake_case(key)
#             response["detailed_scoring"][snake_case_key] = {
#                 "score": detail.score,
#                 "max_score": detail.max_score,
#                 "percentage": detail.percentage,
#                 "details": detail.details
#             }
        
#         response["strengths_analysis"] = [
#             {
#                 "strength": s.strength,
#                 "why_its_strong": s.why_its_strong,
#                 "ats_benefit": s.ats_benefit,
#                 "competitive_advantage": s.competitive_advantage,
#                 "evidence": s.evidence
#             }
#             for s in analysis.strengths_analysis
#         ]
        
#         response["weaknesses_analysis"] = [
#             {
#                 "weakness": w.weakness,
#                 "why_problematic": w.why_problematic,
#                 "ats_impact": w.ats_impact,
#                 "how_it_hurts": w.how_it_hurts,
#                 "fix_priority": w.fix_priority,
#                 "specific_fix": w.specific_fix,
#                 "timeline": w.timeline
#             }
#             for w in analysis.weaknesses_analysis
#         ]
        
#         response["improvement_plan"] = {
#             "critical": analysis.improvement_plan.critical,
#             "high": analysis.improvement_plan.high,
#             "medium": analysis.improvement_plan.medium
#         }
        
#         response["job_market_analysis"] = {
#             "role_compatibility": analysis.job_market_analysis.role_compatibility,
#             "market_positioning": analysis.job_market_analysis.market_positioning,
#             "career_advancement": analysis.job_market_analysis.career_advancement,
#             "skill_development": analysis.job_market_analysis.skill_development
#         }
        
#         response["ai_insights"] = {
#             "overall_score": analysis.overall_score,
#             "recommendation_level": analysis.recommendation_level,
#             "key_strengths_count": len(analysis.strengths_analysis),
#             "improvement_areas_count": len(analysis.weaknesses_analysis)
#         }
    
#     def _populate_fallback_response(self, response: Dict, raw_result: str, word_count: int, target_role: str):
#         """Fallback method to populate response from raw LLM output"""
#         try:
#             json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
#             if json_match:
#                 parsed_data = json.loads(json_match.group())
                
#                 if "professional_profile" in parsed_data:
#                     response["executive_summary"]["professional_profile"] = parsed_data["professional_profile"]
#                 if "contact_presentation" in parsed_data:
#                     response["executive_summary"]["contact_presentation"] = parsed_data["contact_presentation"]
#                 if "overall_score" in parsed_data:
#                     response["executive_summary"]["overall_assessment"] = {
#                         "score_percentage": parsed_data.get("overall_score", 0),
#                         "level": parsed_data.get("recommendation_level", "Unknown"),
#                         "description": f"AI-determined resume quality: {parsed_data.get('overall_score', 0)}%",
#                         "recommendation": parsed_data.get("recommendation_level", "Unknown")
#                     }
                
#                 detailed_scoring = parsed_data.get("detailed_scoring", {})
#                 converted_scoring = {}
#                 for key, value in detailed_scoring.items():
#                     snake_case_key = self._convert_to_snake_case(key)
#                     converted_scoring[snake_case_key] = value
#                 response["detailed_scoring"] = converted_scoring
                
#                 response["strengths_analysis"] = parsed_data.get("strengths_analysis", [])
#                 response["weaknesses_analysis"] = parsed_data.get("weaknesses_analysis", [])
#                 response["improvement_plan"] = parsed_data.get("improvement_plan", {"critical": [], "high": [], "medium": []})
#                 response["job_market_analysis"] = parsed_data.get("job_market_analysis", {})
#                 response["ai_insights"] = {
#                     "overall_score": parsed_data.get("overall_score", 0),
#                     "recommendation_level": parsed_data.get("recommendation_level", "Unknown"),
#                     "key_strengths_count": len(parsed_data.get("strengths_analysis", [])),
#                     "improvement_areas_count": len(parsed_data.get("weaknesses_analysis", []))
#                 }
                
#         except Exception as e:
#             logger.error(f"Fallback parsing error: {e}")
    
#     def _generate_error_response(self, error_message: str, target_role: str = None, word_count: int = 0, username: str = None) -> Dict[str, Any]:
#         """Generate error response maintaining standard structure"""
#         response = self._get_standard_response_template(target_role or "unknown", word_count)
#         response["success"] = False
#         response["error"] = f"AI analysis failed: {error_message}"
#         response["resume_metadata"]["validation_message"] = "Analysis encountered an error"
#         if username:
#             response["username"] = username
#         return response

# # ===== INITIALIZE COMPONENTS =====

# pdf_extractor = OptimizedPDFExtractor()
# high_perf_analyzer = None

# if openai_api_key:
#     try:
#         high_perf_analyzer = HighPerformanceLangChainAnalyzer(openai_api_key)
#         logger.info("High-performance analyzer initialized successfully")
#     except Exception as init_error:
#         logger.error(f"Failed to initialize analyzer: {init_error}")

# # ===== ENDPOINTS =====

# @app.post("/analyze-resume")
# async def analyze_resume(
#     file: UploadFile = File(..., description="Resume PDF file"),
#     username: str = Form(..., description="Username for whom the analysis is being done"),
#     target_role: str = Form(None, description="Target job position/role"),
#     search_jobs: bool = Form(True, description="Whether to search for relevant jobs"),
#     location: str = Form("India", description="Location for job search")
# ):
#     """
#     Comprehensive resume analysis with guaranteed standard JSON output and job search integration.
#     Includes a three-layer validation gate that rejects non-resume documents before analysis runs.
#     """
#     start_time = asyncio.get_event_loop().time()
    
#     try:
#         if not high_perf_analyzer:
#             raise HTTPException(status_code=500, detail="AI analyzer not initialized.")
        
#         if not file.content_type or "pdf" not in file.content_type.lower():
#             raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
#         # Extract PDF text
#         resume_text = await pdf_extractor.extract_text_from_pdf(file)
        
#         if not resume_text:
#             raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
#         if len(resume_text.strip()) < 100:
#             raise HTTPException(status_code=400, detail="Resume content too short.")

#         # THREE-LAYER VALIDATION PIPELINE
#         # Layer 1: Document Classification
#         logger.info("Running Layer 1: Document classification")
#         classification_result = await high_perf_analyzer.document_classifier.classify(resume_text)
        
#         if classification_result.label == "non_resume" and classification_result.confidence >= 0.7:
#             raise HTTPException(
#                 status_code=400,
#                 detail={
#                     "error": "not_a_resume",
#                     "message": "The uploaded document does not appear to be a resume/CV.",
#                     "validation": {
#                         "is_resume": False,
#                         "confidence": "high",
#                         "method": "llm_classifier",
#                         "reason": classification_result.reason,
#                         "classifier_confidence": classification_result.confidence,
#                     },
#                 },
#             )
        
#         logger.info(
#             f"Layer 1 result: {classification_result.label} "
#             f"(confidence: {classification_result.confidence:.2f})"
#         )
        
#         # Layer 2 & 3: Heuristic + LLM Validation
#         logger.info("Running Layer 2/3: Heuristic + LLM validation")
#         validation_result = await high_perf_analyzer.resume_validator.validate(resume_text)

#         if not validation_result.is_resume:
#             raise HTTPException(
#                 status_code=400,
#                 detail={
#                     "error": "not_a_resume",
#                     "message": "The uploaded document does not appear to be a resume/CV.",
#                     "validation": {
#                         "is_resume": validation_result.is_resume,
#                         "confidence": validation_result.confidence,
#                         "method": validation_result.method,
#                         "reason": validation_result.reason,
#                         "classifier_label": classification_result.label,
#                         "classifier_confidence": classification_result.confidence,
#                     },
#                 },
#             )

#         logger.info(
#             f"All validation layers passed (final method={validation_result.method}, "
#             f"confidence={validation_result.confidence}). Proceeding to analysis."
#         )
        
#         # Perform analysis
#         analysis_result = await asyncio.wait_for(
#             high_perf_analyzer.analyze_resume_with_jobs(
#                 resume_text=resume_text,
#                 username=username,
#                 target_role=target_role,
#                 search_jobs=search_jobs and bool(target_role),
#                 location=location
#             ),
#             timeout=60.0
#         )

#         # Save to shared database
#         analysis_id = str(uuid.uuid4())
#         shared_db.save_resume_analysis(
#             username=username,
#             analysis_id=analysis_id,
#             analysis_data={
#                 "target_role": target_role or "general position",
#                 "overall_score": analysis_result.get("ai_insights", {}).get("overall_score", 0),
#                 "recommendation_level": analysis_result.get("ai_insights", {}).get("recommendation_level", "Unknown"),
#                 "analysis_result": analysis_result,
#                 "uploaded_at": datetime.now().isoformat(),
#                 "validation_method": validation_result.method,
#                 "validation_confidence": validation_result.confidence
#             }
#         )
        
#         analysis_result["analysis_id"] = analysis_id
#         analysis_result["saved_to_database"] = True
        
#         processing_time = asyncio.get_event_loop().time() - start_time
#         logger.info(f"Analysis completed in {processing_time:.2f}s for user: {username}")
        
#         return analysis_result
        
#     except asyncio.TimeoutError:
#         raise HTTPException(status_code=408, detail="Analysis timeout.")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Analysis endpoint error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# @app.get("/user/{username}/analyses")
# async def get_user_analyses(username: str):
#     """Get all analyses for a specific user"""
#     try:
#         analyses = shared_db.get_user_resume_analyses(username)
#         return {
#             "username": username, 
#             "total_analyses": len(analyses), 
#             "analyses": analyses
#         }
#     except Exception as e:
#         logger.error(f"Error fetching analyses for user {username}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analyses: {str(e)}")

# @app.get("/analysis/{username}/{analysis_id}")
# async def get_analysis(username: str, analysis_id: str):
#     """Get a specific analysis by ID for a user"""
#     try:
#         # Since shared_database doesn't have a direct method to get by ID,
#         # we need to fetch all and filter
#         analyses = shared_db.get_user_resume_analyses(username)
#         for analysis in analyses:
#             if analysis.get("analysis_id") == analysis_id:
#                 return analysis
        
#         raise HTTPException(status_code=404, detail="Analysis not found")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching analysis {analysis_id} for user {username}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

# @app.delete("/analysis/{username}/{analysis_id}")
# async def delete_analysis(username: str, analysis_id: str):
#     """Delete a specific analysis"""
#     try:
#         # Use shared_database's delete_interaction method
#         shared_db.delete_interaction(username, "resume_analyzer", analysis_id)
#         return {"message": f"Analysis {analysis_id} deleted successfully for user {username}"}
#     except Exception as e:
#         logger.error(f"Error deleting analysis {analysis_id} for user {username}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

# @app.get("/health")
# async def health_check():
#     """Health check with comprehensive features display"""
#     try:
#         # Get database stats
#         all_users = shared_db.get_all_users()
#         all_analyses = []
#         analyses_by_user = {}
        
#         for user in all_users:
#             analyses = shared_db.get_user_resume_analyses(user)
#             all_analyses.extend(analyses)
#             analyses_by_user[user] = len(analyses)
        
#         return {
#             "status": "healthy",
#             "timestamp": datetime.now().isoformat(),
#             "service": "AI Resume Analyzer with Comprehensive Features",
#             "version": "4.0.0",
#             "features": {
#                 "three_layer_validation": "✅",
#                 "llm_document_classifier": "✅",
#                 "heuristic_validator": "✅",
#                 "llm_validator": "✅",
#                 "resume_analysis": "✅",
#                 "job_search_integration": "✅",
#                 "ats_scoring": "✅",
#                 "strengths_analysis": "✅",
#                 "weaknesses_analysis": "✅",
#                 "improvement_plan": "✅",
#                 "job_market_analysis": "✅",
#                 "quantified_scoring": "✅",
#                 "detailed_breakdown": "✅",
#                 "caching_mechanism": "✅",
#                 "shared_database": "✅",
#                 "user_tracking": "✅",
#                 "per_user_analyses": "✅",
#                 "analysis_history": "✅",
#                 "pdf_extraction": "✅",
#                 "error_handling": "✅",
#                 "performance_optimization": "✅",
#                 "consistent_json_output": "✅",
#                 "snake_case_naming": "✅",
#                 "deterministic_output": "✅"
#             },
#             "validation_pipeline": {
#                 "layer1": "LLM Document Classifier - Quick pre-screening",
#                 "layer2": "Heuristic Validator - Fast keyword-based scoring",
#                 "layer3": "LLM Validator - Deep analysis for ambiguous cases"
#             },
#             "database": {
#                 "type": "external_api",
#                 "url": EXTERNAL_DB_API_URL,
#                 "total_users": len(all_users),
#                 "total_analyses": len(all_analyses),
#                 "analyses_by_user": analyses_by_user
#             },
#             "performance": {
#                 "caching_enabled": True,
#                 "cache_size": len(analysis_cache),
#                 "parallel_processing": True,
#                 "optimized_pdf_extraction": True
#             },
#             "openai_configured": bool(openai_api_key),
#             "analyzer_available": bool(high_perf_analyzer),
#             "langchain_version": "Latest (LCEL)",
#             "guarantees": [
#                 "✅ Non-resume documents rejected before analysis",
#                 "✅ Consistent JSON structure every time",
#                 "✅ All standard fields present",
#                 "✅ Snake case field naming in detailed_scoring",
#                 "✅ Frontend-compatible format",
#                 "✅ Optional job listings",
#                 "✅ Deterministic output for identical resumes",
#                 "✅ Per-user analysis tracking",
#                 "✅ Full analysis history available"
#             ]
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         return {
#             "status": "degraded",
#             "service": "AI Resume Analyzer",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }

# @app.get("/")
# async def root():
#     """Root endpoint with comprehensive feature listing"""
#     return {
#         "service": "AI Resume Analyzer with Comprehensive Features",
#         "version": "4.0.0",
#         "description": "AI resume analysis with three-layer validation and comprehensive feature set",
#         "features": {
#             "validation_pipeline": "✅ Three-layer validation (LLM classifier + heuristic + LLM validator)",
#             "resume_analysis": "✅ Complete resume analysis with ATS scoring",
#             "job_search": "✅ Integrated job search with realistic listings",
#             "scoring_system": "✅ Multi-category scoring with detailed breakdowns",
#             "strengths_analysis": "✅ Top 5 strengths with ATS benefits",
#             "weaknesses_analysis": "✅ Top 5 weaknesses with fix priorities",
#             "improvement_plan": "✅ Prioritized improvement recommendations",
#             "job_market_analysis": "✅ Role compatibility and market positioning",
#             "caching": "✅ Content-based caching for consistent results",
#             "database": "✅ Shared database integration",
#             "user_tracking": "✅ Per-user analysis storage and retrieval",
#             "analysis_history": "✅ Full analysis history for each user",
#             "pdf_extraction": "✅ Optimized PDF text extraction",
#             "error_handling": "✅ Comprehensive error handling",
#             "performance": "✅ Parallel processing and optimization"
#         },
#         "endpoints": {
#             "/analyze-resume": {
#                 "method": "POST",
#                 "description": "Comprehensive analysis with three-layer validation",
#                 "content_type": "multipart/form-data",
#                 "fields": {
#                     "file": "PDF file (required)",
#                     "username": "string (required) - User identifier",
#                     "target_role": "string (optional)",
#                     "search_jobs": "boolean (default: true)",
#                     "location": "string (default: India)"
#                 }
#             },
#             "/user/{username}/analyses": {
#                 "method": "GET",
#                 "description": "Get all analyses for a specific user"
#             },
#             "/analysis/{username}/{analysis_id}": {
#                 "method": "GET",
#                 "description": "Get a specific analysis by ID"
#             },
#             "/analysis/{username}/{analysis_id}": {
#                 "method": "DELETE",
#                 "description": "Delete a specific analysis"
#             },
#             "/health": {
#                 "method": "GET",
#                 "description": "Service health check with features"
#             },
#             "/docs": {
#                 "method": "GET",
#                 "description": "API documentation"
#             }
#         },
#         "validation_pipeline_details": [
#             "Layer 1: LLM Document Classifier - Quick pre-screening (catches invoices, forms, job descriptions)",
#             "Layer 2: Heuristic Validator - Fast keyword-based scoring with weighted signals",
#             "Layer 3: LLM Validator - Deep analysis only for ambiguous cases"
#         ],
#         "output_guarantees": [
#             "Consistent JSON structure",
#             "All fields always present",
#             "Snake case naming in detailed_scoring",
#             "Frontend-compatible format",
#             "Deterministic for identical inputs",
#             "Username always included in response",
#             "Analysis ID for retrieval"
#         ]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("=" * 70)
#     print("🚀 Starting AI Resume Analyzer with Comprehensive Features")
#     print("=" * 70)
#     print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
#     print(f"🔑 OpenAI: {'✅ Configured' if openai_api_key else '❌ Not configured'}")
#     print(f"🔧 Analyzer: {'✅ Ready' if high_perf_analyzer else '❌ Not available'}")
#     print(f"🎯 Version: 4.0.0")
#     print(f"💬 Features:")
#     print(f"   • Three-Layer Validation: ✅")
#     print(f"   • LLM Document Classifier: ✅")
#     print(f"   • Heuristic Validator: ✅")
#     print(f"   • LLM Validator: ✅")
#     print(f"   • Resume Analysis: ✅")
#     print(f"   • Job Search Integration: ✅")
#     print(f"   • ATS Scoring: ✅")
#     print(f"   • Strengths Analysis: ✅")
#     print(f"   • Weaknesses Analysis: ✅")
#     print(f"   • Improvement Plan: ✅")
#     print(f"   • Job Market Analysis: ✅")
#     print(f"   • Quantified Scoring: ✅")
#     print(f"   • Detailed Breakdown: ✅")
#     print(f"   • Caching Mechanism: ✅")
#     print(f"   • Shared Database: ✅")
#     print(f"   • User Tracking: ✅")
#     print(f"   • Per-User Analyses: ✅")
#     print(f"   • Analysis History: ✅")
#     print(f"   • PDF Extraction: ✅")
#     print(f"   • Error Handling: ✅")
#     print(f"   • Performance Optimization: ✅")
#     print(f"   • Consistent JSON Output: ✅")
#     print(f"   • Snake Case Naming: ✅")
#     print(f"   • Deterministic Output: ✅")
#     print(f"🔗 API: http://localhost:8000")
#     print(f"📚 Docs: http://localhost:8000/docs")
#     print("=" * 70)
    
#     uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")



# """
# AI Resume Analyzer with Comprehensive Features & Three-Layer Validation
# This version combines robust resume analysis with extensive feature tracking
# UPDATED: Role-specific analysis with honest scoring and mismatch detection
# """

# from fastapi import FastAPI, HTTPException, File, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# from PyPDF2 import PdfReader
# import os
# import json
# import logging
# from dotenv import load_dotenv
# from typing import Optional, Dict, Any, List
# import asyncio
# import io
# import concurrent.futures
# import re
# from datetime import datetime
# import hashlib
# import uuid

# # Updated LangChain imports
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import HumanMessage, SystemMessage
# from pydantic import BaseModel, Field, validator

# # Import shared database
# from shared_database import SharedDatabase, EXTERNAL_DB_API_URL

# # Load environment variables
# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI
# app = FastAPI(title="AI Resume Analyzer API with Comprehensive Features")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize OpenAI client
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     logger.warning("OPENAI_API_KEY not found in environment variables")

# # Initialize shared database
# shared_db = SharedDatabase()

# # ===== CACHE FOR CONSISTENT RESULTS =====
# analysis_cache = {}

# def get_content_hash(resume_text: str, target_role: str) -> str:
#     """Generate consistent hash for caching"""
#     content = f"{resume_text[:1000]}_{target_role}"
#     return hashlib.md5(content.encode()).hexdigest()

# # ===== REQUEST/RESPONSE MODELS =====

# class AnalyzeResumeRequest(BaseModel):
#     """Request model for resume analysis"""
#     username: str = Field(..., description="Username for whom the analysis is being done")
#     target_role: Optional[str] = Field(None, description="Target job position/role")
#     search_jobs: bool = Field(True, description="Whether to search for relevant jobs")
#     location: str = Field("India", description="Location for job search")

#     class Config:
#         schema_extra = {
#             "example": {
#                 "username": "john_doe",
#                 "target_role": "Senior Software Engineer",
#                 "search_jobs": True,
#                 "location": "India"
#             }
#         }

# # ===== PYDANTIC MODELS =====

# class ProfessionalProfile(BaseModel):
#     experience_level: str = Field(description="Years of experience and seniority level")
#     technical_skills_count: int = Field(description="Number of technical skills identified")
#     project_portfolio_size: str = Field(description="Size and quality of project portfolio")
#     achievement_metrics: str = Field(description="Quality of quantified achievements")
#     technical_sophistication: str = Field(description="Level of technical expertise")

# class ContactPresentation(BaseModel):
#     email_address: str = Field(description="Email presence and quality")
#     phone_number: str = Field(description="Phone number presence")
#     education: str = Field(description="Education background quality")
#     resume_length: str = Field(description="Resume length assessment")
#     action_verbs: str = Field(description="Use of strong action verbs")

# class OverallAssessment(BaseModel):
#     score_percentage: int = Field(description="Overall score percentage")
#     level: str = Field(description="Assessment level")
#     description: str = Field(description="Score description")
#     recommendation: str = Field(description="Overall recommendation")

# class ExecutiveSummary(BaseModel):
#     professional_profile: ProfessionalProfile
#     contact_presentation: ContactPresentation
#     overall_assessment: OverallAssessment

# class ScoringDetail(BaseModel):
#     score: int = Field(description="Score out of max points")
#     max_score: int = Field(description="Maximum possible score")
#     percentage: float = Field(description="Percentage score")
#     details: List[str] = Field(description="Detailed breakdown of scoring")

# class StrengthAnalysis(BaseModel):
#     strength: str = Field(description="Main strength identified")
#     why_its_strong: str = Field(description="Explanation of why it's a strength")
#     ats_benefit: str = Field(description="How it helps with ATS systems")
#     competitive_advantage: str = Field(description="Competitive advantage provided")
#     evidence: str = Field(description="Supporting evidence from resume")

# class WeaknessAnalysis(BaseModel):
#     weakness: str = Field(description="Main weakness identified")
#     why_problematic: str = Field(description="Why this is problematic")
#     ats_impact: str = Field(description="Impact on ATS systems")
#     how_it_hurts: str = Field(description="How it hurts candidacy")
#     fix_priority: str = Field(description="Priority level: CRITICAL/HIGH/MEDIUM")
#     specific_fix: str = Field(description="Specific steps to fix")
#     timeline: str = Field(description="Timeline for implementation")

# class ImprovementPlan(BaseModel):
#     critical: List[str] = Field(default_factory=list, description="Critical improvements")
#     high: List[str] = Field(default_factory=list, description="High priority improvements")
#     medium: List[str] = Field(default_factory=list, description="Medium priority improvements")

# class JobMarketAnalysis(BaseModel):
#     role_compatibility: str = Field(description="Compatibility with target role: Low / Moderate / High")
#     market_positioning: str = Field(description="Position in job market for this specific role")
#     career_advancement: str = Field(description="Career advancement opportunities specific to target role")
#     skill_development: str = Field(description="Skill development recommendations for target role")

# class AIInsights(BaseModel):
#     overall_score: int = Field(description="Overall AI-determined score")
#     recommendation_level: str = Field(description="Recommendation level")
#     key_strengths_count: int = Field(description="Number of key strengths")
#     improvement_areas_count: int = Field(description="Number of improvement areas")

# class ResumeAnalysis(BaseModel):
#     """Main analysis model matching standard JSON structure"""
#     professional_profile: ProfessionalProfile
#     contact_presentation: ContactPresentation
#     detailed_scoring: Dict[str, ScoringDetail]
#     strengths_analysis: List[StrengthAnalysis] = Field(min_items=5)
#     weaknesses_analysis: List[WeaknessAnalysis] = Field(min_items=5)
#     improvement_plan: ImprovementPlan
#     job_market_analysis: JobMarketAnalysis
#     overall_score: int = Field(ge=0, le=100, description="Overall resume score out of 100")
#     recommendation_level: str = Field(description="Overall recommendation level")

# class JobListing(BaseModel):
#     company_name: str = Field(description="Name of the hiring company")
#     position: str = Field(description="Job position/title")
#     location: str = Field(description="Job location")
#     ctc: str = Field(description="Compensation/Salary range")
#     experience_required: str = Field(description="Required years of experience")
#     last_date_to_apply: str = Field(description="Application deadline")
#     about_job: str = Field(description="Brief description about the job")
#     job_description: str = Field(description="Detailed job description")
#     job_requirements: str = Field(description="Required skills and qualifications")
#     application_url: Optional[str] = Field(description="Link to apply")

# # ===== DOCUMENT CLASSIFICATION =====

# class DocumentClassificationResult(BaseModel):
#     """Result of initial document classification"""
#     label: str              # "resume" | "non_resume"
#     confidence: float       # 0.0 to 1.0
#     reason: str            # Explanation from classifier

# class DocumentClassifier:
#     """
#     Initial LLM-based document classifier.
#     Runs BEFORE the heuristic validator as a quick first pass.
#     """
    
#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm
    
#     async def classify(self, text: str, max_chars: int = 6000) -> DocumentClassificationResult:
#         """Quick classification using GPT to determine if document is a resume"""
        
#         system_prompt = """You are an expert HR document classifier.

# Task: Classify whether the given document is a Resume/CV or NOT.

# Guidelines:
# - Resume/CV includes: education, work experience, skills, certifications, projects, personal information
# - Resumes may vary in format (tables, bullet points, paragraphs, columns)
# - Job descriptions, technical documentation, invoices, letters, articles, forms are NOT resumes
# - Developer resumes will include technical skills like Docker, Kubernetes, architecture - this is NORMAL

# Output rules:
# - Respond ONLY with valid JSON
# - No extra text before or after JSON"""

#         user_prompt = f"""Classify the following document:

# {text[:max_chars]}

# Return JSON exactly in this format:
# {{
#   "label": "resume" or "non_resume",
#   "confidence": number between 0 and 1,
#   "reason": "short explanation"
# }}"""

#         try:
#             response = await self.llm.ainvoke(
#                 f"{system_prompt}\n\n{user_prompt}"
#             )
            
#             response_text = response.content if hasattr(response, 'content') else str(response)
            
#             # Extract JSON from response
#             json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
#             if json_match:
#                 parsed = json.loads(json_match.group())
#                 return DocumentClassificationResult(
#                     label=parsed.get("label", "non_resume"),
#                     confidence=float(parsed.get("confidence", 0.0)),
#                     reason=parsed.get("reason", "Classification completed")
#                 )
#             else:
#                 logger.warning(f"Could not parse classifier JSON: {response_text}")
#                 return DocumentClassificationResult(
#                     label="non_resume",
#                     confidence=0.5,
#                     reason="Could not parse classifier response"
#                 )
                
#         except Exception as e:
#             logger.error(f"Document classification error: {e}")
#             return DocumentClassificationResult(
#                 label="non_resume",
#                 confidence=0.0,
#                 reason=f"Classification failed: {e}"
#             )

# # ===== RESUME VALIDATION =====

# class ResumeValidationResult(BaseModel):
#     """Result of resume validation"""
#     is_resume: bool
#     confidence: str        # "high" | "medium" | "low"
#     method: str            # "heuristic" | "llm" | "heuristic+llm"
#     reason: str            # Human-readable explanation

# class ResumeValidator:
#     """
#     Two-layer resume validator with comprehensive keyword sets.
#     """
    
#     # Resume signals with weights
#     RESUME_SIGNALS: List[tuple] = [
#         # Identity / contact block (very strong resume signals)
#         ("linkedin.com",           3),
#         ("github.com",             3),
#         # Education statements
#         ("bachelor",               2),
#         ("master",                 2),
#         ("b.tech",                 2),
#         ("m.tech",                 2),
#         ("b.sc",                   2),
#         ("m.sc",                   2),
#         ("mba",                    2),
#         ("university",             2),
#         ("degree",                 2),
#         # Classic resume section headers
#         ("work experience",        2),
#         ("professional experience",2),
#         ("employment history",     2),
#         ("education",              2),
#         ("certifications",         2),
#         ("technical skills",       2),
#         ("skills",                 1),
#         ("objective",              1),
#         ("summary",                1),
#         ("achievements",           2),
#         ("projects",               1),
#         ("personal statement",     2),
#         # Developer-CV specific headers / phrases
#         ("full stack developer",   2),
#         ("software developer",     2),
#         ("software engineer",      2),
#         ("frontend developer",     2),
#         ("backend developer",      2),
#         ("freelancer",             2),
#         # Action phrases common in experience bullets
#         ("responsible for",        1),
#         ("managed",                1),
#         ("developed",              1),
#         ("led a team",             1),
#         ("experience in",          1),
#         ("proficient in",          1),
#         # Percentage / score (common in Indian CVs for marks)
#         ("percentage:",            2),
#     ]

#     # Non-resume signals with weights
#     NON_RESUME_SIGNALS: List[tuple] = [
#         # Multi-word phrases that only appear in doc/spec writing
#         ("technical documentation", 3),
#         ("system design",           2),
#         ("requirements document",   3),
#         ("data model",              2),
#         ("database schema",         2),
#         ("flow lifecycle",          3),
#         ("api endpoint",            2),
#         # Actual code syntax (with trailing space / brace to reduce false positives)
#         ("def ",                    2),
#         ("import ",                 1),
#         ("class {",                 3),
#         ("extends model",           3),
#         ("enum ",                   2),
#         # Project-management / agile docs
#         ("user story",              2),
#         ("sprint",                  2),
#         ("backlog",                 2),
#         ("readme",                  2),
#         ("changelog",               2),
#         # Academic / research
#         ("abstract",                2),
#         ("methodology",             2),
#         ("bibliography",            3),
#         ("hypothesis",              3),
#         ("literature review",       3),
#     ]

#     RESUME_NET_THRESHOLD     =  2   # net >= this → resume
#     NON_RESUME_NET_THRESHOLD = -4   # net <= this → not a resume

#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm

#     def _heuristic_check(self, text: str) -> ResumeValidationResult:
#         """Compute weighted resume and non-resume scores"""
#         lower_text = text.lower()

#         resume_score = 0
#         resume_hits: List[str] = []
#         for phrase, weight in self.RESUME_SIGNALS:
#             if phrase in lower_text:
#                 resume_score += weight
#                 resume_hits.append(phrase)

#         non_resume_score = 0
#         non_resume_hits: List[str] = []
#         for phrase, weight in self.NON_RESUME_SIGNALS:
#             if phrase in lower_text:
#                 non_resume_score += weight
#                 non_resume_hits.append(phrase)

#         net = resume_score - non_resume_score

#         logger.info(
#             f"Heuristic — resume_score: {resume_score} (hits: {resume_hits}), "
#             f"non_resume_score: {non_resume_score} (hits: {non_resume_hits}), "
#             f"net: {net}"
#         )

#         if net >= self.RESUME_NET_THRESHOLD:
#             return ResumeValidationResult(
#                 is_resume=True,
#                 confidence="high",
#                 method="heuristic",
#                 reason=(
#                     f"Document matched resume indicators with a weighted score of "
#                     f"{resume_score} vs {non_resume_score} for non-resume indicators "
#                     f"(net: +{net})."
#                 ),
#             )

#         if net <= self.NON_RESUME_NET_THRESHOLD:
#             return ResumeValidationResult(
#                 is_resume=False,
#                 confidence="high",
#                 method="heuristic",
#                 reason=(
#                     f"Document matched non-resume indicators with a weighted score of "
#                     f"{non_resume_score} vs {resume_score} for resume indicators "
#                     f"(net: {net})."
#                 ),
#             )

#         return ResumeValidationResult(
#             is_resume=False,
#             confidence="low",
#             method="heuristic",
#             reason=f"Ambiguous signal (net: {net}) — escalating to LLM classification.",
#         )

#     async def _llm_check(self, text: str) -> ResumeValidationResult:
#         """Send a lightweight classification prompt to the LLM"""
#         max_chars = 2000
#         if len(text) > max_chars:
#             half = max_chars // 2
#             snippet = text[:half] + "\n\n[... middle section omitted ...]\n\n" + text[-half:]
#         else:
#             snippet = text

#         prompt = (
#             "You are a document classifier. Read the following document excerpt and decide "
#             "whether it is a RESUME (also called a CV) or NOT a resume.\n\n"
#             "A resume/CV is a personal document that lists an individual's education, "
#             "work experience, skills, and qualifications for the purpose of job applications.\n\n"
#             "Respond with EXACTLY one of these two JSON objects and nothing else:\n"
#             '  {"is_resume": true, "reason": "<brief explanation>"}\n'
#             '  {"is_resume": false, "reason": "<brief explanation of what the document actually is>"}\n\n'
#             "Document excerpt:\n"
#             "---\n"
#             f"{snippet}\n"
#             "---\n"
#         )

#         try:
#             response = await self.llm.ainvoke(prompt)
#             response_text = response.content if hasattr(response, 'content') else str(response)
            
#             json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
#             if json_match:
#                 parsed = json.loads(json_match.group())
#                 is_resume = bool(parsed.get("is_resume", False))
#                 reason = parsed.get("reason", "LLM classification completed.")
#                 return ResumeValidationResult(
#                     is_resume=is_resume,
#                     confidence="high",
#                     method="llm",
#                     reason=reason,
#                 )
#             else:
#                 logger.warning(f"LLM validation: could not parse JSON from response: {response_text}")
#                 return ResumeValidationResult(
#                     is_resume=False,
#                     confidence="medium",
#                     method="llm",
#                     reason="LLM response could not be parsed reliably.",
#                 )
#         except Exception as e:
#             logger.error(f"LLM validation error: {e}")
#             return ResumeValidationResult(
#                 is_resume=False,
#                 confidence="low",
#                 method="llm",
#                 reason=f"LLM classification failed ({e}).",
#             )

#     async def validate(self, text: str) -> ResumeValidationResult:
#         """Run Layer 1. If ambiguous, run Layer 2."""
#         heuristic_result = self._heuristic_check(text)

#         if heuristic_result.confidence == "high":
#             logger.info(f"Validation decided by heuristic: is_resume={heuristic_result.is_resume}")
#             return heuristic_result

#         logger.info("Heuristic ambiguous — running LLM classification.")
#         llm_result = await self._llm_check(text)
#         llm_result.method = "heuristic+llm"
#         logger.info(f"Validation decided by LLM: is_resume={llm_result.is_resume}")
#         return llm_result

# # ===== PDF EXTRACTION =====

# class OptimizedPDFExtractor:
#     """Optimized PDF text extraction"""
    
#     @staticmethod
#     async def extract_text_from_pdf(uploaded_file) -> Optional[str]:
#         try:
#             uploaded_file.seek(0)
#             content = await uploaded_file.read()
            
#             def process_pdf(content_bytes):
#                 pdf_file = io.BytesIO(content_bytes)
#                 pdf_reader = PdfReader(pdf_file)
                
#                 extracted_text = ""
#                 for page_num, page in enumerate(pdf_reader.pages):
#                     try:
#                         page_text = page.extract_text()
#                         if page_text and page_text.strip():
#                             extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
#                     except Exception as page_error:
#                         logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
#                         continue
                
#                 return extracted_text.strip()
            
#             loop = asyncio.get_event_loop()
#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 extracted_text = await loop.run_in_executor(pool, process_pdf, content)
            
#             return extracted_text if extracted_text else None
            
#         except Exception as e:
#             logger.error(f"PDF extraction error: {str(e)}")
#             return None

# # ===== JOB SEARCH =====

# class JobSearchService:
#     """Service to search and parse job listings"""
    
#     def __init__(self, llm: ChatOpenAI):
#         self.llm = llm
    
#     async def search_jobs(self, target_role: str, location: str = "India") -> List[Dict[str, Any]]:
#         """Search for jobs and extract structured information"""
#         try:
#             job_extraction_prompt = f"""
#             Generate 5-10 realistic current job listings for the position: {target_role} in {location}.
            
#             For each job listing, provide EXACTLY these fields in JSON format:
#             {{
#                 "company_name": "Company name",
#                 "position": "Exact job title",
#                 "location": "City/region in {location}",
#                 "ctc": "Salary range with currency",
#                 "experience_required": "X-Y years",
#                 "last_date_to_apply": "YYYY-MM-DD format",
#                 "about_job": "2-3 sentence summary",
#                 "job_description": "Detailed responsibilities and duties",
#                 "job_requirements": "Required skills, qualifications, and education",
#                 "application_url": "https://company-careers.com/job-id"
#             }}
            
#             Return ONLY a valid JSON array with no additional text. Make the data realistic and relevant to the current job market in 2025.
#             """
            
#             response = await self.llm.ainvoke(job_extraction_prompt)
#             response_text = response.content if hasattr(response, 'content') else str(response)
            
#             # Parse the JSON response
#             try:
#                 json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
#                 if json_match:
#                     jobs_data = json.loads(json_match.group())
#                 else:
#                     jobs_data = json.loads(response_text)
                
#                 return jobs_data
#             except json.JSONDecodeError as e:
#                 logger.error(f"Failed to parse job listings JSON: {e}")
#                 return []
                
#         except Exception as e:
#             logger.error(f"Job search error: {str(e)}")
#             return []

# # ===== RESUME ANALYZER =====

# class HighPerformanceLangChainAnalyzer:
#     """High-performance AI analyzer with guaranteed standard JSON output and role-specific evaluation"""
    
#     def __init__(self, openai_api_key: str):
#         self.llm = ChatOpenAI(
#             api_key=openai_api_key,
#             model_name="gpt-3.5-turbo-16k",
#             temperature=0.0,
#             max_tokens=4000,
#             request_timeout=30
#         )
        
#         self.output_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
#         self.document_classifier = DocumentClassifier(llm=self.llm)
#         self.resume_validator = ResumeValidator(llm=self.llm)
#         self.job_search = JobSearchService(self.llm)
#         self._setup_analysis_chain()
    
#     def _setup_analysis_chain(self):
#         """Setup the analysis chain using LCEL with strict role-specific evaluation"""
        
#         analysis_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are an expert resume analyst and career counselor. Your ONLY job is to evaluate how well this resume fits the TARGET ROLE provided by the user.

# =====================================
# MANDATORY ROLE-SPECIFIC EVALUATION RULES — THESE OVERRIDE EVERYTHING ELSE
# =====================================

# 1. TARGET ROLE IS THE LENS FOR ALL ANALYSIS
#    Every strength, weakness, score, and recommendation MUST be evaluated ONLY through the lens of the target role.
#    - Target role = "Dancer" → evaluate dance training, performance experience, choreography, stage presence, physical conditioning, audition history, dance styles mastered.
#    - Target role = "Software Engineer" → evaluate coding skills, system design, projects, CS fundamentals, frameworks.
#    - Target role = "Marketing Manager" → evaluate campaign management, analytics, brand strategy, copywriting, market research.
#    NEVER analyze skills that are irrelevant to the target role as strengths.

# 2. HONEST SCORING — DO NOT INFLATE SCORES
#    The overall_score (0–100) MUST honestly reflect how qualified this candidate is for the TARGET ROLE:
#    - 0–20%   → Completely mismatched background, zero relevant experience
#    - 21–40%  → Severe mismatch, only very minor transferable skills exist
#    - 41–55%  → Partial mismatch, some transferable soft skills but missing core requirements
#    - 56–70%  → Moderate fit, has some relevant skills but lacks key requirements
#    - 71–85%  → Good fit, meets most requirements with minor gaps
#    - 86–100% → Excellent fit, strong match for the role

#    EXAMPLES OF CORRECT SCORING:
#    - CS/Engineering resume for "Dancer" role → Score: 10–25 (no dance background)
#    - Marketing resume for "Data Scientist" role → Score: 20–35 (lacks technical skills)
#    - Software Engineer resume for "Software Engineer" role → Score depends on actual skills

# 3. STRENGTHS ANALYSIS — ROLE-RELEVANT ONLY
#    List ONLY strengths that are DIRECTLY relevant or transferable to the target role.
#    - For "Dancer": Valid strengths = dance training, performance experience, flexibility, rhythm, stage presence
#    - For "Dancer": INVALID strengths = "Diverse Technical Skills", "Machine Learning Knowledge", "Python proficiency"
#    - If the candidate has NO relevant strengths, list only genuine transferable soft skills (discipline, teamwork, communication) and clearly label them as "transferable soft skills, not role-specific"
#    - Each strength entry's "why_its_strong" and "ats_benefit" fields MUST reference the target role explicitly

# 4. WEAKNESSES ANALYSIS — CALL OUT THE MISMATCH FIRST
#    If the resume background does not match the target role, the FIRST and MOST CRITICAL weakness MUST be the background mismatch itself.
#    - Example for CS resume + Dancer role: weakness = "No dance training or performance experience", fix_priority = "CRITICAL"
#    - List other role-specific gaps after the primary mismatch
#    - Be specific about WHAT is missing for the target role (e.g., "No choreography portfolio", "Missing formal dance school training")

# 5. IMPROVEMENT PLAN — TARGET ROLE SPECIFIC
#    All improvement suggestions MUST be actionable steps toward getting the target role:
#    - For "Dancer": "Enroll in a formal dance academy", "Build a video performance portfolio", "Attend local auditions", "Get certified in specific dance styles"
#    - For "Software Engineer": "Add GitHub projects", "Contribute to open source", "Get AWS certification"
#    - NEVER suggest generic improvements that don't help with the target role

# 6. JOB MARKET ANALYSIS — HONEST COMPATIBILITY
#    - role_compatibility: Set to "Low", "Moderate", or "High" based on actual fit
#    - A mismatched resume MUST get "Low" compatibility — do not soften this
#    - market_positioning: Describe the candidate's actual position in the target role's job market
#    - career_advancement: Describe the realistic path to break into the target role
#    - skill_development: List the most critical skills/training needed for the target role

# 7. DETAILED SCORING — SCORE AGAINST ROLE REQUIREMENTS
#    Each scoring category should reflect performance against WHAT THE TARGET ROLE NEEDS:
#    - contact_information: Standard (same for all roles)
#    - technical_skills: Score based on skills RELEVANT TO TARGET ROLE only
#    - experience_quality: Score based on experience IN OR RELATED TO TARGET ROLE only
#    - quantified_achievements: Score based on achievements RELEVANT TO TARGET ROLE
#    - content_optimization: Score based on how well the resume is optimized FOR THE TARGET ROLE

# TARGET ROLE: {target_role}

# {format_instructions}

# FINAL REMINDER: Return ONLY valid JSON. Be honest. Be role-specific. Do not inflate scores. A candidate deserves accurate feedback to make real career decisions."""),
#             ("human", "Target Role: {target_role}\n\nResume Content:\n{resume_text}")
#         ]).partial(format_instructions=self.output_parser.get_format_instructions())
        
#         self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()
    
#     async def _check_role_mismatch(self, resume_text: str, target_role: str) -> dict:
#         """
#         Quick pre-check to detect obvious role-resume mismatches.
#         Returns mismatch metadata that is attached to the final response.
#         """
#         prompt = f"""You are a career advisor. Assess whether this resume's background is relevant to the target role.

# Target Role: {target_role}
# Resume Snippet (first 1500 chars): {resume_text[:1500]}

# Be honest and specific. If the candidate's background is completely unrelated to the target role, say so clearly.

# Respond with JSON only — no extra text:
# {{
#   "is_relevant": true or false,
#   "compatibility": "Low" or "Moderate" or "High",
#   "candidate_background": "One sentence describing what field/domain this resume is actually from",
#   "target_role_requirements": "One sentence describing what the target role actually needs",
#   "mismatch_reason": "One sentence explaining the gap (or 'Good match' if compatible)",
#   "estimated_score_range": "e.g. 10-25 or 60-75"
# }}"""

#         try:
#             response = await self.llm.ainvoke(prompt)
#             text = response.content if hasattr(response, 'content') else str(response)
#             json_match = re.search(r'\{.*?\}', text, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
#                 logger.info(f"Role mismatch check: compatibility={result.get('compatibility')}, is_relevant={result.get('is_relevant')}")
#                 return result
#         except Exception as e:
#             logger.warning(f"Role mismatch check failed: {e}")
        
#         return {
#             "is_relevant": True,
#             "compatibility": "Unknown",
#             "candidate_background": "Could not assess",
#             "target_role_requirements": "Could not assess",
#             "mismatch_reason": "Assessment unavailable",
#             "estimated_score_range": "N/A"
#         }

#     def _get_standard_response_template(self, target_role: str, word_count: int) -> Dict[str, Any]:
#         """Returns the standard response structure"""
#         return {
#             "success": True,
#             "analysis_method": "AI-Powered LangChain Analysis with Three-Layer Validation",
#             "resume_metadata": {
#                 "word_count": word_count,
#                 "validation_message": "Comprehensive AI analysis completed",
#                 "target_role": target_role or "general position"
#             },
#             "executive_summary": {
#                 "professional_profile": {},
#                 "contact_presentation": {},
#                 "overall_assessment": {}
#             },
#             "detailed_scoring": {},
#             "strengths_analysis": [],
#             "weaknesses_analysis": [],
#             "improvement_plan": {
#                 "critical": [],
#                 "high": [],
#                 "medium": []
#             },
#             "job_market_analysis": {},
#             "ai_insights": {},
#             "role_fit_assessment": {}
#         }
    
#     def _convert_to_snake_case(self, key: str) -> str:
#         """Convert title case to snake_case"""
#         mapping = {
#             "Contact Information": "contact_information",
#             "Technical Skills": "technical_skills",
#             "Experience Quality": "experience_quality",
#             "Quantified Achievements": "quantified_achievements",
#             "Content Optimization": "content_optimization"
#         }
#         return mapping.get(key, key.lower().replace(" ", "_"))
    
#     async def analyze_resume_with_jobs(
#         self, 
#         resume_text: str, 
#         username: str,
#         target_role: Optional[str] = None,
#         search_jobs: bool = True,
#         location: str = "India"
#     ) -> Dict[str, Any]:
#         """Analyze resume with role-specific evaluation and optional job search"""
#         try:
#             role_context = target_role or "general position"
#             word_count = len(resume_text.split())
            
#             # Check cache first
#             cache_key = get_content_hash(resume_text, role_context)
#             if cache_key in analysis_cache:
#                 logger.info("Returning cached analysis result")
#                 return analysis_cache[cache_key]
            
#             # Initialize response with standard structure
#             response = self._get_standard_response_template(role_context, word_count)
            
#             # Run role mismatch check first (lightweight, fast)
#             logger.info(f"Running role fit check for target role: {role_context}")
#             mismatch_info = await self._check_role_mismatch(resume_text, role_context)
            
#             # Run resume analysis and job search in parallel if needed
#             if search_jobs and target_role:
#                 analysis_task = self.analysis_chain.ainvoke({
#                     "resume_text": resume_text,
#                     "target_role": role_context
#                 })
#                 jobs_task = self.job_search.search_jobs(target_role, location)
                
#                 analysis_result, job_listings = await asyncio.gather(
#                     analysis_task,
#                     jobs_task,
#                     return_exceptions=True
#                 )
                
#                 if isinstance(analysis_result, Exception):
#                     raise analysis_result
#                 if isinstance(job_listings, Exception):
#                     logger.error(f"Job search failed: {job_listings}")
#                     job_listings = []
#             else:
#                 analysis_result = await self.analysis_chain.ainvoke({
#                     "resume_text": resume_text,
#                     "target_role": role_context
#                 })
#                 job_listings = []
            
#             # Parse and populate response
#             try:
#                 parsed_analysis = self.output_parser.parse(analysis_result)
#                 self._populate_response(response, parsed_analysis, word_count, role_context)
                
#             except Exception as parse_error:
#                 logger.warning(f"Structured parsing failed, using fallback: {parse_error}")
#                 self._populate_fallback_response(response, analysis_result, word_count, role_context)
            
#             # Attach role fit assessment to response
#             response["role_fit_assessment"] = {
#                 "target_role": role_context,
#                 "is_relevant": mismatch_info.get("is_relevant", True),
#                 "compatibility": mismatch_info.get("compatibility", "Unknown"),
#                 "candidate_background": mismatch_info.get("candidate_background", ""),
#                 "target_role_requirements": mismatch_info.get("target_role_requirements", ""),
#                 "mismatch_reason": mismatch_info.get("mismatch_reason", ""),
#                 "estimated_score_range": mismatch_info.get("estimated_score_range", ""),
#                 "note": (
#                     "⚠️ This score reflects fit with the specified target role, not overall resume quality. "
#                     "The same resume may score much higher for a role that matches the candidate's background."
#                     if not mismatch_info.get("is_relevant", True)
#                     else "Score reflects fit with the specified target role."
#                 )
#             }

#             # Add job listings if available
#             if job_listings:
#                 response["job_listings"] = {
#                     "total_jobs_found": len(job_listings),
#                     "search_query": f"{target_role} in {location}",
#                     "jobs": job_listings
#                 }
            
#             # Add username to response
#             response["username"] = username
            
#             # Cache the result
#             analysis_cache[cache_key] = response
#             logger.info(f"Cached analysis result for key: {cache_key}")
            
#             return response
                
#         except Exception as e:
#             logger.error(f"Analysis error: {str(e)}")
#             return self._generate_error_response(str(e), target_role, word_count, username)
    
#     def _populate_response(self, response: Dict, analysis: ResumeAnalysis, word_count: int, target_role: str):
#         """Populate response with parsed analysis data"""
        
#         response["executive_summary"] = {
#             "professional_profile": {
#                 "experience_level": analysis.professional_profile.experience_level,
#                 "technical_skills_count": analysis.professional_profile.technical_skills_count,
#                 "project_portfolio_size": analysis.professional_profile.project_portfolio_size,
#                 "achievement_metrics": analysis.professional_profile.achievement_metrics,
#                 "technical_sophistication": analysis.professional_profile.technical_sophistication
#             },
#             "contact_presentation": {
#                 "email_address": analysis.contact_presentation.email_address,
#                 "phone_number": analysis.contact_presentation.phone_number,
#                 "education": analysis.contact_presentation.education,
#                 "resume_length": analysis.contact_presentation.resume_length,
#                 "action_verbs": analysis.contact_presentation.action_verbs
#             },
#             "overall_assessment": {
#                 "score_percentage": analysis.overall_score,
#                 "level": analysis.recommendation_level,
#                 "description": f"Role-fit score for '{target_role}': {analysis.overall_score}%",
#                 "recommendation": analysis.recommendation_level
#             }
#         }
        
#         response["detailed_scoring"] = {}
#         for key, detail in analysis.detailed_scoring.items():
#             snake_case_key = self._convert_to_snake_case(key)
#             response["detailed_scoring"][snake_case_key] = {
#                 "score": detail.score,
#                 "max_score": detail.max_score,
#                 "percentage": detail.percentage,
#                 "details": detail.details
#             }
        
#         response["strengths_analysis"] = [
#             {
#                 "strength": s.strength,
#                 "why_its_strong": s.why_its_strong,
#                 "ats_benefit": s.ats_benefit,
#                 "competitive_advantage": s.competitive_advantage,
#                 "evidence": s.evidence
#             }
#             for s in analysis.strengths_analysis
#         ]
        
#         response["weaknesses_analysis"] = [
#             {
#                 "weakness": w.weakness,
#                 "why_problematic": w.why_problematic,
#                 "ats_impact": w.ats_impact,
#                 "how_it_hurts": w.how_it_hurts,
#                 "fix_priority": w.fix_priority,
#                 "specific_fix": w.specific_fix,
#                 "timeline": w.timeline
#             }
#             for w in analysis.weaknesses_analysis
#         ]
        
#         response["improvement_plan"] = {
#             "critical": analysis.improvement_plan.critical,
#             "high": analysis.improvement_plan.high,
#             "medium": analysis.improvement_plan.medium
#         }
        
#         response["job_market_analysis"] = {
#             "role_compatibility": analysis.job_market_analysis.role_compatibility,
#             "market_positioning": analysis.job_market_analysis.market_positioning,
#             "career_advancement": analysis.job_market_analysis.career_advancement,
#             "skill_development": analysis.job_market_analysis.skill_development
#         }
        
#         response["ai_insights"] = {
#             "overall_score": analysis.overall_score,
#             "recommendation_level": analysis.recommendation_level,
#             "key_strengths_count": len(analysis.strengths_analysis),
#             "improvement_areas_count": len(analysis.weaknesses_analysis)
#         }
    
#     def _populate_fallback_response(self, response: Dict, raw_result: str, word_count: int, target_role: str):
#         """Fallback method to populate response from raw LLM output"""
#         try:
#             json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
#             if json_match:
#                 parsed_data = json.loads(json_match.group())
                
#                 if "professional_profile" in parsed_data:
#                     response["executive_summary"]["professional_profile"] = parsed_data["professional_profile"]
#                 if "contact_presentation" in parsed_data:
#                     response["executive_summary"]["contact_presentation"] = parsed_data["contact_presentation"]
#                 if "overall_score" in parsed_data:
#                     response["executive_summary"]["overall_assessment"] = {
#                         "score_percentage": parsed_data.get("overall_score", 0),
#                         "level": parsed_data.get("recommendation_level", "Unknown"),
#                         "description": f"Role-fit score for '{target_role}': {parsed_data.get('overall_score', 0)}%",
#                         "recommendation": parsed_data.get("recommendation_level", "Unknown")
#                     }
                
#                 detailed_scoring = parsed_data.get("detailed_scoring", {})
#                 converted_scoring = {}
#                 for key, value in detailed_scoring.items():
#                     snake_case_key = self._convert_to_snake_case(key)
#                     converted_scoring[snake_case_key] = value
#                 response["detailed_scoring"] = converted_scoring
                
#                 response["strengths_analysis"] = parsed_data.get("strengths_analysis", [])
#                 response["weaknesses_analysis"] = parsed_data.get("weaknesses_analysis", [])
#                 response["improvement_plan"] = parsed_data.get("improvement_plan", {"critical": [], "high": [], "medium": []})
#                 response["job_market_analysis"] = parsed_data.get("job_market_analysis", {})
#                 response["ai_insights"] = {
#                     "overall_score": parsed_data.get("overall_score", 0),
#                     "recommendation_level": parsed_data.get("recommendation_level", "Unknown"),
#                     "key_strengths_count": len(parsed_data.get("strengths_analysis", [])),
#                     "improvement_areas_count": len(parsed_data.get("weaknesses_analysis", []))
#                 }
                
#         except Exception as e:
#             logger.error(f"Fallback parsing error: {e}")
    
#     def _generate_error_response(self, error_message: str, target_role: str = None, word_count: int = 0, username: str = None) -> Dict[str, Any]:
#         """Generate error response maintaining standard structure"""
#         response = self._get_standard_response_template(target_role or "unknown", word_count)
#         response["success"] = False
#         response["error"] = f"AI analysis failed: {error_message}"
#         response["resume_metadata"]["validation_message"] = "Analysis encountered an error"
#         if username:
#             response["username"] = username
#         return response

# # ===== INITIALIZE COMPONENTS =====

# pdf_extractor = OptimizedPDFExtractor()
# high_perf_analyzer = None

# if openai_api_key:
#     try:
#         high_perf_analyzer = HighPerformanceLangChainAnalyzer(openai_api_key)
#         logger.info("High-performance analyzer initialized successfully")
#     except Exception as init_error:
#         logger.error(f"Failed to initialize analyzer: {init_error}")

# # ===== ENDPOINTS =====

# @app.post("/analyze-resume")
# async def analyze_resume(
#     file: UploadFile = File(..., description="Resume PDF file"),
#     username: str = Form(..., description="Username for whom the analysis is being done"),
#     target_role: str = Form(None, description="Target job position/role"),
#     search_jobs: bool = Form(True, description="Whether to search for relevant jobs"),
#     location: str = Form("India", description="Location for job search")
# ):
#     """
#     Comprehensive resume analysis with role-specific scoring, honest mismatch detection,
#     and guaranteed standard JSON output. Includes three-layer validation and job search integration.
#     """
#     start_time = asyncio.get_event_loop().time()
    
#     try:
#         if not high_perf_analyzer:
#             raise HTTPException(status_code=500, detail="AI analyzer not initialized.")
        
#         if not file.content_type or "pdf" not in file.content_type.lower():
#             raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
#         # Extract PDF text
#         resume_text = await pdf_extractor.extract_text_from_pdf(file)
        
#         if not resume_text:
#             raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
#         if len(resume_text.strip()) < 100:
#             raise HTTPException(status_code=400, detail="Resume content too short.")

#         # THREE-LAYER VALIDATION PIPELINE
#         # Layer 1: Document Classification
#         logger.info("Running Layer 1: Document classification")
#         classification_result = await high_perf_analyzer.document_classifier.classify(resume_text)
        
#         if classification_result.label == "non_resume" and classification_result.confidence >= 0.7:
#             raise HTTPException(
#                 status_code=400,
#                 detail={
#                     "error": "not_a_resume",
#                     "message": "The uploaded document does not appear to be a resume/CV.",
#                     "validation": {
#                         "is_resume": False,
#                         "confidence": "high",
#                         "method": "llm_classifier",
#                         "reason": classification_result.reason,
#                         "classifier_confidence": classification_result.confidence,
#                     },
#                 },
#             )
        
#         logger.info(
#             f"Layer 1 result: {classification_result.label} "
#             f"(confidence: {classification_result.confidence:.2f})"
#         )
        
#         # Layer 2 & 3: Heuristic + LLM Validation
#         logger.info("Running Layer 2/3: Heuristic + LLM validation")
#         validation_result = await high_perf_analyzer.resume_validator.validate(resume_text)

#         if not validation_result.is_resume:
#             raise HTTPException(
#                 status_code=400,
#                 detail={
#                     "error": "not_a_resume",
#                     "message": "The uploaded document does not appear to be a resume/CV.",
#                     "validation": {
#                         "is_resume": validation_result.is_resume,
#                         "confidence": validation_result.confidence,
#                         "method": validation_result.method,
#                         "reason": validation_result.reason,
#                         "classifier_label": classification_result.label,
#                         "classifier_confidence": classification_result.confidence,
#                     },
#                 },
#             )

#         logger.info(
#             f"All validation layers passed (final method={validation_result.method}, "
#             f"confidence={validation_result.confidence}). Proceeding to analysis."
#         )
        
#         # Perform analysis
#         analysis_result = await asyncio.wait_for(
#             high_perf_analyzer.analyze_resume_with_jobs(
#                 resume_text=resume_text,
#                 username=username,
#                 target_role=target_role,
#                 search_jobs=search_jobs and bool(target_role),
#                 location=location
#             ),
#             timeout=60.0
#         )

#         # Save to shared database
#         analysis_id = str(uuid.uuid4())
#         shared_db.save_resume_analysis(
#             username=username,
#             analysis_id=analysis_id,
#             analysis_data={
#                 "target_role": target_role or "general position",
#                 "overall_score": analysis_result.get("ai_insights", {}).get("overall_score", 0),
#                 "recommendation_level": analysis_result.get("ai_insights", {}).get("recommendation_level", "Unknown"),
#                 "role_compatibility": analysis_result.get("role_fit_assessment", {}).get("compatibility", "Unknown"),
#                 "analysis_result": analysis_result,
#                 "uploaded_at": datetime.now().isoformat(),
#                 "validation_method": validation_result.method,
#                 "validation_confidence": validation_result.confidence
#             }
#         )
        
#         analysis_result["analysis_id"] = analysis_id
#         analysis_result["saved_to_database"] = True
        
#         processing_time = asyncio.get_event_loop().time() - start_time
#         logger.info(f"Analysis completed in {processing_time:.2f}s for user: {username}")
        
#         return analysis_result
        
#     except asyncio.TimeoutError:
#         raise HTTPException(status_code=408, detail="Analysis timeout.")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Analysis endpoint error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# @app.get("/user/{username}/analyses")
# async def get_user_analyses(username: str):
#     """Get all analyses for a specific user"""
#     try:
#         analyses = shared_db.get_user_resume_analyses(username)
#         return {
#             "username": username, 
#             "total_analyses": len(analyses), 
#             "analyses": analyses
#         }
#     except Exception as e:
#         logger.error(f"Error fetching analyses for user {username}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analyses: {str(e)}")

# @app.get("/analysis/{username}/{analysis_id}")
# async def get_analysis(username: str, analysis_id: str):
#     """Get a specific analysis by ID for a user"""
#     try:
#         analyses = shared_db.get_user_resume_analyses(username)
#         for analysis in analyses:
#             if analysis.get("analysis_id") == analysis_id:
#                 return analysis
        
#         raise HTTPException(status_code=404, detail="Analysis not found")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching analysis {analysis_id} for user {username}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

# @app.delete("/analysis/{username}/{analysis_id}")
# async def delete_analysis(username: str, analysis_id: str):
#     """Delete a specific analysis"""
#     try:
#         shared_db.delete_interaction(username, "resume_analyzer", analysis_id)
#         return {"message": f"Analysis {analysis_id} deleted successfully for user {username}"}
#     except Exception as e:
#         logger.error(f"Error deleting analysis {analysis_id} for user {username}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

# @app.get("/health")
# async def health_check():
#     """Health check with comprehensive features display"""
#     try:
#         all_users = shared_db.get_all_users()
#         all_analyses = []
#         analyses_by_user = {}
        
#         for user in all_users:
#             analyses = shared_db.get_user_resume_analyses(user)
#             all_analyses.extend(analyses)
#             analyses_by_user[user] = len(analyses)
        
#         return {
#             "status": "healthy",
#             "timestamp": datetime.now().isoformat(),
#             "service": "AI Resume Analyzer with Role-Specific Analysis",
#             "version": "5.0.0",
#             "features": {
#                 "three_layer_validation": "✅",
#                 "llm_document_classifier": "✅",
#                 "heuristic_validator": "✅",
#                 "llm_validator": "✅",
#                 "role_specific_analysis": "✅",
#                 "honest_scoring": "✅",
#                 "mismatch_detection": "✅",
#                 "role_fit_assessment": "✅",
#                 "resume_analysis": "✅",
#                 "job_search_integration": "✅",
#                 "ats_scoring": "✅",
#                 "strengths_analysis": "✅",
#                 "weaknesses_analysis": "✅",
#                 "improvement_plan": "✅",
#                 "job_market_analysis": "✅",
#                 "quantified_scoring": "✅",
#                 "detailed_breakdown": "✅",
#                 "caching_mechanism": "✅",
#                 "shared_database": "✅",
#                 "user_tracking": "✅",
#                 "per_user_analyses": "✅",
#                 "analysis_history": "✅",
#                 "pdf_extraction": "✅",
#                 "error_handling": "✅",
#                 "performance_optimization": "✅",
#                 "consistent_json_output": "✅",
#                 "snake_case_naming": "✅",
#                 "deterministic_output": "✅"
#             },
#             "validation_pipeline": {
#                 "layer1": "LLM Document Classifier - Quick pre-screening",
#                 "layer2": "Heuristic Validator - Fast keyword-based scoring",
#                 "layer3": "LLM Validator - Deep analysis for ambiguous cases"
#             },
#             "role_specific_scoring": {
#                 "description": "Scores now reflect fit with the TARGET ROLE, not generic resume quality",
#                 "mismatch_detection": "Pre-analysis role-fit check runs before full analysis",
#                 "honest_scoring_bands": {
#                     "0-20": "Completely mismatched background",
#                     "21-40": "Severe mismatch, very few transferable skills",
#                     "41-55": "Partial mismatch, some transferable soft skills",
#                     "56-70": "Moderate fit, lacks key role requirements",
#                     "71-85": "Good fit, meets most requirements",
#                     "86-100": "Excellent fit, strong match for the role"
#                 }
#             },
#             "database": {
#                 "type": "external_api",
#                 "url": EXTERNAL_DB_API_URL,
#                 "total_users": len(all_users),
#                 "total_analyses": len(all_analyses),
#                 "analyses_by_user": analyses_by_user
#             },
#             "performance": {
#                 "caching_enabled": True,
#                 "cache_size": len(analysis_cache),
#                 "parallel_processing": True,
#                 "optimized_pdf_extraction": True
#             },
#             "openai_configured": bool(openai_api_key),
#             "analyzer_available": bool(high_perf_analyzer),
#             "langchain_version": "Latest (LCEL)",
#             "guarantees": [
#                 "✅ Non-resume documents rejected before analysis",
#                 "✅ Role-specific strengths and weaknesses only",
#                 "✅ Honest scores that reflect actual role fit",
#                 "✅ Mismatch detection with clear explanation",
#                 "✅ Consistent JSON structure every time",
#                 "✅ All standard fields present",
#                 "✅ Snake case field naming in detailed_scoring",
#                 "✅ Frontend-compatible format",
#                 "✅ Optional job listings",
#                 "✅ Deterministic output for identical resumes",
#                 "✅ Per-user analysis tracking",
#                 "✅ Full analysis history available"
#             ]
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         return {
#             "status": "degraded",
#             "service": "AI Resume Analyzer",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }

# @app.get("/")
# async def root():
#     """Root endpoint with comprehensive feature listing"""
#     return {
#         "service": "AI Resume Analyzer with Role-Specific Analysis",
#         "version": "5.0.0",
#         "description": "AI resume analysis with three-layer validation, role-specific scoring, and honest mismatch detection",
#         "what_changed_in_v5": {
#             "problem_fixed": "Previously the analyzer gave high scores and generic strengths regardless of role fit",
#             "fix_1": "System prompt now enforces strict role-specific evaluation for ALL analysis sections",
#             "fix_2": "Added pre-analysis role-fit check (_check_role_mismatch) that runs before full analysis",
#             "fix_3": "Added role_fit_assessment block in every response with compatibility rating and mismatch explanation",
#             "fix_4": "Honest scoring bands defined — a CS resume for Dancer role now correctly scores 10-25%",
#             "fix_5": "Strengths now only list skills RELEVANT to the target role",
#             "fix_6": "Primary weakness is now the background mismatch itself when roles don't align"
#         },
#         "features": {
#             "role_specific_analysis": "✅ All strengths, weaknesses, and scores tied to target role",
#             "honest_scoring": "✅ Low score for mismatched roles, high for matching ones",
#             "mismatch_detection": "✅ Pre-analysis check detects role-background mismatch",
#             "role_fit_assessment": "✅ Dedicated block in response explaining compatibility",
#             "validation_pipeline": "✅ Three-layer validation (LLM classifier + heuristic + LLM validator)",
#             "resume_analysis": "✅ Complete resume analysis with ATS scoring",
#             "job_search": "✅ Integrated job search with realistic listings",
#             "scoring_system": "✅ Multi-category scoring with detailed breakdowns",
#             "strengths_analysis": "✅ Role-relevant strengths only",
#             "weaknesses_analysis": "✅ Mismatch-first weaknesses with fix priorities",
#             "improvement_plan": "✅ Role-specific actionable recommendations",
#             "job_market_analysis": "✅ Honest role compatibility and market positioning",
#             "caching": "✅ Content-based caching for consistent results",
#             "database": "✅ Shared database integration",
#             "user_tracking": "✅ Per-user analysis storage and retrieval",
#             "analysis_history": "✅ Full analysis history for each user",
#             "pdf_extraction": "✅ Optimized PDF text extraction",
#             "error_handling": "✅ Comprehensive error handling",
#             "performance": "✅ Parallel processing and optimization"
#         },
#         "endpoints": {
#             "/analyze-resume": {
#                 "method": "POST",
#                 "description": "Role-specific analysis with three-layer validation",
#                 "content_type": "multipart/form-data",
#                 "fields": {
#                     "file": "PDF file (required)",
#                     "username": "string (required) - User identifier",
#                     "target_role": "string (optional) - Critical for role-specific scoring",
#                     "search_jobs": "boolean (default: true)",
#                     "location": "string (default: India)"
#                 }
#             },
#             "/user/{username}/analyses": {
#                 "method": "GET",
#                 "description": "Get all analyses for a specific user"
#             },
#             "/analysis/{username}/{analysis_id}": {
#                 "method": "GET",
#                 "description": "Get a specific analysis by ID"
#             },
#             "/analysis/{username}/{analysis_id}": {
#                 "method": "DELETE",
#                 "description": "Delete a specific analysis"
#             },
#             "/health": {
#                 "method": "GET",
#                 "description": "Service health check with features"
#             }
#         },
#         "scoring_guide": {
#             "0-20":   "Completely mismatched background — no relevant experience for target role",
#             "21-40":  "Severe mismatch — only very minor transferable skills",
#             "41-55":  "Partial mismatch — transferable soft skills but missing core requirements",
#             "56-70":  "Moderate fit — some relevant skills but lacks key requirements",
#             "71-85":  "Good fit — meets most requirements with minor gaps",
#             "86-100": "Excellent fit — strong match for the target role"
#         }
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("=" * 70)
#     print("🚀 Starting AI Resume Analyzer v5.0 — Role-Specific Analysis")
#     print("=" * 70)
#     print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
#     print(f"🔑 OpenAI: {'✅ Configured' if openai_api_key else '❌ Not configured'}")
#     print(f"🔧 Analyzer: {'✅ Ready' if high_perf_analyzer else '❌ Not available'}")
#     print(f"🎯 Version: 5.0.0")
#     print(f"")
#     print(f"🆕 What's new in v5.0:")
#     print(f"   • Role-Specific Scoring: ✅  (scores now reflect TARGET ROLE fit)")
#     print(f"   • Honest Mismatch Detection: ✅  (CS resume for Dancer = low score)")
#     print(f"   • Role Fit Assessment block: ✅  (in every API response)")
#     print(f"   • Role-only Strengths: ✅  (no more generic technical skill praise)")
#     print(f"   • Mismatch-first Weaknesses: ✅  (primary gap called out clearly)")
#     print(f"   • Role-specific Improvement Plan: ✅")
#     print(f"")
#     print(f"💬 Core Features:")
#     print(f"   • Three-Layer Validation: ✅")
#     print(f"   • LLM Document Classifier: ✅")
#     print(f"   • Heuristic Validator: ✅")
#     print(f"   • LLM Validator: ✅")
#     print(f"   • Resume Analysis: ✅")
#     print(f"   • Job Search Integration: ✅")
#     print(f"   • ATS Scoring: ✅")
#     print(f"   • Caching Mechanism: ✅")
#     print(f"   • Shared Database: ✅")
#     print(f"   • User Tracking: ✅")
#     print(f"   • Analysis History: ✅")
#     print(f"   • PDF Extraction: ✅")
#     print(f"   • Error Handling: ✅")
#     print(f"   • Consistent JSON Output: ✅")
#     print(f"")
#     print(f"🔗 API: http://localhost:8002")
#     print(f"📚 Docs: http://localhost:8002/docs")
#     print("=" * 70)
    
#     uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")





# _________________________Chatbot_________________________


# """
# AI Academic Chatbot with FULL Personalization Integration & Enhanced Features
# This version combines friend-like conversational style with comprehensive personalization
# and dynamic resume awareness
# """

# import os
# import logging
# import json
# import re
# import requests
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# from collections import defaultdict
# import uuid
# from pathlib import Path

# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel, Field
# import uvicorn

# # LangChain imports
# from langchain_openai import ChatOpenAI
# import openai

# from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
# from langchain_core.exceptions import OutputParserException

# from dotenv import load_dotenv

# # Import shared database
# from shared_database import SharedDatabase, EXTERNAL_DB_API_URL

# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app
# app = FastAPI(
#     title="AI Academic Chatbot with Personalization & Enhanced Features",
#     description="Personalized chatbot with friend-like conversations, smart intent detection, and resume awareness",
#     version="6.0.0"
# )

# # ============================
# # Personalization Integration
# # ============================

# class PersonalizationIntegration:
#     """Handles all personalization API calls and context building"""
    
#     def __init__(self, personalization_url: str = "http://localhost:8001"):
#         self.api_url = personalization_url
#         self.cache = {}  # Cache personalization data
#         self.cache_timeout = 300  # 5 minutes
    
#     def get_user_profile(self, username: str) -> Optional[Dict]:
#         """Fetch user profile from personalization module"""
#         try:
#             # Check cache
#             cache_key = f"profile_{username}"
#             if cache_key in self.cache:
#                 cached_data, timestamp = self.cache[cache_key]
#                 if (datetime.now().timestamp() - timestamp) < self.cache_timeout:
#                     return cached_data
            
#             response = requests.get(f"{self.api_url}/user/{username}/profile", timeout=5)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 self.cache[cache_key] = (data, datetime.now().timestamp())
#                 return data
#             else:
#                 logger.warning(f"Personalization API returned {response.status_code}")
#                 return None
                
#         except requests.exceptions.ConnectionError:
#             logger.warning("Personalization module not available")
#             return None
#         except Exception as e:
#             logger.error(f"Error fetching profile: {e}")
#             return None
    
#     def build_personalization_context(self, username: str) -> str:
#         """Build comprehensive personalization context for LLM"""
#         profile = self.get_user_profile(username)
        
#         if not profile or not profile.get("data_available", False):
#             return ""
        
#         context_parts = ["\n=== USER PERSONALIZATION CONTEXT ==="]
        
#         # Personality traits
#         traits = profile.get("personality_traits", {})
#         if traits:
#             high_traits = [k.replace('_', ' ').title() for k, v in traits.items() if v > 0.6]
#             if high_traits:
#                 context_parts.append(f"🧠 Personality: {', '.join(high_traits)}")
        
#         # Communication style
#         comm_style = profile.get("communication_style", {})
#         if comm_style:
#             formality = comm_style.get("formality", "mixed")
#             verbosity = comm_style.get("verbosity", "moderate")
#             context_parts.append(f"💬 Communication: {formality} tone, {verbosity} responses")
        
#         # Topics of interest
#         topics = profile.get("topics_of_interest", [])
#         if topics:
#             context_parts.append(f"📚 Interests: {', '.join(topics[:5])}")
        
#         # Professional interests from resume
#         prof_interests = profile.get("professional_interests", [])
#         if prof_interests:
#             context_parts.append(f"💼 Professional Interests: {', '.join(prof_interests[:5])}")
        
#         # Career goals
#         career_goals = profile.get("career_goals", [])
#         if career_goals:
#             context_parts.append(f"🎯 Career Goals: {', '.join(career_goals[:3])}")
        
#         # Skill levels
#         skills = profile.get("skill_levels", {})
#         if skills:
#             skill_info = ", ".join([f"{k}: {v}" for k, v in skills.items()])
#             context_parts.append(f"🎯 Skills: {skill_info}")
        
#         # Resume insights (IMPORTANT!)
#         resume_insights = profile.get("resume_insights", {})
#         if resume_insights and resume_insights.get("total_analyses", 0) > 0:
#             avg_score = resume_insights.get("average_score", 0)
#             trend = resume_insights.get("improvement_trend", "stable")
#             target_roles = resume_insights.get("target_roles", [])
#             strengths = resume_insights.get("common_strengths", [])
#             weaknesses = resume_insights.get("common_weaknesses", [])
            
#             context_parts.append(f"📄 **Resume Performance:**")
#             context_parts.append(f"   - Average Score: {avg_score}%")
#             context_parts.append(f"   - Trend: {trend}")
#             if target_roles:
#                 context_parts.append(f"   - Target Roles: {', '.join(target_roles[:3])}")
#             if strengths:
#                 context_parts.append(f"   - Key Strengths: {', '.join(strengths[:3])}")
#             if weaknesses:
#                 context_parts.append(f"   - Areas to Improve: {', '.join(weaknesses[:3])}")
            
#             # Add recent analyses
#             analyses_history = resume_insights.get("analyses_history", [])
#             if analyses_history:
#                 latest = analyses_history[0]
#                 context_parts.append(f"   - Latest: {latest.get('score')}% for {latest.get('role')}")
        
#         # Recommendations
#         recommendations = profile.get("recommendations", {})
#         if recommendations:
#             learning_recs = recommendations.get("learning_style", [])
#             if learning_recs:
#                 context_parts.append(f"💡 Recommendations: {'; '.join(learning_recs[:2])}")
        
#         context_parts.append("=== END PERSONALIZATION CONTEXT ===\n")
        
#         return "\n".join(context_parts)
    
#     def get_detailed_resume_insights(self, username: str) -> Dict[str, Any]:
#         """Get detailed resume insights from personalization module"""
#         profile = self.get_user_profile(username)
        
#         if not profile:
#             return {}
        
#         return profile.get("resume_insights", {})
    
#     def trigger_profile_update(self, username: str):
#         """Trigger profile update in background"""
#         try:
#             requests.post(f"{self.api_url}/user/{username}/update", timeout=2)
#         except:
#             pass  # Non-critical, fail silently


# # ============================
# # Request/Response Models
# # ============================

# class ChatRequest(BaseModel):
#     message: str

# class CollegeRecommendation(BaseModel):
#     """College recommendation model"""
#     id: str
#     name: str
#     location: str
#     type: str
#     courses_offered: str
#     website: str
#     admission_process: str
#     approximate_fees: str
#     notable_features: str
#     source: str

# class ChatResponse(BaseModel):
#     response: str
#     is_recommendation: bool
#     timestamp: str
#     conversation_title: Optional[str] = None
#     recommendations: Optional[List[CollegeRecommendation]] = []
#     personalized: bool = False

# class UserPreferences(BaseModel):
#     """User preferences extracted from conversation"""
#     location: Optional[str] = Field(None, description="Preferred city or state for college")
#     state: Optional[str] = Field(None, description="Preferred state for college")
#     course_type: Optional[str] = Field(None, description="Type of course like Engineering, Medicine, Arts, Commerce, etc.")
#     college_type: Optional[str] = Field(None, description="Government, Private, or Deemed university")
#     level: Optional[str] = Field(None, description="UG (Undergraduate) or PG (Postgraduate)")
#     budget_range: Optional[str] = Field(None, description="Budget preference like low, medium, high")
#     specific_course: Optional[str] = Field(None, description="Specific course like BTech, MBA, MBBS, etc.")
#     specific_institution_type: Optional[str] = Field(None, description="Specific institution type like IIT, NIT, IIIT, AIIMS, etc.")

# # ============================
# # Conversation Memory Manager (Enhanced for Shared DB)
# # ============================

# class ConversationMemoryManager:
#     """Manages conversation memory with Shared Database persistence - Enhanced"""
    
#     def __init__(self, db: SharedDatabase):
#         self.db = db
#         self.active_memories = {}  # In-memory cache
#         # SINGLE UNIFIED MEMORY - maintains context across ALL conversations
#         self.chat_memories = defaultdict(lambda: [])  # Simple list instead of ChatMessageHistory
    
#     def load_conversation(self, chat_id: str, username: str) -> dict:
#         """Load conversation from database"""
#         conv = self.db.get_chatbot_conversation(username, chat_id)
#         if conv:
#             self.active_memories[chat_id] = conv
            
#             # Also load messages into memory
#             for msg in conv.get('messages', []):
#                 if msg['role'] == 'human':
#                     self.chat_memories[chat_id].append(
#                         HumanMessage(content=msg['content'])
#                     )
#                 elif msg['role'] == 'ai':
#                     self.chat_memories[chat_id].append(
#                         AIMessage(content=msg['content'])
#                     )
#             return conv
#         return None
    
#     def add_message(self, chat_id: str, username: str, role: str, content: str, is_recommendation: bool = False):
#         """Add message to conversation"""
#         if chat_id not in self.active_memories:
#             conv = self.db.get_chatbot_conversation(username, chat_id)
#             if conv:
#                 self.active_memories[chat_id] = conv
#             else:
#                 self.active_memories[chat_id] = {
#                     "title": "New Conversation",
#                     "messages": [],
#                     "preferences": {}
#                 }
        
#         self.active_memories[chat_id]['messages'].append({
#             'role': role,
#             'content': content,
#             'is_recommendation': is_recommendation,
#             'timestamp': datetime.now().isoformat()
#         })
        
#         # Add to memory for context
#         if role == 'human':
#             self.chat_memories[chat_id].append(
#                 HumanMessage(content=content)
#             )
#         elif role == 'ai':
#             self.chat_memories[chat_id].append(
#                 AIMessage(content=content)
#             )
        
#         # Save to shared database
#         self.db.save_chatbot_conversation(username, chat_id, self.active_memories[chat_id])
    
#     def get_messages(self, chat_id: str, last_n: int = None) -> List[Dict]:
#         """Get messages for a chat"""
#         if chat_id not in self.active_memories:
#             return []
        
#         messages = self.active_memories[chat_id]['messages']
#         if last_n:
#             return messages[-last_n:]
#         return messages
    
#     def set_title(self, chat_id: str, username: str, title: str):
#         """Set conversation title"""
#         if chat_id not in self.active_memories:
#             self.load_conversation(chat_id, username)
        
#         if chat_id in self.active_memories:
#             self.active_memories[chat_id]['title'] = title
#             self.db.save_chatbot_conversation(username, chat_id, self.active_memories[chat_id])
    
#     def get_title(self, chat_id: str) -> Optional[str]:
#         """Get conversation title"""
#         if chat_id in self.active_memories:
#             return self.active_memories[chat_id]['title']
#         return None
    
#     def set_preferences(self, chat_id: str, username: str, preferences: dict):
#         """Set user preferences"""
#         if chat_id not in self.active_memories:
#             self.load_conversation(chat_id, username)
        
#         if chat_id in self.active_memories:
#             self.active_memories[chat_id]['preferences'].update(preferences)
#             self.db.save_chatbot_conversation(username, chat_id, self.active_memories[chat_id])
    
#     def get_preferences(self, chat_id: str) -> dict:
#         """Get user preferences"""
#         if chat_id in self.active_memories:
#             return self.active_memories[chat_id]['preferences']
#         return {}
    
#     def get_memory_context(self, chat_id: str, max_messages: int = 15) -> List[BaseMessage]:
#         """Get memory context (last N messages)"""
#         if chat_id in self.chat_memories:
#             all_messages = self.chat_memories[chat_id]
#             return all_messages[-max_messages:] if len(all_messages) > max_messages else all_messages
#         return []

# # ============================
# # Enhanced Academic Chatbot with Personalization
# # ============================

# class PersonalizedAcademicChatbot:
#     """Academic chatbot with personalization, friend-like conversations, and enhanced features"""
    
#     def __init__(self, openai_api_key: str, storage_dir: str = None, model_name: str = "gpt-4o-mini"):
#         self.openai_api_key = openai_api_key
#         openai.api_key = openai_api_key
        
#         # Initialize database (external API - no local storage)
#         self.db = SharedDatabase(storage_dir)
        
#         # Personalization integration
#         self.personalization = PersonalizationIntegration()
        
#         # Single LLM for all operations
#         self.llm = ChatOpenAI(
#             model=model_name,
#             temperature=0.7,
#             max_tokens=1000,
#             api_key=openai_api_key
#         )
        
#         # Enhanced Memory manager
#         self.memory_manager = ConversationMemoryManager(self.db)
        
#         # Setup enhanced chains
#         self._setup_unified_chain()
#         self._setup_intent_classifier()
#         self._setup_preference_extraction()
    
#     def _setup_unified_chain(self):
#         """Setup single unified conversational chain - friend-like, with personalization"""
#         unified_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are Alex, a warm and friendly academic companion. You chat naturally like a supportive friend who genuinely cares.

# 🎯 YOUR PERSONALITY:
# - Talk like a friend, not a formal assistant
# - Be warm, encouraging, and relatable
# - DON'T bombard with questions - just flow naturally
# - Remember everything from the conversation
# - Respond directly to what the user asks
# - Adapt your style based on the user's personality and preferences

# 💬 CONVERSATION STYLE:
# - If someone says "I want to study astrophysics" → Be excited! Share encouragement, maybe mention it's fascinating, and naturally weave in that you can help find colleges if they want
# - If they ask for college recommendations → Jump right in with specific suggestions based on what you know
# - If they ask follow-up questions about colleges you mentioned → Reference them naturally like "Oh yeah, IIT Delhi that I mentioned earlier..."
# - For general questions → Just answer them warmly and directly
# - For resume questions → Reference their actual resume data and provide personalized, specific feedback based on their strengths and areas for improvement

# 🚫 WHAT NOT TO DO:
# - DON'T ask "Are you looking for college recommendations or information?" - just respond naturally
# - DON'T list multiple options like "I can help you with: 1. 2. 3." unless explicitly asked
# - DON'T be overly formal or robotic
# - DON'T ask obvious questions - if they say they want to study something, they probably want help with it
# - DON'T give generic resume advice - use their actual resume data

# ✅ WHAT TO DO:
# - Be conversational and natural
# - Show enthusiasm about their goals
# - Offer help smoothly without being pushy
# - If college data is in the context, integrate it naturally
# - Remember and reference previous parts of the conversation
# - Be encouraging and supportive
# - Use personalization data when available to tailor your responses
# - When discussing resumes, reference their specific strengths and areas for improvement

# CONTEXT AWARENESS:
# - You maintain full memory of the conversation
# - If you recommended colleges earlier, you can discuss them
# - If they mentioned preferences before, you remember them
# - Be naturally conversational - like texting with a knowledgeable friend
# - Use personalization context to adapt your communication style

# PERSONALIZATION CONTEXT (if available):
# {personalization_context}

# Remember: You're a friend who happens to know a lot about academics and colleges, not a Q&A machine!"""),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ])
        
#         # Create a runnable that gets chat history and personalization
#         def get_chat_history_and_context(input_dict: dict) -> dict:
#             chat_id = input_dict.get("chat_id", "default")
#             username = input_dict.get("username", "unknown")
#             chat_history = self.memory_manager.get_memory_context(chat_id, max_messages=15)
            
#             # Get personalization context
#             personalization_context = self.personalization.build_personalization_context(username)
            
#             return {
#                 "chat_history": chat_history,
#                 "input": input_dict.get("input", ""),
#                 "personalization_context": personalization_context
#             }
        
#         self.unified_chain = (
#             RunnableLambda(get_chat_history_and_context)
#             | unified_prompt
#             | self.llm
#             | StrOutputParser()
#         )
    
#     def _setup_intent_classifier(self):
#         """Setup intent classification to determine if user wants college recommendations"""
#         intent_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are an intent classifier. Analyze if the user is EXPLICITLY asking for college recommendations.

# RETURN "YES" ONLY IF:
# 1. User explicitly asks for college suggestions/recommendations/list
# 2. User asks "which colleges should I consider" or similar direct questions
# 3. User asks to "show me colleges" or "tell me about colleges for X"
# 4. User asks "where can I study X" expecting a list of institutions

# RETURN "NO" IF:
# 1. User is just talking about their interests ("I want to study physics")
# 2. User is asking general information about a field/course
# 3. User is greeting or having general conversation
# 4. User is asking follow-up questions about already mentioned colleges (they already have recommendations)
# 5. User is asking about admission process, eligibility, etc. without asking for new colleges

# Be strict - only return YES when user clearly wants a list of college recommendations.

# Answer with just one word: YES or NO"""),
#             ("human", "Message: {message}\nContext: {context}")
#         ])
        
#         self.intent_chain = intent_prompt | self.llm | StrOutputParser()
    
#     def _setup_preference_extraction(self):
#         """Setup preference extraction"""
#         self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        
#         extraction_prompt = ChatPromptTemplate.from_messages([
#             ("system", """Extract user preferences for college search from the conversation.

# Conversation History:
# {conversation_history}

# Current Message:
# {current_message}

# Extract whatever preferences you can find. If nothing specific is mentioned, return null values.

# {format_instructions}

# Extract preferences as JSON."""),
#             ("human", "Extract preferences from the conversation above.")
#         ])
        
#         self.preference_chain = (
#             extraction_prompt.partial(
#                 format_instructions=self.preference_parser.get_format_instructions()
#             )
#             | self.llm
#             | self.preference_parser
#         )
    
#     def _detect_resume_question(self, message: str) -> bool:
#         """Detect if user is asking about their resume"""
#         resume_keywords = [
#             'resume', 'cv', 'my application', 'job application',
#             'my profile', 'career', 'how am i doing', 'my performance',
#             'resume score', 'resume analysis', 'resume feedback',
#             'my resume', 'check my resume', 'review my resume',
#             'how is my resume', 'what do you think of my resume',
#             'resume review', 'resume suggestions', 'improve my resume',
#             'resume help', 'resume advice', 'my resume feedback'
#         ]
#         message_lower = message.lower()
#         return any(keyword in message_lower for keyword in resume_keywords)
    
#     def _get_resume_insights_context(self, username: str) -> str:
#         """Get detailed resume insights for personalization"""
#         try:
#             # Get resume insights from personalization module
#             profile = self.personalization.get_user_profile(username)
            
#             if not profile or not profile.get("resume_insights"):
#                 return ""
            
#             resume_insights = profile.get("resume_insights", {})
            
#             if resume_insights.get("total_analyses", 0) == 0:
#                 return ""
            
#             context_parts = ["\n=== USER'S RESUME INSIGHTS ==="]
            
#             # Basic stats
#             context_parts.append(f"📊 Resume Performance:")
#             context_parts.append(f"   - Average Score: {resume_insights.get('average_score', 0)}%")
#             context_parts.append(f"   - Latest Score: {resume_insights.get('latest_score', 0)}%")
#             context_parts.append(f"   - Trend: {resume_insights.get('improvement_trend', 'stable')}")
            
#             # Target roles
#             target_roles = resume_insights.get('target_roles', [])
#             if target_roles:
#                 context_parts.append(f"🎯 Target Roles: {', '.join(target_roles)}")
            
#             # Common strengths - THIS IS WHAT THE USER IS GOOD AT
#             strengths = resume_insights.get('common_strengths', [])
#             if strengths:
#                 context_parts.append(f"💪 Key Strengths (what user excels at):")
#                 for strength in strengths[:3]:
#                     context_parts.append(f"   - {strength}")
            
#             # Common weaknesses - AREAS WHERE USER NEEDS HELP
#             weaknesses = resume_insights.get('common_weaknesses', [])
#             if weaknesses:
#                 context_parts.append(f"📝 Areas for Improvement:")
#                 for weakness in weaknesses[:3]:
#                     context_parts.append(f"   - {weakness}")
            
#             # Experience level
#             exp_levels = resume_insights.get('experience_levels', [])
#             if exp_levels:
#                 most_common_exp = max(set(exp_levels), key=exp_levels.count) if exp_levels else ""
#                 context_parts.append(f"👔 Experience Level: {most_common_exp}")
            
#             context_parts.append("=== END RESUME INSIGHTS ===\n")
            
#             return "\n".join(context_parts)
            
#         except Exception as e:
#             logger.error(f"Error getting resume insights: {e}")
#             return ""
    
#     def _get_detailed_resume_analysis(self, username: str) -> Dict[str, Any]:
#         """Get detailed resume analysis for the latest resume"""
#         try:
#             # Get the latest resume analysis
#             analyses = self.db.get_user_resume_analyses(username)
            
#             if not analyses:
#                 return {}
            
#             latest = analyses[0]  # Most recent
#             analysis_result = latest.get("analysis_result", {})
            
#             # Extract detailed information
#             detailed_info = {
#                 "overall_score": latest.get("overall_score", 0),
#                 "target_role": latest.get("target_role", ""),
#                 "strengths": [],
#                 "weaknesses": [],
#                 "improvement_plan": {},
#                 "job_market_analysis": {}
#             }
            
#             # Get strengths with detailed analysis
#             strengths_analysis = analysis_result.get("strengths_analysis", [])
#             for strength in strengths_analysis:
#                 detailed_info["strengths"].append({
#                     "strength": strength.get("strength", ""),
#                     "why_strong": strength.get("why_its_strong", ""),
#                     "evidence": strength.get("evidence", "")
#                 })
            
#             # Get weaknesses with specific fixes
#             weaknesses_analysis = analysis_result.get("weaknesses_analysis", [])
#             for weakness in weaknesses_analysis:
#                 detailed_info["weaknesses"].append({
#                     "weakness": weakness.get("weakness", ""),
#                     "priority": weakness.get("fix_priority", ""),
#                     "specific_fix": weakness.get("specific_fix", ""),
#                     "timeline": weakness.get("timeline", "")
#                 })
            
#             # Get improvement plan
#             improvement_plan = analysis_result.get("improvement_plan", {})
#             detailed_info["improvement_plan"] = {
#                 "critical": improvement_plan.get("critical", []),
#                 "high": improvement_plan.get("high", []),
#                 "medium": improvement_plan.get("medium", [])
#             }
            
#             # Get job market analysis
#             job_market = analysis_result.get("job_market_analysis", {})
#             detailed_info["job_market_analysis"] = {
#                 "role_compatibility": job_market.get("role_compatibility", ""),
#                 "market_positioning": job_market.get("market_positioning", ""),
#                 "skill_development": job_market.get("skill_development", "")
#             }
            
#             return detailed_info
            
#         except Exception as e:
#             logger.error(f"Error getting detailed resume analysis: {e}")
#             return {}
    
#     def get_personalized_resume_feedback(self, username: str) -> str:
#         """Get personalized feedback based on resume analysis"""
#         try:
#             analyses = self.db.get_user_resume_analyses(username)
            
#             if not analyses:
#                 return "I notice you haven't uploaded your resume for analysis yet. Would you like me to guide you through the Resume Analyzer? It can help identify your strengths and areas for improvement!"
            
#             latest = analyses[0]  # Most recent
#             score = latest.get("overall_score", 0)
#             strengths = latest.get("strengths", [])
#             weaknesses = latest.get("weaknesses", [])
#             target_role = latest.get("target_role", "your target role")
            
#             # Get detailed analysis for deeper insights
#             detailed = self._get_detailed_resume_analysis(username)
            
#             # Build personalized response
#             response = f"Hey! 👋 I've looked at your resume analysis. Here's my personalized feedback:\n\n"
            
#             # Overall score with interpretation
#             if score >= 80:
#                 response += f"✨ **Great news!** Your resume scored **{score}%**, which is excellent! You're in a strong position for {target_role} roles.\n\n"
#             elif score >= 70:
#                 response += f"👍 **Good progress!** Your resume scored **{score}%**. You're on the right track for {target_role} positions.\n\n"
#             elif score >= 60:
#                 response += f"📝 Your resume scored **{score}%**. With some improvements, you'll be in great shape for {target_role} roles.\n\n"
#             else:
#                 response += f"📊 Your resume scored **{score}%**. Don't worry - this gives us a clear roadmap for improvement!\n\n"
            
#             # Strengths section - PERSONALIZED
#             if detailed.get("strengths"):
#                 response += "💪 **What You're Doing Well:**\n"
#                 for i, strength_data in enumerate(detailed["strengths"][:3]):
#                     response += f"• **{strength_data['strength']}** - {strength_data['why_strong'][:100]}...\n"
#                 response += "\n"
#             elif strengths:
#                 response += "💪 **Your Key Strengths:**\n"
#                 for strength in strengths[:3]:
#                     response += f"• {strength}\n"
#                 response += "\n"
            
#             # Weaknesses with SPECIFIC FIXES
#             if detailed.get("weaknesses"):
#                 response += "🔧 **Areas to Work On (with specific fixes):**\n"
#                 for weakness_data in detailed["weaknesses"][:3]:
#                     priority = weakness_data.get("priority", "MEDIUM")
#                     weakness = weakness_data.get("weakness", "")
#                     fix = weakness_data.get("specific_fix", "")
                    
#                     priority_emoji = "🔴" if priority == "CRITICAL" else "🟡" if priority == "HIGH" else "🟢"
#                     response += f"{priority_emoji} **{weakness}** ({priority} priority)\n"
#                     if fix:
#                         response += f"   → **Suggestion**: {fix[:150]}\n"
                    
#                     timeline = weakness_data.get("timeline", "")
#                     if timeline:
#                         response += f"   → **Timeline**: {timeline}\n"
#                 response += "\n"
#             elif weaknesses:
#                 response += "🔧 **Areas to Work On:**\n"
#                 for weakness in weaknesses[:3]:
#                     response += f"• {weakness}\n"
#                 response += "\n"
            
#             # Improvement plan - ACTIONABLE STEPS
#             improvement_plan = detailed.get("improvement_plan", {})
#             if improvement_plan.get("critical") or improvement_plan.get("high"):
#                 response += "📋 **Recommended Action Plan:**\n"
                
#                 for item in improvement_plan.get("critical", [])[:2]:
#                     response += f"🔴 **Critical**: {item}\n"
#                 for item in improvement_plan.get("high", [])[:2]:
#                     response += f"🟡 **High Priority**: {item}\n"
#                 for item in improvement_plan.get("medium", [])[:2]:
#                     response += f"🟢 **Medium Priority**: {item}\n"
#                 response += "\n"
            
#             # Career guidance based on resume
#             job_market = detailed.get("job_market_analysis", {})
#             if job_market.get("role_compatibility") or job_market.get("market_positioning"):
#                 response += "🎯 **Career Insights Just for You:**\n"
#                 if job_market.get("role_compatibility"):
#                     response += f"• {job_market['role_compatibility']}\n"
#                 if job_market.get("market_positioning"):
#                     response += f"• {job_market['market_positioning']}\n"
#                 if job_market.get("skill_development"):
#                     response += f"• **Focus on**: {job_market['skill_development']}\n"
            
#             # Encouraging close
#             response += "\n✨ Want me to help you with any specific section or suggest colleges that match your profile?"
            
#             return response
            
#         except Exception as e:
#             logger.error(f"Error generating personalized feedback: {e}")
#             return "I'd love to give you personalized feedback on your resume, but I'm having trouble accessing the analysis right now. Could you try uploading your resume again through the Resume Analyzer?"
    
#     def should_get_college_recommendations(self, message: str, chat_id: str) -> bool:
#         """Determine if we should fetch college recommendations using LLM intent classification"""
#         try:
#             # Get recent conversation context
#             recent_messages = self.memory_manager.get_messages(chat_id, last_n=5)
#             context = " | ".join([f"{msg['role']}: {msg['content'][:100]}" for msg in recent_messages[-3:]])
            
#             # Use LLM to classify intent
#             result = self.intent_chain.invoke({
#                 "message": message,
#                 "context": context
#             })
            
#             intent = result.strip().upper()
#             logger.info(f"Intent classification: {intent} for message: '{message[:50]}...'")
            
#             return intent == "YES"
            
#         except Exception as e:
#             logger.error(f"Error in intent classification: {e}")
#             # Fallback to simple keyword matching if LLM fails
#             message_lower = message.lower().strip()
#             fallback_indicators = [
#                 'recommend college', 'suggest college', 'which college should',
#                 'show me college', 'list of college', 'colleges for',
#                 'where should i study', 'where can i study', 'best college for'
#             ]
#             return any(indicator in message_lower for indicator in fallback_indicators)
    
#     def extract_preferences(self, chat_id: str, username: str, current_message: str) -> UserPreferences:
#         """Extract user preferences using LLM"""
#         try:
#             messages = self.memory_manager.get_messages(chat_id, last_n=10)
#             conversation_history = "\n".join([
#                 f"{msg['role'].title()}: {msg['content']}" for msg in messages
#             ])
            
#             preferences = self.preference_chain.invoke({
#                 "conversation_history": conversation_history,
#                 "current_message": current_message
#             })
            
#             # Save preferences to memory
#             if any(value for value in preferences.dict().values()):
#                 self.memory_manager.set_preferences(chat_id, username, preferences.dict())
            
#             return preferences
                
#         except Exception as e:
#             logger.error(f"Error extracting preferences: {e}")
#             prev_prefs = self.memory_manager.get_preferences(chat_id)
#             if prev_prefs:
#                 return UserPreferences(**prev_prefs)
#             return UserPreferences()
    
#     def get_openai_recommendations(self, preferences: UserPreferences, chat_history: str) -> List[Dict]:
#         """Get college recommendations from OpenAI with context awareness"""
#         try:
#             pref_parts = []
            
#             if preferences.specific_institution_type:
#                 pref_parts.append(f"Institution type: {preferences.specific_institution_type}")
#             if preferences.location:
#                 pref_parts.append(f"Location: {preferences.location}")
#             if preferences.state:
#                 pref_parts.append(f"State: {preferences.state}")
#             if preferences.course_type:
#                 pref_parts.append(f"Course type: {preferences.course_type}")
#             if preferences.specific_course:
#                 pref_parts.append(f"Specific course: {preferences.specific_course}")
#             if preferences.college_type:
#                 pref_parts.append(f"College type: {preferences.college_type}")
#             if preferences.budget_range:
#                 pref_parts.append(f"Budget: {preferences.budget_range}")
            
#             # Build comprehensive prompt
#             if pref_parts:
#                 preference_text = ", ".join(pref_parts)
#                 prompt = f"""Based on these preferences: {preference_text}

# Conversation context:
# {chat_history[-500:]}

# Recommend 5 best colleges in India that match these criteria."""
#             else:
#                 prompt = f"""Based on this conversation:
# {chat_history[-500:]}

# Recommend 5 diverse, well-known colleges in India that would be relevant."""
            
#             prompt += """

# Return as JSON array with this exact structure:
# [
#     {
#         "name": "Full College Name",
#         "location": "City, State",
#         "type": "Government/Private/Deemed",
#         "courses": "Main courses offered (be specific)",
#         "features": "Key highlights and why it's recommended",
#         "website": "Official website URL if known, otherwise 'Visit official website'",
#         "admission": "Brief admission process info",
#         "fees": "Approximate annual fee range"
#     }
# ]

# Return ONLY the JSON array, no additional text."""
            
#             response = openai.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.5,
#                 max_tokens=2000
#             )
            
#             result = response.choices[0].message.content.strip()
            
#             try:
#                 colleges = json.loads(result)
#                 return colleges[:5]
#             except json.JSONDecodeError:
#                 json_match = re.search(r'\[.*\]', result, re.DOTALL)
#                 if json_match:
#                     colleges = json.loads(json_match.group())
#                     return colleges[:5]
#                 return []
                
#         except Exception as e:
#             logger.error(f"Error getting OpenAI recommendations: {e}")
#             return []
    
#     def convert_openai_college_to_json(self, college_data: Dict) -> Optional[CollegeRecommendation]:
#         """Convert OpenAI college to standardized JSON format"""
#         try:
#             return CollegeRecommendation(
#                 id=str(uuid.uuid4()),
#                 name=college_data.get('name', 'N/A'),
#                 location=college_data.get('location', 'N/A'),
#                 type=college_data.get('type', 'N/A'),
#                 courses_offered=college_data.get('courses', 'N/A'),
#                 website=college_data.get('website', 'Visit official website for details'),
#                 admission_process=college_data.get('admission', 'Check official website'),
#                 approximate_fees=college_data.get('fees', 'Contact institution for fee details'),
#                 notable_features=college_data.get('features', 'Quality education institution'),
#                 source="openai_knowledge"
#             )
            
#         except Exception as e:
#             logger.error(f"Error converting OpenAI college: {e}")
#             return None
    
#     def format_college_context(self, colleges: List[Dict]) -> str:
#         """Format college information as context for the LLM"""
#         if not colleges:
#             return ""
        
#         context_parts = ["\n[COLLEGE RECOMMENDATIONS AVAILABLE:"]
        
#         for i, college in enumerate(colleges, 1):
#             context_parts.append(f"""
# {i}. {college.get('name', 'N/A')} ({college.get('location', 'N/A')})
#    Type: {college.get('type', 'N/A')}
#    Courses: {college.get('courses', 'N/A')}
#    Features: {college.get('features', 'N/A')}
#    Fees: {college.get('fees', 'N/A')}
#    Website: {college.get('website', 'N/A')}
# """)
        
#         context_parts.append("]")
#         return "\n".join(context_parts)
    
#     def generate_conversation_title(self, message: str, chat_id: str) -> str:
#         """Generate conversation title"""
#         try:
#             messages = self.memory_manager.get_messages(chat_id, last_n=3)
#             context = " ".join([msg['content'][:100] for msg in messages])
            
#             title_prompt = ChatPromptTemplate.from_messages([
#                 ("system", "Generate a 3-8 word title for a conversation."),
#                 ("human", f"Message: {message[:200]}\nContext: {context[:300]}\nTitle:")
#             ])
            
#             title_chain = title_prompt | self.llm | StrOutputParser()
#             title = title_chain.invoke({})
            
#             title = title.strip().replace('"', '').replace("'", "")
#             if len(title) > 50:
#                 title = title[:47] + "..."
            
#             return title if title else "Academic Discussion"
            
#         except Exception as e:
#             logger.error(f"Error generating title: {e}")
#             return "Academic Conversation"
    
#     def get_response(self, message: str, chat_id: str, username: str) -> Dict[str, Any]:
#         """Main unified processing function - conversational, personalized, and context-aware"""
#         timestamp = datetime.now().isoformat()
        
#         # Ensure user exists in database
#         self.db.get_or_create_user(username)
        
#         # Load conversation if exists
#         self.memory_manager.load_conversation(chat_id, username)
        
#         # Save user message
#         self.memory_manager.add_message(chat_id, username, 'human', message, False)
        
#         # Generate or retrieve conversation title
#         existing_title = self.memory_manager.get_title(chat_id)
#         conversation_title = existing_title
        
#         if not existing_title and len(message.strip()) > 10:
#             conversation_title = self.generate_conversation_title(message, chat_id)
#             self.memory_manager.set_title(chat_id, username, conversation_title)
#         elif not existing_title:
#             conversation_title = "New Conversation"
        
#         # Check if asking about resume
#         if self._detect_resume_question(message):
#             logger.info(f"🎯 Resume question detected for {username}")
            
#             # Get personalized resume feedback
#             resume_feedback = self.get_personalized_resume_feedback(username)
            
#             # Also get resume insights for context
#             resume_insights_context = self._get_resume_insights_context(username)
            
#             # Combine with personalization
#             enhanced_message = f"{message}\n\nContext: The user is asking about their resume. Here's what I know about them:\n{resume_insights_context}"
            
#             # Process through unified chain for natural response
#             try:
#                 response = self.unified_chain.invoke({
#                     "input": enhanced_message,
#                     "chat_id": chat_id,
#                     "username": username
#                 })
#             except Exception as e:
#                 logger.error(f"Unified chain failed, using fallback: {e}")
#                 response = resume_feedback
            
#             # Save AI response
#             self.memory_manager.add_message(chat_id, username, 'ai', response, False)
            
#             return {
#                 "response": response,
#                 "is_recommendation": False,
#                 "timestamp": timestamp,
#                 "conversation_title": conversation_title,
#                 "recommendations": [],
#                 "personalized": True
#             }
        
#         # Check if we should fetch college recommendations
#         should_recommend = self.should_get_college_recommendations(message, chat_id)
        
#         logger.info(f"🎯 Recommendation triggered: {should_recommend}")
        
#         # Prepare input for unified chain
#         enhanced_message = message
#         recommendations_data = []
        
#         # Check personalization availability
#         profile = self.personalization.get_user_profile(username)
#         has_personalization = bool(profile and profile.get("data_available", False))
        
#         # If recommendations needed, add college context
#         if should_recommend:
#             try:
#                 logger.info("📚 Fetching college recommendations...")
                
#                 # Extract preferences
#                 preferences = self.extract_preferences(chat_id, username, message)
#                 logger.info(f"Extracted preferences: {preferences.dict()}")
                
#                 # Get conversation history for context
#                 messages = self.memory_manager.get_messages(chat_id)
#                 chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
#                 # Get recommendations from OpenAI
#                 openai_colleges = self.get_openai_recommendations(preferences, chat_history)
                
#                 # Convert to standardized format
#                 for college in openai_colleges:
#                     json_rec = self.convert_openai_college_to_json(college)
#                     if json_rec:
#                         recommendations_data.append(json_rec)
                
#                 # Add context to message
#                 if recommendations_data:
#                     college_context = self.format_college_context(openai_colleges)
#                     enhanced_message = f"{message}\n\n{college_context}"
#                     logger.info(f"✅ Added {len(recommendations_data)} college recommendations to context")
                    
#             except Exception as e:
#                 logger.error(f"Error fetching recommendations: {e}")
        
#         # Process through unified chain
#         try:
#             response = self.unified_chain.invoke({
#                 "input": enhanced_message,
#                 "chat_id": chat_id,
#                 "username": username
#             })
            
#             # Save AI response to memory and database
#             self.memory_manager.add_message(chat_id, username, 'ai', response, should_recommend)
            
#             # Trigger profile update occasionally (every 10 messages)
#             if len(self.memory_manager.get_memory_context(chat_id)) % 10 == 0:
#                 self.personalization.trigger_profile_update(username)
            
#             logger.info(f"✅ Response generated successfully (personalized: {has_personalization})")
            
#             return {
#                 "response": response,
#                 "is_recommendation": should_recommend,
#                 "timestamp": timestamp,
#                 "conversation_title": conversation_title,
#                 "recommendations": recommendations_data,
#                 "personalized": has_personalization
#             }
            
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             return {
#                 "response": "I'm having a bit of trouble right now. Could you try asking that again? 😊",
#                 "is_recommendation": False,
#                 "timestamp": timestamp,
#                 "conversation_title": conversation_title,
#                 "recommendations": [],
#                 "personalized": has_personalization
#             }

# # ============================
# # Initialize
# # ============================

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     logger.error("OPENAI_API_KEY not found in environment variables!")
#     raise ValueError("OPENAI_API_KEY is required")

# try:
#     chatbot = PersonalizedAcademicChatbot(OPENAI_API_KEY)
#     logger.info("✅ Enhanced Personalized Academic Chatbot initialized with Shared Database")
#     logger.info("📦 Using LATEST LangChain packages")
# except Exception as e:
#     logger.error(f"❌ Error initializing chatbot: {e}")
#     raise

# # ============================
# # FastAPI Routes
# # ============================

# @app.get("/")
# async def root():
#     return {
#         "message": "AI Academic Chatbot with Personalization & Enhanced Features",
#         "version": "6.0.0",
#         "description": "Friend-like academic chatbot with personalization, smart intent detection, and resume awareness",
#         "features": {
#             "unified_pipeline": "✅",
#             "natural_conversations": "✅",
#             "smart_intent_detection": "✅",
#             "context_awareness": "✅",
#             "friend_like_personality": "✅",
#             "personalization": "✅",
#             "resume_awareness": "✅",
#             "dynamic_profile_updates": "✅",
#             "personalized_resume_feedback": "✅",
#             "communication_style_matching": "✅",
#             "personality_adaptation": "✅",
#             "college_recommendations": "✅",
#             "shared_database": "✅",
#             "user_tracking": "✅",
#             "conversation_memory": "✅"
#         },
#         "enhancements": [
#             "✅ Friend-like conversational style",
#             "✅ Full personalization integration",
#             "✅ Resume-aware responses with specific feedback",
#             "✅ Dynamic profile updates based on interactions",
#             "✅ Personality trait adaptation",
#             "✅ Communication style matching",
#             "✅ Improved intent classification",
#             "✅ Better context awareness",
#             "✅ Enhanced college recommendations",
#             "✅ Unified memory system",
#             "✅ Latest LangChain packages"
#         ]
#     }

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(
#     request: ChatRequest,
#     chat_id: str = Query(..., description="Chat ID managed by backend"),
#     username: str = Query(..., description="Username")
# ):
#     """Enhanced personalized chat endpoint"""
#     if not request.message.strip():
#         raise HTTPException(status_code=400, detail="Message cannot be empty")
    
#     if not username.strip():
#         raise HTTPException(status_code=400, detail="Username cannot be empty")
    
#     if not chat_id.strip():
#         raise HTTPException(status_code=400, detail="Chat ID cannot be empty")
    
#     try:
#         result = chatbot.get_response(
#             message=request.message,
#             chat_id=chat_id,
#             username=username
#         )
#         return ChatResponse(**result)
    
#     except Exception as e:
#         logger.error(f"Chat endpoint error: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/user/{username}/conversations")
# async def get_user_conversations(username: str):
#     """Get all chatbot conversations for a user"""
#     try:
#         conversations = chatbot.db.get_user_chatbot_conversations(username)
#         return {
#             "username": username,
#             "total_conversations": len(conversations),
#             "conversations": conversations
#         }
#     except Exception as e:
#         logger.error(f"Error fetching conversations: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/conversation/{username}/{chat_id}")
# async def get_conversation(username: str, chat_id: str):
#     """Get specific conversation"""
#     try:
#         conversation = chatbot.db.get_chatbot_conversation(username, chat_id)
#         if not conversation:
#             raise HTTPException(status_code=404, detail="Conversation not found")
        
#         # Add memory context if available
#         memory_context = chatbot.memory_manager.get_memory_context(chat_id)
#         conversation["memory_context_count"] = len(memory_context)
        
#         return conversation
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching conversation: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.delete("/conversation/{username}/{chat_id}")
# async def delete_conversation(username: str, chat_id: str):
#     """Delete a conversation"""
#     try:
#         chatbot.db.delete_interaction(username, "chatbot", chat_id)
        
#         # Also clear from memory manager
#         if chat_id in chatbot.memory_manager.active_memories:
#             del chatbot.memory_manager.active_memories[chat_id]
#         if chat_id in chatbot.memory_manager.chat_memories:
#             del chatbot.memory_manager.chat_memories[chat_id]
        
#         return {"message": "Conversation deleted successfully"}
#     except Exception as e:
#         logger.error(f"Error deleting conversation: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/user/{username}/personalization")
# async def get_user_personalization(username: str):
#     """Get user personalization status"""
#     profile = chatbot.personalization.get_user_profile(username)
    
#     if not profile:
#         return {
#             "username": username,
#             "personalization_available": False,
#             "message": "Personalization module not available or user has no data"
#         }
    
#     return {
#         "username": username,
#         "personalization_available": True,
#         "has_resume_data": profile.get("resume_insights", {}).get("total_analyses", 0) > 0,
#         "total_interactions": profile.get("total_interactions", 0),
#         "personality_traits": profile.get("personality_traits", {}),
#         "communication_style": profile.get("communication_style", {}),
#         "resume_insights": profile.get("resume_insights", {}),
#         "topics_of_interest": profile.get("topics_of_interest", []),
#         "professional_interests": profile.get("professional_interests", [])
#     }

# @app.post("/user/{username}/update-personalization")
# async def trigger_personalization_update(username: str):
#     """Manually trigger personalization update"""
#     chatbot.personalization.trigger_profile_update(username)
#     return {"message": f"Personalization update triggered for {username}"}

# @app.get("/health")
# async def health_check():
#     """Health check with comprehensive status"""
#     try:
#         # Check personalization module
#         personalization_status = "connected"
#         try:
#             response = requests.get("http://localhost:8001/health", timeout=2)
#             if response.status_code != 200:
#                 personalization_status = "disconnected"
#         except:
#             personalization_status = "disconnected"
        
#         # Get stats from shared database
#         all_users = chatbot.db.get_all_users()
        
#         # Get memory stats
#         active_conversations = len(chatbot.memory_manager.active_memories)
#         active_memories = len(chatbot.memory_manager.chat_memories)
        
#         return {
#             "status": "healthy",
#             "timestamp": datetime.now().isoformat(),
#             "service": "AI Academic Chatbot with Personalization & Enhanced Features",
#             "version": "6.0.0",
#             "features": {
#                 "unified_pipeline": "✅",
#                 "natural_conversations": "✅",
#                 "smart_intent_detection": "✅",
#                 "context_awareness": "✅",
#                 "friend_like_personality": "✅",
#                 "personalization": "✅",
#                 "resume_awareness": "✅",
#                 "dynamic_profile_updates": "✅",
#                 "communication_style_matching": "✅",
#                 "personality_adaptation": "✅",
#                 "college_recommendations": "✅",
#                 "shared_database": "✅",
#                 "user_tracking": "✅",
#                 "conversation_memory": "✅"
#             },
#             "database": {
#                 "type": "external_api",
#                 "url": EXTERNAL_DB_API_URL,
#                 "total_users": len(all_users)
#             },
#             "memory": {
#                 "active_conversations": active_conversations,
#                 "active_memories": active_memories,
#                 "type": "Shared Database + In-memory context"
#             },
#             "personalization_module": personalization_status,
#             "langchain_version": "Latest (langchain-core, langchain-openai)",
#             "enhancements": [
#                 "✅ Friend-like conversational style",
#                 "✅ Full personalization integration",
#                 "✅ Resume-aware responses with specific feedback",
#                 "✅ Dynamic profile updates based on interactions",
#                 "✅ Personality trait adaptation",
#                 "✅ Communication style matching",
#                 "✅ Improved intent classification",
#                 "✅ Better context awareness",
#                 "✅ Enhanced college recommendations",
#                 "✅ Unified memory system",
#                 "✅ Latest LangChain packages"
#             ]
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# if __name__ == "__main__":
#     print("=" * 70)
#     print("🚀 Starting Enhanced Personalized Academic Chatbot")
#     print("=" * 70)
#     print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
#     print(f"🧠 Personalization: Enabled")
#     print(f"🎯 Version: 6.0.0 - Friend-like Conversations with Personalization")
#     print(f"💬 Features:")
#     print(f"   • Unified Pipeline: ✅")
#     print(f"   • Natural Conversations: ✅")
#     print(f"   • Smart Intent Detection: ✅")
#     print(f"   • Context Awareness: ✅")
#     print(f"   • Friend-like Personality: ✅")
#     print(f"   • Personalization: ✅")
#     print(f"   • Resume Awareness: ✅")
#     print(f"   • Dynamic Profile Updates: ✅")
#     print(f"   • Personalized Resume Feedback: ✅")
#     print(f"   • Communication Style Matching: ✅")
#     print(f"   • Personality Adaptation: ✅")
#     print(f"   • College Recommendations: ✅")
#     print(f"   • Shared Database: ✅")
#     print(f"   • User Tracking: ✅")
#     print(f"   • Conversation Memory: ✅")
#     print(f"🔗 API: http://localhost:8000")
#     print(f"📚 Docs: http://localhost:8000/docs")
#     print("=" * 70)
    
#     uvicorn.run(
#         app,
#         host="127.0.0.1",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )




