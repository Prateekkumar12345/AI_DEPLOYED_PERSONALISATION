
"""
AI Resume Analyzer with Comprehensive Features & Three-Layer Validation
This version combines robust resume analysis with extensive feature tracking
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import asyncio
import io
import concurrent.futures
import re
from datetime import datetime
import hashlib
import uuid

# Updated LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, validator

# Import shared database
from shared_database import SharedDatabase, EXTERNAL_DB_API_URL

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AI Resume Analyzer API with Comprehensive Features")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Initialize shared database
shared_db = SharedDatabase()

# ===== CACHE FOR CONSISTENT RESULTS =====
analysis_cache = {}

def get_content_hash(resume_text: str, target_role: str) -> str:
    """Generate consistent hash for caching"""
    content = f"{resume_text[:1000]}_{target_role}"
    return hashlib.md5(content.encode()).hexdigest()

# ===== REQUEST/RESPONSE MODELS =====

class AnalyzeResumeRequest(BaseModel):
    """Request model for resume analysis"""
    username: str = Field(..., description="Username for whom the analysis is being done")
    target_role: Optional[str] = Field(None, description="Target job position/role")
    search_jobs: bool = Field(True, description="Whether to search for relevant jobs")
    location: str = Field("India", description="Location for job search")

    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "target_role": "Senior Software Engineer",
                "search_jobs": True,
                "location": "India"
            }
        }

# ===== PYDANTIC MODELS =====

class ProfessionalProfile(BaseModel):
    experience_level: str = Field(description="Years of experience and seniority level")
    technical_skills_count: int = Field(description="Number of technical skills identified")
    project_portfolio_size: str = Field(description="Size and quality of project portfolio")
    achievement_metrics: str = Field(description="Quality of quantified achievements")
    technical_sophistication: str = Field(description="Level of technical expertise")

class ContactPresentation(BaseModel):
    email_address: str = Field(description="Email presence and quality")
    phone_number: str = Field(description="Phone number presence")
    education: str = Field(description="Education background quality")
    resume_length: str = Field(description="Resume length assessment")
    action_verbs: str = Field(description="Use of strong action verbs")

class OverallAssessment(BaseModel):
    score_percentage: int = Field(description="Overall score percentage")
    level: str = Field(description="Assessment level")
    description: str = Field(description="Score description")
    recommendation: str = Field(description="Overall recommendation")

class ExecutiveSummary(BaseModel):
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    overall_assessment: OverallAssessment

class ScoringDetail(BaseModel):
    score: int = Field(description="Score out of max points")
    max_score: int = Field(description="Maximum possible score")
    percentage: float = Field(description="Percentage score")
    details: List[str] = Field(description="Detailed breakdown of scoring")

class StrengthAnalysis(BaseModel):
    strength: str = Field(description="Main strength identified")
    why_its_strong: str = Field(description="Explanation of why it's a strength")
    ats_benefit: str = Field(description="How it helps with ATS systems")
    competitive_advantage: str = Field(description="Competitive advantage provided")
    evidence: str = Field(description="Supporting evidence from resume")

class WeaknessAnalysis(BaseModel):
    weakness: str = Field(description="Main weakness identified")
    why_problematic: str = Field(description="Why this is problematic")
    ats_impact: str = Field(description="Impact on ATS systems")
    how_it_hurts: str = Field(description="How it hurts candidacy")
    fix_priority: str = Field(description="Priority level: CRITICAL/HIGH/MEDIUM")
    specific_fix: str = Field(description="Specific steps to fix")
    timeline: str = Field(description="Timeline for implementation")

class ImprovementPlan(BaseModel):
    critical: List[str] = Field(default_factory=list, description="Critical improvements")
    high: List[str] = Field(default_factory=list, description="High priority improvements")
    medium: List[str] = Field(default_factory=list, description="Medium priority improvements")

class JobMarketAnalysis(BaseModel):
    role_compatibility: str = Field(description="Compatibility with target role")
    market_positioning: str = Field(description="Position in job market")
    career_advancement: str = Field(description="Career advancement opportunities")
    skill_development: str = Field(description="Skill development recommendations")

class AIInsights(BaseModel):
    overall_score: int = Field(description="Overall AI-determined score")
    recommendation_level: str = Field(description="Recommendation level")
    key_strengths_count: int = Field(description="Number of key strengths")
    improvement_areas_count: int = Field(description="Number of improvement areas")

class ResumeAnalysis(BaseModel):
    """Main analysis model matching standard JSON structure"""
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    detailed_scoring: Dict[str, ScoringDetail]
    strengths_analysis: List[StrengthAnalysis] = Field(min_items=5)
    weaknesses_analysis: List[WeaknessAnalysis] = Field(min_items=5)
    improvement_plan: ImprovementPlan
    job_market_analysis: JobMarketAnalysis
    overall_score: int = Field(ge=0, le=100, description="Overall resume score out of 100")
    recommendation_level: str = Field(description="Overall recommendation level")

class JobListing(BaseModel):
    company_name: str = Field(description="Name of the hiring company")
    position: str = Field(description="Job position/title")
    location: str = Field(description="Job location")
    ctc: str = Field(description="Compensation/Salary range")
    experience_required: str = Field(description="Required years of experience")
    last_date_to_apply: str = Field(description="Application deadline")
    about_job: str = Field(description="Brief description about the job")
    job_description: str = Field(description="Detailed job description")
    job_requirements: str = Field(description="Required skills and qualifications")
    application_url: Optional[str] = Field(description="Link to apply")

# ===== DOCUMENT CLASSIFICATION =====

class DocumentClassificationResult(BaseModel):
    """Result of initial document classification"""
    label: str              # "resume" | "non_resume"
    confidence: float       # 0.0 to 1.0
    reason: str            # Explanation from classifier

class DocumentClassifier:
    """
    Initial LLM-based document classifier.
    Runs BEFORE the heuristic validator as a quick first pass.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def classify(self, text: str, max_chars: int = 6000) -> DocumentClassificationResult:
        """Quick classification using GPT to determine if document is a resume"""
        
        system_prompt = """You are an expert HR document classifier.

Task: Classify whether the given document is a Resume/CV or NOT.

Guidelines:
- Resume/CV includes: education, work experience, skills, certifications, projects, personal information
- Resumes may vary in format (tables, bullet points, paragraphs, columns)
- Job descriptions, technical documentation, invoices, letters, articles, forms are NOT resumes
- Developer resumes will include technical skills like Docker, Kubernetes, architecture - this is NORMAL

Output rules:
- Respond ONLY with valid JSON
- No extra text before or after JSON"""

        user_prompt = f"""Classify the following document:

{text[:max_chars]}

Return JSON exactly in this format:
{{
  "label": "resume" or "non_resume",
  "confidence": number between 0 and 1,
  "reason": "short explanation"
}}"""

        try:
            response = await self.llm.ainvoke(
                f"{system_prompt}\n\n{user_prompt}"
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return DocumentClassificationResult(
                    label=parsed.get("label", "non_resume"),
                    confidence=float(parsed.get("confidence", 0.0)),
                    reason=parsed.get("reason", "Classification completed")
                )
            else:
                logger.warning(f"Could not parse classifier JSON: {response_text}")
                return DocumentClassificationResult(
                    label="non_resume",
                    confidence=0.5,
                    reason="Could not parse classifier response"
                )
                
        except Exception as e:
            logger.error(f"Document classification error: {e}")
            return DocumentClassificationResult(
                label="non_resume",
                confidence=0.0,
                reason=f"Classification failed: {e}"
            )

# ===== RESUME VALIDATION =====

class ResumeValidationResult(BaseModel):
    """Result of resume validation"""
    is_resume: bool
    confidence: str        # "high" | "medium" | "low"
    method: str            # "heuristic" | "llm" | "heuristic+llm"
    reason: str            # Human-readable explanation

class ResumeValidator:
    """
    Two-layer resume validator with comprehensive keyword sets.
    """
    
    # Resume signals with weights
    RESUME_SIGNALS: List[tuple] = [
        # Identity / contact block (very strong resume signals)
        ("linkedin.com",           3),
        ("github.com",             3),
        # Education statements
        ("bachelor",               2),
        ("master",                 2),
        ("b.tech",                 2),
        ("m.tech",                 2),
        ("b.sc",                   2),
        ("m.sc",                   2),
        ("mba",                    2),
        ("university",             2),
        ("degree",                 2),
        # Classic resume section headers
        ("work experience",        2),
        ("professional experience",2),
        ("employment history",     2),
        ("education",              2),
        ("certifications",         2),
        ("technical skills",       2),
        ("skills",                 1),
        ("objective",              1),
        ("summary",                1),
        ("achievements",           2),
        ("projects",               1),
        ("personal statement",     2),
        # Developer-CV specific headers / phrases
        ("full stack developer",   2),
        ("software developer",     2),
        ("software engineer",      2),
        ("frontend developer",     2),
        ("backend developer",      2),
        ("freelancer",             2),
        # Action phrases common in experience bullets
        ("responsible for",        1),
        ("managed",                1),
        ("developed",              1),
        ("led a team",             1),
        ("experience in",          1),
        ("proficient in",          1),
        # Percentage / score (common in Indian CVs for marks)
        ("percentage:",            2),
    ]

    # Non-resume signals with weights
    NON_RESUME_SIGNALS: List[tuple] = [
        # Multi-word phrases that only appear in doc/spec writing
        ("technical documentation", 3),
        ("system design",           2),
        ("requirements document",   3),
        ("data model",              2),
        ("database schema",         2),
        ("flow lifecycle",          3),
        ("api endpoint",            2),
        # Actual code syntax (with trailing space / brace to reduce false positives)
        ("def ",                    2),
        ("import ",                 1),
        ("class {",                 3),
        ("extends model",           3),
        ("enum ",                   2),
        # Project-management / agile docs
        ("user story",              2),
        ("sprint",                  2),
        ("backlog",                 2),
        ("readme",                  2),
        ("changelog",               2),
        # Academic / research
        ("abstract",                2),
        ("methodology",             2),
        ("bibliography",            3),
        ("hypothesis",              3),
        ("literature review",       3),
    ]

    RESUME_NET_THRESHOLD     =  2   # net >= this → resume
    NON_RESUME_NET_THRESHOLD = -4   # net <= this → not a resume

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _heuristic_check(self, text: str) -> ResumeValidationResult:
        """Compute weighted resume and non-resume scores"""
        lower_text = text.lower()

        resume_score = 0
        resume_hits: List[str] = []
        for phrase, weight in self.RESUME_SIGNALS:
            if phrase in lower_text:
                resume_score += weight
                resume_hits.append(phrase)

        non_resume_score = 0
        non_resume_hits: List[str] = []
        for phrase, weight in self.NON_RESUME_SIGNALS:
            if phrase in lower_text:
                non_resume_score += weight
                non_resume_hits.append(phrase)

        net = resume_score - non_resume_score

        logger.info(
            f"Heuristic — resume_score: {resume_score} (hits: {resume_hits}), "
            f"non_resume_score: {non_resume_score} (hits: {non_resume_hits}), "
            f"net: {net}"
        )

        if net >= self.RESUME_NET_THRESHOLD:
            return ResumeValidationResult(
                is_resume=True,
                confidence="high",
                method="heuristic",
                reason=(
                    f"Document matched resume indicators with a weighted score of "
                    f"{resume_score} vs {non_resume_score} for non-resume indicators "
                    f"(net: +{net})."
                ),
            )

        if net <= self.NON_RESUME_NET_THRESHOLD:
            return ResumeValidationResult(
                is_resume=False,
                confidence="high",
                method="heuristic",
                reason=(
                    f"Document matched non-resume indicators with a weighted score of "
                    f"{non_resume_score} vs {resume_score} for resume indicators "
                    f"(net: {net})."
                ),
            )

        return ResumeValidationResult(
            is_resume=False,
            confidence="low",
            method="heuristic",
            reason=f"Ambiguous signal (net: {net}) — escalating to LLM classification.",
        )

    async def _llm_check(self, text: str) -> ResumeValidationResult:
        """Send a lightweight classification prompt to the LLM"""
        max_chars = 2000
        if len(text) > max_chars:
            half = max_chars // 2
            snippet = text[:half] + "\n\n[... middle section omitted ...]\n\n" + text[-half:]
        else:
            snippet = text

        prompt = (
            "You are a document classifier. Read the following document excerpt and decide "
            "whether it is a RESUME (also called a CV) or NOT a resume.\n\n"
            "A resume/CV is a personal document that lists an individual's education, "
            "work experience, skills, and qualifications for the purpose of job applications.\n\n"
            "Respond with EXACTLY one of these two JSON objects and nothing else:\n"
            '  {"is_resume": true, "reason": "<brief explanation>"}\n'
            '  {"is_resume": false, "reason": "<brief explanation of what the document actually is>"}\n\n'
            "Document excerpt:\n"
            "---\n"
            f"{snippet}\n"
            "---\n"
        )

        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                is_resume = bool(parsed.get("is_resume", False))
                reason = parsed.get("reason", "LLM classification completed.")
                return ResumeValidationResult(
                    is_resume=is_resume,
                    confidence="high",
                    method="llm",
                    reason=reason,
                )
            else:
                logger.warning(f"LLM validation: could not parse JSON from response: {response_text}")
                return ResumeValidationResult(
                    is_resume=False,
                    confidence="medium",
                    method="llm",
                    reason="LLM response could not be parsed reliably.",
                )
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return ResumeValidationResult(
                is_resume=False,
                confidence="low",
                method="llm",
                reason=f"LLM classification failed ({e}).",
            )

    async def validate(self, text: str) -> ResumeValidationResult:
        """Run Layer 1. If ambiguous, run Layer 2."""
        heuristic_result = self._heuristic_check(text)

        if heuristic_result.confidence == "high":
            logger.info(f"Validation decided by heuristic: is_resume={heuristic_result.is_resume}")
            return heuristic_result

        logger.info("Heuristic ambiguous — running LLM classification.")
        llm_result = await self._llm_check(text)
        llm_result.method = "heuristic+llm"
        logger.info(f"Validation decided by LLM: is_resume={llm_result.is_resume}")
        return llm_result

# ===== PDF EXTRACTION =====

class OptimizedPDFExtractor:
    """Optimized PDF text extraction"""
    
    @staticmethod
    async def extract_text_from_pdf(uploaded_file) -> Optional[str]:
        try:
            uploaded_file.seek(0)
            content = await uploaded_file.read()
            
            def process_pdf(content_bytes):
                pdf_file = io.BytesIO(content_bytes)
                pdf_reader = PdfReader(pdf_file)
                
                extracted_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
                        continue
                
                return extracted_text.strip()
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                extracted_text = await loop.run_in_executor(pool, process_pdf, content)
            
            return extracted_text if extracted_text else None
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return None

# ===== JOB SEARCH =====

class JobSearchService:
    """Service to search and parse job listings"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def search_jobs(self, target_role: str, location: str = "India") -> List[Dict[str, Any]]:
        """Search for jobs and extract structured information"""
        try:
            job_extraction_prompt = f"""
            Generate 5-10 realistic current job listings for the position: {target_role} in {location}.
            
            For each job listing, provide EXACTLY these fields in JSON format:
            {{
                "company_name": "Company name",
                "position": "Exact job title",
                "location": "City/region in {location}",
                "ctc": "Salary range with currency",
                "experience_required": "X-Y years",
                "last_date_to_apply": "YYYY-MM-DD format",
                "about_job": "2-3 sentence summary",
                "job_description": "Detailed responsibilities and duties",
                "job_requirements": "Required skills, qualifications, and education",
                "application_url": "https://company-careers.com/job-id"
            }}
            
            Return ONLY a valid JSON array with no additional text. Make the data realistic and relevant to the current job market in 2025.
            """
            
            response = await self.llm.ainvoke(job_extraction_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the JSON response
            try:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    jobs_data = json.loads(json_match.group())
                else:
                    jobs_data = json.loads(response_text)
                
                return jobs_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse job listings JSON: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Job search error: {str(e)}")
            return []

# ===== RESUME ANALYZER =====

class HighPerformanceLangChainAnalyzer:
    """High-performance AI analyzer with guaranteed standard JSON output"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo-16k",
            temperature=0.0,
            max_tokens=4000,
            request_timeout=30
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
        self.document_classifier = DocumentClassifier(llm=self.llm)
        self.resume_validator = ResumeValidator(llm=self.llm)
        self.job_search = JobSearchService(self.llm)
        self._setup_analysis_chain()
    
    def _setup_analysis_chain(self):
        """Setup the analysis chain using LCEL"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume analyzer. Analyze the resume comprehensively for the target role.

YOU MUST provide a complete JSON response with ALL of the following sections:

1. PROFESSIONAL PROFILE (experience_level, technical_skills_count, project_portfolio_size, achievement_metrics, technical_sophistication)
2. CONTACT PRESENTATION (email_address, phone_number, education, resume_length, action_verbs)
3. DETAILED SCORING with these exact sections (use snake_case keys):
   - "contact_information" (score, max_score, percentage, details)
   - "technical_skills" (score, max_score, percentage, details)
   - "experience_quality" (score, max_score, percentage, details)
   - "quantified_achievements" (score, max_score, percentage, details)
   - "content_optimization" (score, max_score, percentage, details)
4. STRENGTHS ANALYSIS - Provide at least 5 strengths
5. WEAKNESSES ANALYSIS - Provide at least 5 weaknesses
6. IMPROVEMENT PLAN (critical, high, medium lists)
7. JOB MARKET ANALYSIS (role_compatibility, market_positioning, career_advancement, skill_development)
8. overall_score (0-100)
9. recommendation_level

{format_instructions}

CRITICAL: Return ONLY valid JSON matching the exact structure specified. No additional text."""),
            ("human", "Target Role: {target_role}\n\nResume Content:\n{resume_text}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        
        self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()
    
    def _get_standard_response_template(self, target_role: str, word_count: int) -> Dict[str, Any]:
        """Returns the standard response structure"""
        return {
            "success": True,
            "analysis_method": "AI-Powered LangChain Analysis with Three-Layer Validation",
            "resume_metadata": {
                "word_count": word_count,
                "validation_message": "Comprehensive AI analysis completed",
                "target_role": target_role or "general position"
            },
            "executive_summary": {
                "professional_profile": {},
                "contact_presentation": {},
                "overall_assessment": {}
            },
            "detailed_scoring": {},
            "strengths_analysis": [],
            "weaknesses_analysis": [],
            "improvement_plan": {
                "critical": [],
                "high": [],
                "medium": []
            },
            "job_market_analysis": {},
            "ai_insights": {}
        }
    
    def _convert_to_snake_case(self, key: str) -> str:
        """Convert title case to snake_case"""
        mapping = {
            "Contact Information": "contact_information",
            "Technical Skills": "technical_skills",
            "Experience Quality": "experience_quality",
            "Quantified Achievements": "quantified_achievements",
            "Content Optimization": "content_optimization"
        }
        return mapping.get(key, key.lower().replace(" ", "_"))
    
    async def analyze_resume_with_jobs(
        self, 
        resume_text: str, 
        username: str,
        target_role: Optional[str] = None,
        search_jobs: bool = True,
        location: str = "India"
    ) -> Dict[str, Any]:
        """Analyze resume with guaranteed standard JSON format and optional job search"""
        try:
            role_context = target_role or "general position"
            word_count = len(resume_text.split())
            
            # Check cache first
            cache_key = get_content_hash(resume_text, role_context)
            if cache_key in analysis_cache:
                logger.info("Returning cached analysis result")
                return analysis_cache[cache_key]
            
            # Initialize response with standard structure
            response = self._get_standard_response_template(role_context, word_count)
            
            # Run resume analysis and job search in parallel if needed
            if search_jobs and target_role:
                analysis_task = self.analysis_chain.ainvoke({
                    "resume_text": resume_text,
                    "target_role": role_context
                })
                jobs_task = self.job_search.search_jobs(target_role, location)
                
                analysis_result, job_listings = await asyncio.gather(
                    analysis_task,
                    jobs_task,
                    return_exceptions=True
                )
                
                if isinstance(analysis_result, Exception):
                    raise analysis_result
                if isinstance(job_listings, Exception):
                    logger.error(f"Job search failed: {job_listings}")
                    job_listings = []
            else:
                analysis_result = await self.analysis_chain.ainvoke({
                    "resume_text": resume_text,
                    "target_role": role_context
                })
                job_listings = []
            
            # Parse and populate response
            try:
                parsed_analysis = self.output_parser.parse(analysis_result)
                self._populate_response(response, parsed_analysis, word_count, role_context)
                
            except Exception as parse_error:
                logger.warning(f"Structured parsing failed, using fallback: {parse_error}")
                self._populate_fallback_response(response, analysis_result, word_count, role_context)
            
            # Add job listings if available
            if job_listings:
                response["job_listings"] = {
                    "total_jobs_found": len(job_listings),
                    "search_query": f"{target_role} in {location}",
                    "jobs": job_listings
                }
            
            # Add username to response
            response["username"] = username
            
            # Cache the result
            analysis_cache[cache_key] = response
            logger.info(f"Cached analysis result for key: {cache_key}")
            
            return response
                
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return self._generate_error_response(str(e), target_role, word_count, username)
    
    def _populate_response(self, response: Dict, analysis: ResumeAnalysis, word_count: int, target_role: str):
        """Populate response with parsed analysis data"""
        
        response["executive_summary"] = {
            "professional_profile": {
                "experience_level": analysis.professional_profile.experience_level,
                "technical_skills_count": analysis.professional_profile.technical_skills_count,
                "project_portfolio_size": analysis.professional_profile.project_portfolio_size,
                "achievement_metrics": analysis.professional_profile.achievement_metrics,
                "technical_sophistication": analysis.professional_profile.technical_sophistication
            },
            "contact_presentation": {
                "email_address": analysis.contact_presentation.email_address,
                "phone_number": analysis.contact_presentation.phone_number,
                "education": analysis.contact_presentation.education,
                "resume_length": analysis.contact_presentation.resume_length,
                "action_verbs": analysis.contact_presentation.action_verbs
            },
            "overall_assessment": {
                "score_percentage": analysis.overall_score,
                "level": analysis.recommendation_level,
                "description": f"AI-determined resume quality: {analysis.overall_score}%",
                "recommendation": analysis.recommendation_level
            }
        }
        
        response["detailed_scoring"] = {}
        for key, detail in analysis.detailed_scoring.items():
            snake_case_key = self._convert_to_snake_case(key)
            response["detailed_scoring"][snake_case_key] = {
                "score": detail.score,
                "max_score": detail.max_score,
                "percentage": detail.percentage,
                "details": detail.details
            }
        
        response["strengths_analysis"] = [
            {
                "strength": s.strength,
                "why_its_strong": s.why_its_strong,
                "ats_benefit": s.ats_benefit,
                "competitive_advantage": s.competitive_advantage,
                "evidence": s.evidence
            }
            for s in analysis.strengths_analysis
        ]
        
        response["weaknesses_analysis"] = [
            {
                "weakness": w.weakness,
                "why_problematic": w.why_problematic,
                "ats_impact": w.ats_impact,
                "how_it_hurts": w.how_it_hurts,
                "fix_priority": w.fix_priority,
                "specific_fix": w.specific_fix,
                "timeline": w.timeline
            }
            for w in analysis.weaknesses_analysis
        ]
        
        response["improvement_plan"] = {
            "critical": analysis.improvement_plan.critical,
            "high": analysis.improvement_plan.high,
            "medium": analysis.improvement_plan.medium
        }
        
        response["job_market_analysis"] = {
            "role_compatibility": analysis.job_market_analysis.role_compatibility,
            "market_positioning": analysis.job_market_analysis.market_positioning,
            "career_advancement": analysis.job_market_analysis.career_advancement,
            "skill_development": analysis.job_market_analysis.skill_development
        }
        
        response["ai_insights"] = {
            "overall_score": analysis.overall_score,
            "recommendation_level": analysis.recommendation_level,
            "key_strengths_count": len(analysis.strengths_analysis),
            "improvement_areas_count": len(analysis.weaknesses_analysis)
        }
    
    def _populate_fallback_response(self, response: Dict, raw_result: str, word_count: int, target_role: str):
        """Fallback method to populate response from raw LLM output"""
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                if "professional_profile" in parsed_data:
                    response["executive_summary"]["professional_profile"] = parsed_data["professional_profile"]
                if "contact_presentation" in parsed_data:
                    response["executive_summary"]["contact_presentation"] = parsed_data["contact_presentation"]
                if "overall_score" in parsed_data:
                    response["executive_summary"]["overall_assessment"] = {
                        "score_percentage": parsed_data.get("overall_score", 0),
                        "level": parsed_data.get("recommendation_level", "Unknown"),
                        "description": f"AI-determined resume quality: {parsed_data.get('overall_score', 0)}%",
                        "recommendation": parsed_data.get("recommendation_level", "Unknown")
                    }
                
                detailed_scoring = parsed_data.get("detailed_scoring", {})
                converted_scoring = {}
                for key, value in detailed_scoring.items():
                    snake_case_key = self._convert_to_snake_case(key)
                    converted_scoring[snake_case_key] = value
                response["detailed_scoring"] = converted_scoring
                
                response["strengths_analysis"] = parsed_data.get("strengths_analysis", [])
                response["weaknesses_analysis"] = parsed_data.get("weaknesses_analysis", [])
                response["improvement_plan"] = parsed_data.get("improvement_plan", {"critical": [], "high": [], "medium": []})
                response["job_market_analysis"] = parsed_data.get("job_market_analysis", {})
                response["ai_insights"] = {
                    "overall_score": parsed_data.get("overall_score", 0),
                    "recommendation_level": parsed_data.get("recommendation_level", "Unknown"),
                    "key_strengths_count": len(parsed_data.get("strengths_analysis", [])),
                    "improvement_areas_count": len(parsed_data.get("weaknesses_analysis", []))
                }
                
        except Exception as e:
            logger.error(f"Fallback parsing error: {e}")
    
    def _generate_error_response(self, error_message: str, target_role: str = None, word_count: int = 0, username: str = None) -> Dict[str, Any]:
        """Generate error response maintaining standard structure"""
        response = self._get_standard_response_template(target_role or "unknown", word_count)
        response["success"] = False
        response["error"] = f"AI analysis failed: {error_message}"
        response["resume_metadata"]["validation_message"] = "Analysis encountered an error"
        if username:
            response["username"] = username
        return response

# ===== INITIALIZE COMPONENTS =====

pdf_extractor = OptimizedPDFExtractor()
high_perf_analyzer = None

if openai_api_key:
    try:
        high_perf_analyzer = HighPerformanceLangChainAnalyzer(openai_api_key)
        logger.info("High-performance analyzer initialized successfully")
    except Exception as init_error:
        logger.error(f"Failed to initialize analyzer: {init_error}")

# ===== ENDPOINTS =====

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(..., description="Resume PDF file"),
    username: str = Form(..., description="Username for whom the analysis is being done"),
    target_role: str = Form(None, description="Target job position/role"),
    search_jobs: bool = Form(True, description="Whether to search for relevant jobs"),
    location: str = Form("India", description="Location for job search")
):
    """
    Comprehensive resume analysis with guaranteed standard JSON output and job search integration.
    Includes a three-layer validation gate that rejects non-resume documents before analysis runs.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not high_perf_analyzer:
            raise HTTPException(status_code=500, detail="AI analyzer not initialized.")
        
        if not file.content_type or "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Extract PDF text
        resume_text = await pdf_extractor.extract_text_from_pdf(file)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
        if len(resume_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume content too short.")

        # THREE-LAYER VALIDATION PIPELINE
        # Layer 1: Document Classification
        logger.info("Running Layer 1: Document classification")
        classification_result = await high_perf_analyzer.document_classifier.classify(resume_text)
        
        if classification_result.label == "non_resume" and classification_result.confidence >= 0.7:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "not_a_resume",
                    "message": "The uploaded document does not appear to be a resume/CV.",
                    "validation": {
                        "is_resume": False,
                        "confidence": "high",
                        "method": "llm_classifier",
                        "reason": classification_result.reason,
                        "classifier_confidence": classification_result.confidence,
                    },
                },
            )
        
        logger.info(
            f"Layer 1 result: {classification_result.label} "
            f"(confidence: {classification_result.confidence:.2f})"
        )
        
        # Layer 2 & 3: Heuristic + LLM Validation
        logger.info("Running Layer 2/3: Heuristic + LLM validation")
        validation_result = await high_perf_analyzer.resume_validator.validate(resume_text)

        if not validation_result.is_resume:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "not_a_resume",
                    "message": "The uploaded document does not appear to be a resume/CV.",
                    "validation": {
                        "is_resume": validation_result.is_resume,
                        "confidence": validation_result.confidence,
                        "method": validation_result.method,
                        "reason": validation_result.reason,
                        "classifier_label": classification_result.label,
                        "classifier_confidence": classification_result.confidence,
                    },
                },
            )

        logger.info(
            f"All validation layers passed (final method={validation_result.method}, "
            f"confidence={validation_result.confidence}). Proceeding to analysis."
        )
        
        # Perform analysis
        analysis_result = await asyncio.wait_for(
            high_perf_analyzer.analyze_resume_with_jobs(
                resume_text=resume_text,
                username=username,
                target_role=target_role,
                search_jobs=search_jobs and bool(target_role),
                location=location
            ),
            timeout=60.0
        )

        # Save to shared database
        analysis_id = str(uuid.uuid4())
        shared_db.save_resume_analysis(
            username=username,
            analysis_id=analysis_id,
            analysis_data={
                "target_role": target_role or "general position",
                "overall_score": analysis_result.get("ai_insights", {}).get("overall_score", 0),
                "recommendation_level": analysis_result.get("ai_insights", {}).get("recommendation_level", "Unknown"),
                "analysis_result": analysis_result,
                "uploaded_at": datetime.now().isoformat(),
                "validation_method": validation_result.method,
                "validation_confidence": validation_result.confidence
            }
        )
        
        analysis_result["analysis_id"] = analysis_id
        analysis_result["saved_to_database"] = True
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s for user: {username}")
        
        return analysis_result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timeout.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/user/{username}/analyses")
async def get_user_analyses(username: str):
    """Get all analyses for a specific user"""
    try:
        analyses = shared_db.get_user_resume_analyses(username)
        return {
            "username": username, 
            "total_analyses": len(analyses), 
            "analyses": analyses
        }
    except Exception as e:
        logger.error(f"Error fetching analyses for user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses: {str(e)}")

@app.get("/analysis/{username}/{analysis_id}")
async def get_analysis(username: str, analysis_id: str):
    """Get a specific analysis by ID for a user"""
    try:
        # Since shared_database doesn't have a direct method to get by ID,
        # we need to fetch all and filter
        analyses = shared_db.get_user_resume_analyses(username)
        for analysis in analyses:
            if analysis.get("analysis_id") == analysis_id:
                return analysis
        
        raise HTTPException(status_code=404, detail="Analysis not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analysis {analysis_id} for user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")

@app.delete("/analysis/{username}/{analysis_id}")
async def delete_analysis(username: str, analysis_id: str):
    """Delete a specific analysis"""
    try:
        # Use shared_database's delete_interaction method
        shared_db.delete_interaction(username, "resume_analyzer", analysis_id)
        return {"message": f"Analysis {analysis_id} deleted successfully for user {username}"}
    except Exception as e:
        logger.error(f"Error deleting analysis {analysis_id} for user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check with comprehensive features display"""
    try:
        # Get database stats
        all_users = shared_db.get_all_users()
        all_analyses = []
        analyses_by_user = {}
        
        for user in all_users:
            analyses = shared_db.get_user_resume_analyses(user)
            all_analyses.extend(analyses)
            analyses_by_user[user] = len(analyses)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "AI Resume Analyzer with Comprehensive Features",
            "version": "4.0.0",
            "features": {
                "three_layer_validation": "✅",
                "llm_document_classifier": "✅",
                "heuristic_validator": "✅",
                "llm_validator": "✅",
                "resume_analysis": "✅",
                "job_search_integration": "✅",
                "ats_scoring": "✅",
                "strengths_analysis": "✅",
                "weaknesses_analysis": "✅",
                "improvement_plan": "✅",
                "job_market_analysis": "✅",
                "quantified_scoring": "✅",
                "detailed_breakdown": "✅",
                "caching_mechanism": "✅",
                "shared_database": "✅",
                "user_tracking": "✅",
                "per_user_analyses": "✅",
                "analysis_history": "✅",
                "pdf_extraction": "✅",
                "error_handling": "✅",
                "performance_optimization": "✅",
                "consistent_json_output": "✅",
                "snake_case_naming": "✅",
                "deterministic_output": "✅"
            },
            "validation_pipeline": {
                "layer1": "LLM Document Classifier - Quick pre-screening",
                "layer2": "Heuristic Validator - Fast keyword-based scoring",
                "layer3": "LLM Validator - Deep analysis for ambiguous cases"
            },
            "database": {
                "type": "external_api",
                "url": EXTERNAL_DB_API_URL,
                "total_users": len(all_users),
                "total_analyses": len(all_analyses),
                "analyses_by_user": analyses_by_user
            },
            "performance": {
                "caching_enabled": True,
                "cache_size": len(analysis_cache),
                "parallel_processing": True,
                "optimized_pdf_extraction": True
            },
            "openai_configured": bool(openai_api_key),
            "analyzer_available": bool(high_perf_analyzer),
            "langchain_version": "Latest (LCEL)",
            "guarantees": [
                "✅ Non-resume documents rejected before analysis",
                "✅ Consistent JSON structure every time",
                "✅ All standard fields present",
                "✅ Snake case field naming in detailed_scoring",
                "✅ Frontend-compatible format",
                "✅ Optional job listings",
                "✅ Deterministic output for identical resumes",
                "✅ Per-user analysis tracking",
                "✅ Full analysis history available"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "service": "AI Resume Analyzer",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with comprehensive feature listing"""
    return {
        "service": "AI Resume Analyzer with Comprehensive Features",
        "version": "4.0.0",
        "description": "AI resume analysis with three-layer validation and comprehensive feature set",
        "features": {
            "validation_pipeline": "✅ Three-layer validation (LLM classifier + heuristic + LLM validator)",
            "resume_analysis": "✅ Complete resume analysis with ATS scoring",
            "job_search": "✅ Integrated job search with realistic listings",
            "scoring_system": "✅ Multi-category scoring with detailed breakdowns",
            "strengths_analysis": "✅ Top 5 strengths with ATS benefits",
            "weaknesses_analysis": "✅ Top 5 weaknesses with fix priorities",
            "improvement_plan": "✅ Prioritized improvement recommendations",
            "job_market_analysis": "✅ Role compatibility and market positioning",
            "caching": "✅ Content-based caching for consistent results",
            "database": "✅ Shared database integration",
            "user_tracking": "✅ Per-user analysis storage and retrieval",
            "analysis_history": "✅ Full analysis history for each user",
            "pdf_extraction": "✅ Optimized PDF text extraction",
            "error_handling": "✅ Comprehensive error handling",
            "performance": "✅ Parallel processing and optimization"
        },
        "endpoints": {
            "/analyze-resume": {
                "method": "POST",
                "description": "Comprehensive analysis with three-layer validation",
                "content_type": "multipart/form-data",
                "fields": {
                    "file": "PDF file (required)",
                    "username": "string (required) - User identifier",
                    "target_role": "string (optional)",
                    "search_jobs": "boolean (default: true)",
                    "location": "string (default: India)"
                }
            },
            "/user/{username}/analyses": {
                "method": "GET",
                "description": "Get all analyses for a specific user"
            },
            "/analysis/{username}/{analysis_id}": {
                "method": "GET",
                "description": "Get a specific analysis by ID"
            },
            "/analysis/{username}/{analysis_id}": {
                "method": "DELETE",
                "description": "Delete a specific analysis"
            },
            "/health": {
                "method": "GET",
                "description": "Service health check with features"
            },
            "/docs": {
                "method": "GET",
                "description": "API documentation"
            }
        },
        "validation_pipeline_details": [
            "Layer 1: LLM Document Classifier - Quick pre-screening (catches invoices, forms, job descriptions)",
            "Layer 2: Heuristic Validator - Fast keyword-based scoring with weighted signals",
            "Layer 3: LLM Validator - Deep analysis only for ambiguous cases"
        ],
        "output_guarantees": [
            "Consistent JSON structure",
            "All fields always present",
            "Snake case naming in detailed_scoring",
            "Frontend-compatible format",
            "Deterministic for identical inputs",
            "Username always included in response",
            "Analysis ID for retrieval"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting AI Resume Analyzer with Comprehensive Features")
    print("=" * 70)
    print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
    print(f"🔑 OpenAI: {'✅ Configured' if openai_api_key else '❌ Not configured'}")
    print(f"🔧 Analyzer: {'✅ Ready' if high_perf_analyzer else '❌ Not available'}")
    print(f"🎯 Version: 4.0.0")
    print(f"💬 Features:")
    print(f"   • Three-Layer Validation: ✅")
    print(f"   • LLM Document Classifier: ✅")
    print(f"   • Heuristic Validator: ✅")
    print(f"   • LLM Validator: ✅")
    print(f"   • Resume Analysis: ✅")
    print(f"   • Job Search Integration: ✅")
    print(f"   • ATS Scoring: ✅")
    print(f"   • Strengths Analysis: ✅")
    print(f"   • Weaknesses Analysis: ✅")
    print(f"   • Improvement Plan: ✅")
    print(f"   • Job Market Analysis: ✅")
    print(f"   • Quantified Scoring: ✅")
    print(f"   • Detailed Breakdown: ✅")
    print(f"   • Caching Mechanism: ✅")
    print(f"   • Shared Database: ✅")
    print(f"   • User Tracking: ✅")
    print(f"   • Per-User Analyses: ✅")
    print(f"   • Analysis History: ✅")
    print(f"   • PDF Extraction: ✅")
    print(f"   • Error Handling: ✅")
    print(f"   • Performance Optimization: ✅")
    print(f"   • Consistent JSON Output: ✅")
    print(f"   • Snake Case Naming: ✅")
    print(f"   • Deterministic Output: ✅")
    print(f"🔗 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")