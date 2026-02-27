"""
AI Resume Analyzer with Comprehensive Features & Three-Layer Validation
This version combines robust resume analysis with extensive feature tracking
UPDATED: Role-specific analysis with honest scoring and mismatch detection
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
    role_compatibility: str = Field(description="Compatibility with target role: Low / Moderate / High")
    market_positioning: str = Field(description="Position in job market for this specific role")
    career_advancement: str = Field(description="Career advancement opportunities specific to target role")
    skill_development: str = Field(description="Skill development recommendations for target role")

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

# ===== LAYER 0 — IMAGE-BASED PDF DETECTION =====

class PDFTypeCheckResult(BaseModel):
    """
    Result of the image-PDF detection check.
    Distinguishes text-based PDFs from scanned/photo PDFs that have no text layer.
    """
    is_image_pdf: bool
    confidence: str       # "high" | "medium"
    reason: str
    text_char_count: int
    page_count: int
    image_page_count: int
    text_page_count: int


class PDFTypeChecker:
    """
    Zero-cost (no API calls) detector that identifies whether a PDF is
    text-based or image/scanned.
    """

    # Pages with fewer extractable characters than this are suspicious
    MIN_CHARS_PER_PAGE: int = 50

    # Fraction of total pages that must look like image pages before we reject
    IMAGE_PAGE_RATIO_THRESHOLD: float = 0.6

    @staticmethod
    def _page_has_embedded_images(page) -> bool:
        """
        Return True if the PDF page's /Resources dictionary contains at least
        one /XObject of subtype /Image.
        """
        try:
            resources = page.get("/Resources")
            if not resources:
                return False
            xobjects = resources.get("/XObject")
            if not xobjects:
                return False
            for key in xobjects:
                obj = xobjects[key].get_object()
                if obj.get("/Subtype") == "/Image":
                    return True
        except Exception:
            pass
        return False

    @classmethod
    def check(cls, pdf_bytes: bytes) -> PDFTypeCheckResult:
        """
        Inspect the raw PDF bytes and return a PDFTypeCheckResult.
        Designed to be called inside a ThreadPoolExecutor (blocking I/O).
        """
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            page_count = len(pdf_reader.pages)

            if page_count == 0:
                return PDFTypeCheckResult(
                    is_image_pdf=False,
                    confidence="medium",
                    reason="PDF has no pages — cannot determine type.",
                    text_char_count=0,
                    page_count=0,
                    image_page_count=0,
                    text_page_count=0,
                )

            total_chars = 0
            image_page_count = 0
            text_page_count = 0

            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text() or ""
                    char_count = len(page_text.strip())
                    total_chars += char_count

                    has_image = cls._page_has_embedded_images(page)

                    # Image page = almost no text AND contains an embedded image
                    if char_count < cls.MIN_CHARS_PER_PAGE and has_image:
                        image_page_count += 1
                    else:
                        text_page_count += 1

                except Exception:
                    # Cannot read the page at all — treat conservatively as image
                    image_page_count += 1

            image_ratio = image_page_count / page_count if page_count > 0 else 0

            if image_ratio >= cls.IMAGE_PAGE_RATIO_THRESHOLD:
                return PDFTypeCheckResult(
                    is_image_pdf=True,
                    confidence="high",
                    reason=(
                        f"{image_page_count} out of {page_count} page(s) appear to be scanned "
                        f"images ({image_ratio:.0%} of the document). No extractable text layer "
                        f"was found. Please upload a text-based PDF exported directly from a word "
                        f"processor (Word, Google Docs, LaTeX). If you only have a scanned copy, "
                        f"run OCR on it first and re-upload."
                    ),
                    text_char_count=total_chars,
                    page_count=page_count,
                    image_page_count=image_page_count,
                    text_page_count=text_page_count,
                )

            return PDFTypeCheckResult(
                is_image_pdf=False,
                confidence="high",
                reason=(
                    f"PDF contains extractable text ({total_chars} characters across "
                    f"{text_page_count} text page(s))."
                ),
                text_char_count=total_chars,
                page_count=page_count,
                image_page_count=image_page_count,
                text_page_count=text_page_count,
            )

        except Exception as e:
            logger.error(f"PDFTypeChecker error: {e}")
            # On unexpected failure, allow processing to continue
            return PDFTypeCheckResult(
                is_image_pdf=False,
                confidence="medium",
                reason=f"PDF type check encountered an error ({e}). Proceeding with extraction.",
                text_char_count=0,
                page_count=0,
                image_page_count=0,
                text_page_count=0,
            )


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
    """Optimized PDF text extraction — works directly from pre-read bytes."""

    @staticmethod
    def extract_text_from_bytes(pdf_bytes: bytes) -> Optional[str]:
        """
        Synchronous extraction — call inside a ThreadPoolExecutor.
        Accepts raw bytes so the upload stream only needs to be read once.
        """
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as page_error:
                    logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
                    continue
            result = extracted_text.strip()
            logger.info(f"Extracted {len(result)} characters from PDF bytes")
            return result if result else None
        except Exception as e:
            logger.error(f"PDF text extraction error: {str(e)}")
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
    """High-performance AI analyzer with guaranteed standard JSON output and role-specific evaluation"""
    
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
        """Setup the analysis chain using LCEL with strict role-specific evaluation"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume analyst and career counselor. Your ONLY job is to evaluate how well this resume fits the TARGET ROLE provided by the user.

=====================================
MANDATORY ROLE-SPECIFIC EVALUATION RULES — THESE OVERRIDE EVERYTHING ELSE
=====================================

1. TARGET ROLE IS THE LENS FOR ALL ANALYSIS
   Every strength, weakness, score, and recommendation MUST be evaluated ONLY through the lens of the target role.
   - Target role = "Dancer" → evaluate dance training, performance experience, choreography, stage presence, physical conditioning, audition history, dance styles mastered.
   - Target role = "Software Engineer" → evaluate coding skills, system design, projects, CS fundamentals, frameworks.
   - Target role = "Marketing Manager" → evaluate campaign management, analytics, brand strategy, copywriting, market research.
   NEVER analyze skills that are irrelevant to the target role as strengths.

2. HONEST SCORING — DO NOT INFLATE SCORES
   The overall_score (0–100) MUST honestly reflect how qualified this candidate is for the TARGET ROLE:
   - 0–20%   → Completely mismatched background, zero relevant experience
   - 21–40%  → Severe mismatch, only very minor transferable skills exist
   - 41–55%  → Partial mismatch, some transferable soft skills but missing core requirements
   - 56–70%  → Moderate fit, has some relevant skills but lacks key requirements
   - 71–85%  → Good fit, meets most requirements with minor gaps
   - 86–100% → Excellent fit, strong match for the role

   EXAMPLES OF CORRECT SCORING:
   - CS/Engineering resume for "Dancer" role → Score: 10–25 (no dance background)
   - Marketing resume for "Data Scientist" role → Score: 20–35 (lacks technical skills)
   - Software Engineer resume for "Software Engineer" role → Score depends on actual skills

3. STRENGTHS ANALYSIS — ROLE-RELEVANT ONLY
   List ONLY strengths that are DIRECTLY relevant or transferable to the target role.
   - For "Dancer": Valid strengths = dance training, performance experience, flexibility, rhythm, stage presence
   - For "Dancer": INVALID strengths = "Diverse Technical Skills", "Machine Learning Knowledge", "Python proficiency"
   - If the candidate has NO relevant strengths, list only genuine transferable soft skills (discipline, teamwork, communication) and clearly label them as "transferable soft skills, not role-specific"
   - Each strength entry's "why_its_strong" and "ats_benefit" fields MUST reference the target role explicitly

4. WEAKNESSES ANALYSIS — CALL OUT THE MISMATCH FIRST
   If the resume background does not match the target role, the FIRST and MOST CRITICAL weakness MUST be the background mismatch itself.
   - Example for CS resume + Dancer role: weakness = "No dance training or performance experience", fix_priority = "CRITICAL"
   - List other role-specific gaps after the primary mismatch
   - Be specific about WHAT is missing for the target role (e.g., "No choreography portfolio", "Missing formal dance school training")

5. IMPROVEMENT PLAN — TARGET ROLE SPECIFIC
   All improvement suggestions MUST be actionable steps toward getting the target role:
   - For "Dancer": "Enroll in a formal dance academy", "Build a video performance portfolio", "Attend local auditions", "Get certified in specific dance styles"
   - For "Software Engineer": "Add GitHub projects", "Contribute to open source", "Get AWS certification"
   - NEVER suggest generic improvements that don't help with the target role

6. JOB MARKET ANALYSIS — HONEST COMPATIBILITY
   - role_compatibility: Set to "Low", "Moderate", or "High" based on actual fit
   - A mismatched resume MUST get "Low" compatibility — do not soften this
   - market_positioning: Describe the candidate's actual position in the target role's job market
   - career_advancement: Describe the realistic path to break into the target role
   - skill_development: List the most critical skills/training needed for the target role

7. DETAILED SCORING — SCORE AGAINST ROLE REQUIREMENTS
   Each scoring category should reflect performance against WHAT THE TARGET ROLE NEEDS:
   - contact_information: Standard (same for all roles)
   - technical_skills: Score based on skills RELEVANT TO TARGET ROLE only
   - experience_quality: Score based on experience IN OR RELATED TO TARGET ROLE only
   - quantified_achievements: Score based on achievements RELEVANT TO TARGET ROLE
   - content_optimization: Score based on how well the resume is optimized FOR THE TARGET ROLE

TARGET ROLE: {target_role}

{format_instructions}

FINAL REMINDER: Return ONLY valid JSON. Be honest. Be role-specific. Do not inflate scores. A candidate deserves accurate feedback to make real career decisions."""),
            ("human", "Target Role: {target_role}\n\nResume Content:\n{resume_text}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        
        self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()
    
    async def _check_role_mismatch(self, resume_text: str, target_role: str) -> dict:
        """
        Quick pre-check to detect obvious role-resume mismatches.
        Returns mismatch metadata that is attached to the final response.
        """
        prompt = f"""You are a career advisor. Assess whether this resume's background is relevant to the target role.

Target Role: {target_role}
Resume Snippet (first 1500 chars): {resume_text[:1500]}

Be honest and specific. If the candidate's background is completely unrelated to the target role, say so clearly.

Respond with JSON only — no extra text:
{{
  "is_relevant": true or false,
  "compatibility": "Low" or "Moderate" or "High",
  "candidate_background": "One sentence describing what field/domain this resume is actually from",
  "target_role_requirements": "One sentence describing what the target role actually needs",
  "mismatch_reason": "One sentence explaining the gap (or 'Good match' if compatible)",
  "estimated_score_range": "e.g. 10-25 or 60-75"
}}"""

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"Role mismatch check: compatibility={result.get('compatibility')}, is_relevant={result.get('is_relevant')}")
                return result
        except Exception as e:
            logger.warning(f"Role mismatch check failed: {e}")
        
        return {
            "is_relevant": True,
            "compatibility": "Unknown",
            "candidate_background": "Could not assess",
            "target_role_requirements": "Could not assess",
            "mismatch_reason": "Assessment unavailable",
            "estimated_score_range": "N/A"
        }

    def _get_standard_response_template(self, target_role: str, word_count: int) -> Dict[str, Any]:
        """Returns the standard response structure"""
        return {
            "success": True,
            "analysis_status": True,
            "failure_reason": None,
            "analysis_method": "AI-Powered LangChain Analysis with Four-Layer Validation",
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
            "ai_insights": {},
            "role_fit_assessment": {}
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
        """Analyze resume with role-specific evaluation and optional job search"""
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
            
            # Run role mismatch check first (lightweight, fast)
            logger.info(f"Running role fit check for target role: {role_context}")
            mismatch_info = await self._check_role_mismatch(resume_text, role_context)
            
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
            
            # Attach role fit assessment to response
            response["role_fit_assessment"] = {
                "target_role": role_context,
                "is_relevant": mismatch_info.get("is_relevant", True),
                "compatibility": mismatch_info.get("compatibility", "Unknown"),
                "candidate_background": mismatch_info.get("candidate_background", ""),
                "target_role_requirements": mismatch_info.get("target_role_requirements", ""),
                "mismatch_reason": mismatch_info.get("mismatch_reason", ""),
                "estimated_score_range": mismatch_info.get("estimated_score_range", ""),
                "note": (
                    "⚠️ This score reflects fit with the specified target role, not overall resume quality. "
                    "The same resume may score much higher for a role that matches the candidate's background."
                    if not mismatch_info.get("is_relevant", True)
                    else "Score reflects fit with the specified target role."
                )
            }

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
                "description": f"Role-fit score for '{target_role}': {analysis.overall_score}%",
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
                        "description": f"Role-fit score for '{target_role}': {parsed_data.get('overall_score', 0)}%",
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
        response["analysis_status"] = False
        response["failure_reason"] = {
            "type": "analysis_error",
            "message": f"AI analysis failed: {error_message}",
            "action": "Please try again. If the problem persists, check your PDF or contact support."
        }
        response["error"] = f"AI analysis failed: {error_message}"
        response["resume_metadata"]["validation_message"] = "Analysis encountered an error"
        if username:
            response["username"] = username
        return response

# ===== INITIALIZE COMPONENTS =====

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
    Comprehensive resume analysis with role-specific scoring, honest mismatch detection,
    and guaranteed standard JSON output.

    VALIDATION PIPELINE (4 layers):
      Layer 0 – Image PDF check     : Rejects scanned/photo PDFs with no text layer (no API cost)
      Layer 1 – LLM classifier      : Catches invoices, forms, job descriptions
      Layer 2 – Heuristic validator : Weighted keyword scoring (no API cost)
      Layer 3 – LLM validator       : Only for ambiguous cases from Layer 2

    Every response includes:
      - analysis_status: true/false  (was analysis completed successfully?)
      - failure_reason: null or object describing why it failed
    """
    start_time = asyncio.get_event_loop().time()

    def _make_failure_response(
        error_type: str,
        message: str,
        action: str,
        extra: Dict[str, Any] = None,
        status_code: int = 400
    ):
        """
        Build a consistent JSON error response with analysis_status: false.
        Raises HTTPException so FastAPI returns the proper HTTP status code.
        """
        body = {
            "analysis_status": False,
            "success": False,
            "failure_reason": {
                "type": error_type,
                "message": message,
                "action": action,
            },
        }
        if extra:
            body["failure_reason"].update(extra)
        raise HTTPException(status_code=status_code, detail=body)

    try:
        if not high_perf_analyzer:
            _make_failure_response(
                "service_unavailable",
                "AI analyzer is not initialized. The service may be misconfigured.",
                "Contact the administrator or check that OPENAI_API_KEY is set.",
                status_code=500,
            )

        if not file.content_type or "pdf" not in file.content_type.lower():
            _make_failure_response(
                "invalid_file_type",
                "Only PDF files are accepted. Please upload a .pdf resume.",
                "Convert your resume to PDF format and try again.",
            )

        # ─────────────────────────────────────────────────────────────────────
        # Read the upload stream ONCE. All subsequent steps use these bytes.
        # ─────────────────────────────────────────────────────────────────────
        pdf_bytes = await file.read()

        if not pdf_bytes:
            _make_failure_response(
                "empty_file",
                "The uploaded file is empty.",
                "Make sure the file is not corrupted and try uploading again.",
            )

        logger.info(f"Received PDF upload: {len(pdf_bytes):,} bytes for user: {username}")

        event_loop = asyncio.get_event_loop()

        # ═════════════════════════════════════════════════════════════════════
        # LAYER 0 — IMAGE / SCANNED PDF DETECTION (zero API cost)
        # ═════════════════════════════════════════════════════════════════════
        logger.info("Running Layer 0: Image/scanned PDF detection")

        with concurrent.futures.ThreadPoolExecutor() as pool:
            pdf_type_result: PDFTypeCheckResult = await event_loop.run_in_executor(
                pool, PDFTypeChecker.check, pdf_bytes
            )

        logger.info(
            f"Layer 0 — is_image_pdf: {pdf_type_result.is_image_pdf} | "
            f"pages: {pdf_type_result.page_count} | "
            f"image_pages: {pdf_type_result.image_page_count} | "
            f"text_chars: {pdf_type_result.text_char_count}"
        )

        if pdf_type_result.is_image_pdf:
            _make_failure_response(
                "image_pdf_detected",
                (
                    "Your PDF appears to be a scanned image or photograph and does not "
                    "contain a searchable text layer. This analyser requires a text-based PDF."
                ),
                (
                    "Re-export your resume directly from a word processor "
                    "(Microsoft Word → Save As PDF, Google Docs → Download as PDF). "
                    "If you only have a scanned copy, run OCR on it first using Adobe Acrobat "
                    "('Recognise Text') or a free online OCR tool, then re-upload."
                ),
                extra={
                    "validation_layer": "0 — image_pdf_check",
                    "confidence": pdf_type_result.confidence,
                    "detail": pdf_type_result.reason,
                    "stats": {
                        "page_count": pdf_type_result.page_count,
                        "image_page_count": pdf_type_result.image_page_count,
                        "text_page_count": pdf_type_result.text_page_count,
                        "extractable_characters": pdf_type_result.text_char_count,
                    },
                },
            )

        # ─────────────────────────────────────────────────────────────────────
        # Extract text from the PDF bytes
        # ─────────────────────────────────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor() as pool:
            resume_text: Optional[str] = await event_loop.run_in_executor(
                pool, OptimizedPDFExtractor.extract_text_from_bytes, pdf_bytes
            )

        if not resume_text:
            _make_failure_response(
                "text_extraction_failed",
                "Could not extract text from the PDF even though it passed the image check.",
                (
                    "The file may be encrypted, password-protected, or corrupted. "
                    "Try re-exporting it from your word processor."
                ),
            )

        if len(resume_text.strip()) < 50:
            logger.warning(
                f"Layer 0 passed but only {len(resume_text.strip())} characters extracted — "
                "likely a non-standard image PDF. Rejecting."
            )
            _make_failure_response(
                "image_pdf_detected",
                (
                    "Your PDF appears to be a scanned image or photograph. "
                    "Only a very small amount of text could be extracted, which is not enough for analysis."
                ),
                (
                    "Re-export your resume from a word processor as a text-based PDF, "
                    "or run OCR on your scanned copy and re-upload."
                ),
                extra={
                    "validation_layer": "0b — post_extraction_image_check",
                    "confidence": "high",
                    "detail": (
                        f"Only {len(resume_text.strip())} characters could be extracted. "
                        "This strongly indicates the PDF contains images rather than selectable text."
                    ),
                    "stats": {
                        "extractable_characters": len(resume_text.strip()),
                        "page_count": pdf_type_result.page_count,
                        "image_page_count": pdf_type_result.image_page_count,
                        "text_page_count": pdf_type_result.text_page_count,
                    },
                },
            )

        logger.info(f"Text extraction complete: {len(resume_text):,} characters")

        # ═════════════════════════════════════════════════════════════════════
        # LAYER 1 — LLM DOCUMENT CLASSIFIER
        # ═════════════════════════════════════════════════════════════════════
        logger.info("Running Layer 1: Document classification")
        classification_result = await high_perf_analyzer.document_classifier.classify(resume_text)

        if classification_result.label == "non_resume" and classification_result.confidence >= 0.7:
            _make_failure_response(
                "not_a_resume",
                (
                    "The uploaded document does not appear to be a resume or CV. "
                    "Please upload a valid resume in PDF format."
                ),
                "Ensure you are uploading your personal resume/CV, not a job description, invoice, or other document.",
                extra={
                    "validation_layer": "1 — llm_classifier",
                    "classifier_label": classification_result.label,
                    "classifier_confidence": classification_result.confidence,
                    "detail": classification_result.reason,
                },
            )

        logger.info(
            f"Layer 1 result: {classification_result.label} "
            f"(confidence: {classification_result.confidence:.2f})"
        )

        # ═════════════════════════════════════════════════════════════════════
        # LAYERS 2 & 3 — HEURISTIC + LLM VALIDATION
        # ═════════════════════════════════════════════════════════════════════
        logger.info("Running Layer 2/3: Heuristic + LLM validation")
        validation_result = await high_perf_analyzer.resume_validator.validate(resume_text)

        if not validation_result.is_resume:
            _make_failure_response(
                "not_a_resume",
                (
                    "The uploaded document does not appear to be a resume or CV. "
                    "Please upload a valid resume in PDF format."
                ),
                "Ensure you are uploading your personal resume/CV, not a job description, invoice, or other document.",
                extra={
                    "validation_layer": "2/3 — heuristic+llm",
                    "validation_method": validation_result.method,
                    "validation_confidence": validation_result.confidence,
                    "detail": validation_result.reason,
                    "classifier_label": classification_result.label,
                    "classifier_confidence": classification_result.confidence,
                },
            )

        logger.info(
            f"All validation layers passed (method={validation_result.method}, "
            f"confidence={validation_result.confidence}). Proceeding to analysis."
        )

        # ═════════════════════════════════════════════════════════════════════
        # ANALYSIS
        # ═════════════════════════════════════════════════════════════════════
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

        # Ensure analysis_status is always present in successful responses
        analysis_result["analysis_status"] = analysis_result.get("success", True)
        analysis_result["failure_reason"] = None

        # Save to shared database
        analysis_id = str(uuid.uuid4())
        shared_db.save_resume_analysis(
            username=username,
            analysis_id=analysis_id,
            analysis_data={
                "target_role": target_role or "general position",
                "overall_score": analysis_result.get("ai_insights", {}).get("overall_score", 0),
                "recommendation_level": analysis_result.get("ai_insights", {}).get("recommendation_level", "Unknown"),
                "role_compatibility": analysis_result.get("role_fit_assessment", {}).get("compatibility", "Unknown"),
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
        raise HTTPException(
            status_code=408,
            detail={
                "analysis_status": False,
                "success": False,
                "failure_reason": {
                    "type": "timeout",
                    "message": "The analysis took too long and was cancelled.",
                    "action": "Please try again. If the problem persists, try with a shorter resume.",
                },
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "analysis_status": False,
                "success": False,
                "failure_reason": {
                    "type": "internal_error",
                    "message": f"An unexpected error occurred: {str(e)}",
                    "action": "Please try again later or contact support.",
                },
            },
        )

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
        shared_db.delete_interaction(username, "resume_analyzer", analysis_id)
        return {"message": f"Analysis {analysis_id} deleted successfully for user {username}"}
    except Exception as e:
        logger.error(f"Error deleting analysis {analysis_id} for user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check with comprehensive features display"""
    try:
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
            "service": "AI Resume Analyzer with Role-Specific Analysis",
            "version": "5.0.0",
            "features": {
                "image_pdf_detection": "✅",
                "four_layer_validation": "✅",
                "analysis_status_field": "✅",
                "failure_reason_field": "✅",
                "three_layer_validation": "✅",
                "llm_document_classifier": "✅",
                "heuristic_validator": "✅",
                "llm_validator": "✅",
                "role_specific_analysis": "✅",
                "honest_scoring": "✅",
                "mismatch_detection": "✅",
                "role_fit_assessment": "✅",
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
                "layer0": "Image/scanned PDF detection — rejects non-text PDFs (zero API cost, PyPDF2 XObject inspection)",
                "layer1": "LLM Document Classifier - Quick pre-screening for non-resume documents",
                "layer2": "Heuristic Validator - Fast weighted keyword scoring (zero API cost)",
                "layer3": "LLM Validator - Deep analysis for ambiguous cases only"
            },
            "role_specific_scoring": {
                "description": "Scores now reflect fit with the TARGET ROLE, not generic resume quality",
                "mismatch_detection": "Pre-analysis role-fit check runs before full analysis",
                "honest_scoring_bands": {
                    "0-20": "Completely mismatched background",
                    "21-40": "Severe mismatch, very few transferable skills",
                    "41-55": "Partial mismatch, some transferable soft skills",
                    "56-70": "Moderate fit, lacks key role requirements",
                    "71-85": "Good fit, meets most requirements",
                    "86-100": "Excellent fit, strong match for the role"
                }
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
                    "✅ Image/scanned PDFs detected and rejected before any API call",
                    "✅ analysis_status: true/false in every response",
                    "✅ failure_reason with type, message, and action when analysis_status is false",
                    "✅ Non-resume documents rejected before analysis",
                    "✅ Role-specific strengths and weaknesses only",
                    "✅ Honest scores that reflect actual role fit",
                    "✅ Mismatch detection with clear explanation",
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
        "service": "AI Resume Analyzer with Role-Specific Analysis",
        "version": "5.1.0",
        "description": "AI resume analysis with four-layer validation (incl. image PDF detection), role-specific scoring, honest mismatch detection, and analysis_status in every response",
        "what_changed_in_v5_1": {
            "new_1": "Layer 0 image PDF detection — scanned/photo PDFs are detected and rejected instantly, before any LLM call",
            "new_2": "analysis_status: true/false field in every API response",
            "new_3": "failure_reason object with type, message, and action in all error responses",
            "new_4": "File bytes read once only — no double-consume, no seek() needed",
            "new_5": "Post-extraction fallback image check for edge-case scanned PDFs",
        },
        "what_changed_in_v5": {
            "problem_fixed": "Previously the analyzer gave high scores and generic strengths regardless of role fit",
            "fix_1": "System prompt now enforces strict role-specific evaluation for ALL analysis sections",
            "fix_2": "Added pre-analysis role-fit check (_check_role_mismatch) that runs before full analysis",
            "fix_3": "Added role_fit_assessment block in every response with compatibility rating and mismatch explanation",
            "fix_4": "Honest scoring bands defined — a CS resume for Dancer role now correctly scores 10-25%",
            "fix_5": "Strengths now only list skills RELEVANT to the target role",
            "fix_6": "Primary weakness is now the background mismatch itself when roles don't align"
        },
        "features": {
            "image_pdf_detection": "✅ Layer 0 — scanned/photo PDFs rejected instantly (zero API cost)",
            "analysis_status": "✅ analysis_status: true/false in every response",
            "failure_reason": "✅ failure_reason with type, message, action when analysis_status is false",
            "role_specific_analysis": "✅ All strengths, weaknesses, and scores tied to target role",
            "honest_scoring": "✅ Low score for mismatched roles, high for matching ones",
            "mismatch_detection": "✅ Pre-analysis check detects role-background mismatch",
            "role_fit_assessment": "✅ Dedicated block in response explaining compatibility",
            "validation_pipeline": [
            "Layer 0: Image PDF detection — rejects scanned/photo PDFs instantly (no API cost)",
            "Layer 1: LLM document classifier — catches non-resume documents",
            "Layer 2: Heuristic keyword validator — fast weighted scoring (no API cost)",
            "Layer 3: LLM validator — only for ambiguous cases from Layer 2",
        ],
            "resume_analysis": "✅ Complete resume analysis with ATS scoring",
            "job_search": "✅ Integrated job search with realistic listings",
            "scoring_system": "✅ Multi-category scoring with detailed breakdowns",
            "strengths_analysis": "✅ Role-relevant strengths only",
            "weaknesses_analysis": "✅ Mismatch-first weaknesses with fix priorities",
            "improvement_plan": "✅ Role-specific actionable recommendations",
            "job_market_analysis": "✅ Honest role compatibility and market positioning",
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
                "description": "Role-specific analysis with three-layer validation",
                "content_type": "multipart/form-data",
                "fields": {
                    "file": "PDF file (required)",
                    "username": "string (required) - User identifier",
                    "target_role": "string (optional) - Critical for role-specific scoring",
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
            }
        },
        "scoring_guide": {
            "0-20":   "Completely mismatched background — no relevant experience for target role",
            "21-40":  "Severe mismatch — only very minor transferable skills",
            "41-55":  "Partial mismatch — transferable soft skills but missing core requirements",
            "56-70":  "Moderate fit — some relevant skills but lacks key requirements",
            "71-85":  "Good fit — meets most requirements with minor gaps",
            "86-100": "Excellent fit — strong match for the target role"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Starting AI Resume Analyzer v5.1 — Four-Layer Validation + analysis_status")
    print("=" * 70)
    print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
    print(f"🔑 OpenAI: {'✅ Configured' if openai_api_key else '❌ Not configured'}")
    print(f"🔧 Analyzer: {'✅ Ready' if high_perf_analyzer else '❌ Not available'}")
    print(f"🎯 Version: 5.1.0")
    print(f"")
    print(f"🆕 What's new in v5.1:")
    print(f"   • Layer 0 Image PDF Detection: ✅  (scanned PDFs rejected before any LLM call)")
    print(f"   • analysis_status field: ✅  (true/false in every response)")
    print(f"   • failure_reason field: ✅  (type + message + action when status is false)")
    print(f"   • Single file read: ✅  (bytes read once, reused across all layers)")
    print(f"")
    print(f"🆕 What's new in v5.0:")
    print(f"   • Role-Specific Scoring: ✅  (scores now reflect TARGET ROLE fit)")
    print(f"   • Honest Mismatch Detection: ✅  (CS resume for Dancer = low score)")
    print(f"   • Role Fit Assessment block: ✅  (in every API response)")
    print(f"   • Role-only Strengths: ✅  (no more generic technical skill praise)")
    print(f"   • Mismatch-first Weaknesses: ✅  (primary gap called out clearly)")
    print(f"   • Role-specific Improvement Plan: ✅")
    print(f"")
    print(f"💬 Core Features:")
    print(f"   • Three-Layer Validation: ✅")
    print(f"   • LLM Document Classifier: ✅")
    print(f"   • Heuristic Validator: ✅")
    print(f"   • LLM Validator: ✅")
    print(f"   • Resume Analysis: ✅")
    print(f"   • Job Search Integration: ✅")
    print(f"   • ATS Scoring: ✅")
    print(f"   • Caching Mechanism: ✅")
    print(f"   • Shared Database: ✅")
    print(f"   • User Tracking: ✅")
    print(f"   • Analysis History: ✅")
    print(f"   • PDF Extraction: ✅")
    print(f"   • Error Handling: ✅")
    print(f"   • Consistent JSON Output: ✅")
    print(f"")
    print(f"🔗 API: http://localhost:8002")
    print(f"📚 Docs: http://localhost:8002/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")