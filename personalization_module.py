
"""
Personalization Module for Resume Analyzer and Academic Chatbot
Analyzes user behavior across both modules and generates personality insights
Data stored in external API at http://3.7.255.54:3003/db
"""

import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import uuid
import os
import requests
import openai
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import shared database
from shared_database import SharedDatabase, API_PATH_PREFIX

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# ============================
# Configuration
# ============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "placeholder-key")
EXTERNAL_DB_API_URL = os.getenv("EXTERNAL_DB_API_URL", "http://3.7.255.54:3003")
logger = logging.getLogger(__name__)

# ============================
# External DB API Client
# ============================

def _api_path(p: str) -> str:
    """Build API path with configurable prefix"""
    prefix = (API_PATH_PREFIX or "db").strip("/")
    return f"/{prefix}/{p}" if prefix else f"/{p}"

class ExternalDBAPIClient:
    """Client for external database API - stores user profiles and personalization reports"""
    
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or EXTERNAL_DB_API_URL).rstrip("/")
        self.timeout = 30
    
    def update_user_profile(self, username: str, profile_data: dict) -> bool:
        """Update user profile via PUT"""
        try:
            url = f"{self.base_url}{_api_path('user-profile')}"
            # Exclude raw_interactions to keep payload size reasonable
            profile = {
                "username": username,
                "total_interactions": profile_data.get("total_interactions", 0),
                "personality_traits": profile_data.get("personality_traits", {}),
                "communication_style": profile_data.get("communication_style", {}),
                "topics_of_interest": profile_data.get("topics_of_interest", []),
                "professional_interests": profile_data.get("professional_interests", []),
                "career_goals": profile_data.get("career_goals", []),
                "skill_levels": profile_data.get("skill_levels", {}),
                "behavior_patterns": profile_data.get("behavior_patterns", {}),
                "recommendations": profile_data.get("recommendations", {}),
                "resume_insights": profile_data.get("resume_insights", {}),
                "data_available": profile_data.get("data_available", False),
                "updated_at": profile_data.get("updated_at", datetime.now().isoformat()),
            }
            payload = {
                "username": username,
                "profile": profile,
            }
            response = requests.put(
                url,
                params={"username": username},
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in (200, 201, 204):
                logger.info(f"Updated user profile for {username} via external API")
                return True
            else:
                logger.error(f"External API returned {response.status_code}: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update user profile via external API: {e}")
            return False
    
    def create_personalization_report(self, username: str, report_data: dict) -> bool:
        """Create personalization report via POST"""
        try:
            url = f"{self.base_url}{_api_path('personalization-report')}"
            payload = {
                "username": username,
                "report_id": report_data.get("report_id", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "generated_at": report_data.get("generated_at", datetime.now().isoformat()),
                "report": {
                    "username": report_data.get("username", username),
                    "personality_type": report_data.get("personality_type", ""),
                    "summary": report_data.get("summary", ""),
                    "has_data": report_data.get("has_data", False),
                    "detailed_analysis": report_data.get("detailed_analysis", {}),
                    "communication_insights": report_data.get("communication_insights", {}),
                    "professional_assessment": report_data.get("professional_assessment", {}),
                    "growth_recommendations": report_data.get("growth_recommendations", {}),
                    "strengths": report_data.get("strengths", []),
                    "areas_for_improvement": report_data.get("areas_for_improvement", []),
                    "resume_performance": report_data.get("resume_performance", {}),
                }
            }
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in (200, 201):
                logger.info(f"Created personalization report for {username} via external API")
                return True
            else:
                logger.error(f"External API returned {response.status_code}: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create personalization report via external API: {e}")
            return False

# ============================
# Data Models
# ============================

class UserInteraction(BaseModel):
    """Single interaction record"""
    module: str  # "resume_analyzer" or "chatbot"
    session_id: str
    timestamp: str
    role: str  # "user" or "assistant" or "system"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

def _default_profile_values() -> dict:
    """Default values for UserProfile fields when loading from API/DB"""
    return {
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "modules_used": [],
        "session_ids": {"chatbot": [], "resume_analyzer": []},
        "total_interactions": 0,
        "personality_traits": {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "emotional_stability": 0.5,
        },
        "communication_style": {
            "formality": "mixed",
            "verbosity": "moderate",
            "questioning_style": "exploratory",
            "emotional_tone": "mixed",
        },
        "topics_of_interest": [],
        "professional_interests": [],
        "career_goals": [],
        "skill_levels": {
            "technical_writing": "intermediate",
            "career_planning": "intermediate",
            "academic_research": "intermediate",
            "self_promotion": "intermediate",
        },
        "behavior_patterns": {},
        "recommendations": {},
        "raw_interactions": [],
        "resume_insights": {},
    }


class UserProfile(BaseModel):
    """User profile with personality analysis"""
    username: str
    created_at: str = ""
    updated_at: str = ""
    modules_used: List[str] = Field(default_factory=list)
    session_ids: Dict[str, List[str]] = Field(default_factory=lambda: {"chatbot": [], "resume_analyzer": []})
    total_interactions: int = 0
    personality_traits: Dict[str, float] = Field(default_factory=dict)
    communication_style: Dict[str, Any] = Field(default_factory=dict)
    topics_of_interest: List[str] = Field(default_factory=list)
    professional_interests: List[str] = Field(default_factory=list)
    career_goals: List[str] = Field(default_factory=list)
    skill_levels: Dict[str, str] = Field(default_factory=dict)
    behavior_patterns: Dict[str, Any] = Field(default_factory=dict)
    recommendations: Dict[str, Any] = Field(default_factory=dict)
    raw_interactions: List[Any] = Field(default_factory=list)  # UserInteraction or dict
    resume_insights: Dict[str, Any] = Field(default_factory=dict)
    data_available: bool = False

class PersonalityReport(BaseModel):
    """Comprehensive personality report"""
    username: str
    report_id: str
    generated_at: str
    summary: str
    personality_type: str
    detailed_analysis: Dict[str, Any]
    communication_insights: Dict[str, Any]
    professional_assessment: Dict[str, Any]
    growth_recommendations: Dict[str, Any]
    strengths: List[str]
    areas_for_improvement: List[str]
    resume_performance: Dict[str, Any]
    has_data: bool = False

# ============================
# Data Collector (Enhanced for Resume Analyzer)
# ============================

class DataCollector:
    """Collects data from Shared Database"""
    
    def __init__(self, shared_db_dir: str = "shared_data"):
        self.shared_db = SharedDatabase(shared_db_dir)
    
    def collect_chatbot_data(self, username: str) -> List[UserInteraction]:
        """Collect data from chatbot module via Shared Database"""
        interactions = []
        
        # Check if user exists in shared database
        all_users = self.shared_db.get_all_users()
        if username not in all_users:
            return interactions
        
        # Get all chatbot interactions for user from shared database
        chatbot_interactions = self.shared_db.get_user_interactions(username, "chatbot")
        
        for interaction in chatbot_interactions:
            data = interaction.get("data", {})
            session_id = interaction["session_id"]
            
            # Extract messages from conversation
            messages = data.get("messages", [])
            for msg in messages:
                user_interaction = UserInteraction(
                    module="chatbot",
                    session_id=session_id,
                    timestamp=msg.get("timestamp", interaction.get("created_at", datetime.now().isoformat())),
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                    metadata={
                        "conversation_title": data.get("title", ""),
                        "is_recommendation": msg.get("is_recommendation", False)
                    }
                )
                interactions.append(user_interaction)
        
        return interactions
    
    def collect_resume_analyzer_data(self, username: str) -> List[UserInteraction]:
        """Collect data from resume analyzer module via Shared Database"""
        interactions = []
        
        # Check if user exists in shared database
        all_users = self.shared_db.get_all_users()
        if username not in all_users:
            return interactions
        
        # Get all resume analyses for user from shared database
        resume_analyses = self.shared_db.get_user_interactions(username, "resume_analyzer")
        
        for analysis in resume_analyses:
            data = analysis.get("data", {})
            session_id = analysis["session_id"]
            
            # Create interaction from analysis summary
            analysis_result = data.get("analysis_result", {})
            
            # Extract key information from resume analysis
            summary_content = self._create_resume_summary(data, analysis_result)
            
            user_interaction = UserInteraction(
                module="resume_analyzer",
                session_id=session_id,
                timestamp=analysis.get("created_at", datetime.now().isoformat()),
                role="system",
                content=summary_content,
                metadata={
                    "target_role": data.get("target_role", ""),
                    "overall_score": data.get("overall_score", 0),
                    "recommendation_level": data.get("recommendation_level", ""),
                    "uploaded_at": data.get("uploaded_at", ""),
                    "strengths": data.get("strengths", []),
                    "weaknesses": data.get("weaknesses", [])
                }
            )
            interactions.append(user_interaction)
        
        return interactions
    
    def _create_resume_summary(self, data: dict, analysis_result: dict) -> str:
        """Create a textual summary of resume analysis for personality insights"""
        target_role = data.get("target_role", "general position")
        score = data.get("overall_score", 0)
        rec_level = data.get("recommendation_level", "Unknown")
        
        # Extract key insights
        strengths = data.get("strengths", [])
        weaknesses = data.get("weaknesses", [])
        
        summary = f"Resume analyzed for {target_role}. Overall score: {score}%. Recommendation: {rec_level}. "
        
        if strengths:
            summary += f"Key strengths: {', '.join(strengths[:3])}. "
        
        if weaknesses:
            summary += f"Areas for improvement: {', '.join(weaknesses[:3])}."
        
        return summary
    
    def collect_all_user_data(self, username: str) -> List[UserInteraction]:
        """Collect all data from both modules"""
        # First check if user exists
        all_users = self.shared_db.get_all_users()
        if username not in all_users:
            print(f"User '{username}' not found in shared database")
            return []
        
        chatbot_data = self.collect_chatbot_data(username)
        resume_data = self.collect_resume_analyzer_data(username)
        
        all_data = chatbot_data + resume_data
        # Sort by timestamp
        all_data.sort(key=lambda x: x.timestamp)
        
        print(f"Collected {len(all_data)} interactions for user '{username}'")
        return all_data
    
    def get_resume_insights(self, username: str) -> Dict[str, Any]:
        """Get aggregated resume analysis insights"""
        resume_analyses = self.shared_db.get_user_resume_analyses(username)
        
        if not resume_analyses:
            return {
                "total_analyses": 0,
                "average_score": 0,
                "latest_score": 0,
                "target_roles": [],
                "improvement_trend": "No data",
                "common_strengths": [],
                "common_weaknesses": [],
                "technical_skills_trend": 0,
                "experience_levels": []
            }
        
        total_analyses = len(resume_analyses)
        scores = [a.get("overall_score", 0) for a in resume_analyses]
        average_score = sum(scores) / len(scores) if scores else 0
        
        # Get target roles
        target_roles = []
        for a in resume_analyses:
            role = a.get("target_role", "")
            if role and role != "general position":
                target_roles.append(role)
        target_roles = list(set(target_roles))
        
        # Collect strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        experience_levels = []
        
        for analysis in resume_analyses:
            all_strengths.extend(analysis.get("strengths", []))
            all_weaknesses.extend(analysis.get("weaknesses", []))
            
            # Extract experience level from professional profile if available
            analysis_result = analysis.get("analysis_result", {})
            exec_summary = analysis_result.get("executive_summary", {})
            prof_profile = exec_summary.get("professional_profile", {})
            exp_level = prof_profile.get("experience_level", "")
            if exp_level:
                experience_levels.append(exp_level)
        
        # Count frequencies
        strength_counter = Counter(all_strengths)
        weakness_counter = Counter(all_weaknesses)
        
        # Calculate improvement trend
        if len(scores) >= 2:
            if scores[0] > scores[-1]:
                trend = "Improving"
            elif scores[0] < scores[-1]:
                trend = "Declining"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"
        
        return {
            "total_analyses": total_analyses,
            "average_score": round(average_score, 1),
            "latest_score": scores[0] if scores else 0,
            "target_roles": target_roles,
            "improvement_trend": trend,
            "common_strengths": [s for s, _ in strength_counter.most_common(5)],
            "common_weaknesses": [w for w, _ in weakness_counter.most_common(5)],
            "technical_skills_trend": sum([a.get("technical_skills_count", 0) for a in resume_analyses]) / total_analyses if total_analyses > 0 else 0,
            "experience_levels": experience_levels,
            "analyses_history": [
                {
                    "date": a.get("created_at", ""),
                    "role": a.get("target_role", ""),
                    "score": a.get("overall_score", 0)
                }
                for a in resume_analyses[:5]
            ]
        }
    
    def user_exists_in_database(self, username: str) -> bool:
        """Check if user exists in shared database"""
        all_users = self.shared_db.get_all_users()
        return username in all_users

# ============================
# Personality Analyzer
# ============================

class PersonalityAnalyzer:
    """Analyzes user personality from interaction data"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name=model_name,
            temperature=0.7
        )
    
    def analyze_interactions(self, interactions: List[UserInteraction], resume_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user interactions and resume data to extract personality traits"""
        
        if not interactions:
            return self._get_default_analysis()
        
        # Prepare data for analysis
        chatbot_msgs = [i for i in interactions if i.module == "chatbot" and i.role == "user"]
        resume_msgs = [i for i in interactions if i.module == "resume_analyzer"]
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(chatbot_msgs, resume_msgs, resume_insights)
        
        try:
            response = self.llm.invoke(prompt)
            analysis = self._parse_llm_response(response.content)
            return analysis
        except Exception as e:
            print(f"Error in personality analysis: {e}")
            return self._get_default_analysis()
    
    def _build_analysis_prompt(self, chatbot_msgs: List[UserInteraction], 
                               resume_msgs: List[UserInteraction],
                               resume_insights: Dict[str, Any]) -> str:
        """Build prompt for LLM analysis with enhanced resume understanding"""
        
        prompt = f"""Analyze the following user data and provide personality insights in JSON format.

CHATBOT INTERACTIONS ({len(chatbot_msgs)} messages):
"""
        
        # Sample up to 20 messages to avoid token limits
        sample_size = min(20, len(chatbot_msgs))
        for msg in chatbot_msgs[:sample_size]:
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            prompt += f"- User said: {content}\n"
        
        prompt += f"""

RESUME ANALYSIS INSIGHTS:
Total analyses: {resume_insights.get('total_analyses', 0)}
Average score: {resume_insights.get('average_score', 0)}%
Target roles: {', '.join(resume_insights.get('target_roles', []))}
Improvement trend: {resume_insights.get('improvement_trend', 'N/A')}

Common strengths (what user is good at):
{chr(10).join(['- ' + s for s in resume_insights.get('common_strengths', [])[:5]])}

Common weaknesses (areas needing improvement):
{chr(10).join(['- ' + w for w in resume_insights.get('common_weaknesses', [])[:5]])}

Based on this data, what can we infer about this user's personality, interests, and behavior?
- What topics are they interested in? (look at both chat messages and target roles)
- What are their professional interests and career goals?
- What is their professional level? (based on experience levels)
- What are their communication patterns?
- How do they approach problems? (based on strengths/weaknesses)
- What can we infer about their learning style?

Analyze and return a JSON object with:
{{
  "personality_traits": {{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "emotional_stability": 0.0-1.0
  }},
  "communication_style": {{
    "formality": "formal/casual/mixed",
    "verbosity": "concise/moderate/detailed",
    "questioning_style": "direct/exploratory/uncertain",
    "emotional_tone": "positive/neutral/analytical/mixed"
  }},
  "topics_of_interest": ["topic1", "topic2", ...],
  "professional_interests": ["field1", "field2", ...],
  "career_goals": ["goal1", "goal2", ...],
  "skill_levels": {{
    "technical_writing": "beginner/intermediate/advanced",
    "career_planning": "beginner/intermediate/advanced",
    "academic_research": "beginner/intermediate/advanced",
    "self_promotion": "beginner/intermediate/advanced"
  }},
  "behavior_patterns": {{
    "engagement_level": "high/medium/low",
    "goal_orientation": "high/medium/low",
    "learning_approach": "structured/exploratory/mixed",
    "help_seeking_behavior": "proactive/reactive/balanced",
    "detail_orientation": "high/medium/low"
  }},
  "inferred_interests_from_resume": ["interest1", "interest2", ...],
  "professional_maturity": "entry/mid/senior/expert"
}}

Provide only valid JSON, no additional text."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Ensure all required fields exist
                if "personality_traits" in data:
                    # Convert any string values to float
                    for key, value in data["personality_traits"].items():
                        if isinstance(value, str):
                            try:
                                data["personality_traits"][key] = float(value)
                            except:
                                data["personality_traits"][key] = 0.5
                
                # Ensure communication_style exists
                if "communication_style" not in data:
                    data["communication_style"] = {
                        "formality": "mixed",
                        "verbosity": "moderate",
                        "questioning_style": "exploratory",
                        "emotional_tone": "mixed"
                    }
                
                # Ensure required lists exist
                if "topics_of_interest" not in data:
                    data["topics_of_interest"] = []
                if "professional_interests" not in data:
                    data["professional_interests"] = []
                if "career_goals" not in data:
                    data["career_goals"] = []
                if "inferred_interests_from_resume" not in data:
                    data["inferred_interests_from_resume"] = []
                
                return data
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
        
        return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis structure"""
        return {
            "personality_traits": {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "emotional_stability": 0.5
            },
            "communication_style": {
                "formality": "mixed",
                "verbosity": "moderate",
                "questioning_style": "exploratory",
                "emotional_tone": "mixed"
            },
            "topics_of_interest": [],
            "professional_interests": [],
            "career_goals": [],
            "skill_levels": {
                "technical_writing": "intermediate",
                "career_planning": "intermediate",
                "academic_research": "intermediate",
                "self_promotion": "intermediate"
            },
            "behavior_patterns": {
                "engagement_level": "medium",
                "goal_orientation": "medium",
                "learning_approach": "mixed",
                "help_seeking_behavior": "balanced",
                "detail_orientation": "medium"
            },
            "inferred_interests_from_resume": [],
            "professional_maturity": "mid"
        }
    
    def generate_personality_type(self, traits: Dict[str, float], username: str, resume_insights: Dict[str, Any]) -> str:
        """Generate personality type label based on traits and resume data"""
        
        if not traits or all(v == 0.5 for v in traits.values()):
            if resume_insights.get("total_analyses", 0) > 0:
                # Use resume data to infer type
                avg_score = resume_insights.get("average_score", 0)
                if avg_score >= 80:
                    return f"HIGH_PERFORMER_{username.upper()}"
                elif avg_score >= 70:
                    return f"RISING_TALENT_{username.upper()}"
                else:
                    return f"GROWING_LEARNER_{username.upper()}"
            return f"NO_DATA_YET_{username.upper()}"
        
        # Determine dominant traits
        high_traits = [k for k, v in traits.items() if v > 0.7]
        
        if "openness" in high_traits and "conscientiousness" in high_traits:
            return "Analytical Explorer"
        elif "extraversion" in high_traits and "agreeableness" in high_traits:
            return "Social Collaborator"
        elif "conscientiousness" in high_traits:
            return "Methodical Achiever"
        elif "openness" in high_traits:
            return "Creative Thinker"
        elif "emotional_stability" in high_traits:
            return "Calm Strategist"
        else:
            return "Balanced Learner"
    
    def generate_recommendations(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate personalized recommendations"""
        
        recommendations = {
            "learning_style": [],
            "career_guidance": [],
            "skill_development": [],
            "communication_tips": []
        }
        
        # Based on personality traits
        traits = profile.personality_traits
        
        if traits.get("openness", 0.5) > 0.7:
            recommendations["learning_style"].append("Explore diverse subjects and interdisciplinary topics")
            recommendations["career_guidance"].append("Consider roles that involve innovation and creativity")
        
        if traits.get("conscientiousness", 0.5) > 0.7:
            recommendations["learning_style"].append("Use structured study plans and schedules")
            recommendations["career_guidance"].append("Excel in detail-oriented and organized environments")
        
        if traits.get("extraversion", 0.5) > 0.7:
            recommendations["communication_tips"].append("Leverage your social skills in group projects")
            recommendations["career_guidance"].append("Consider client-facing or leadership roles")
        
        if traits.get("agreeableness", 0.5) > 0.7:
            recommendations["communication_tips"].append("Your collaborative nature is a strength in team settings")
        
        if traits.get("emotional_stability", 0.5) > 0.7:
            recommendations["communication_tips"].append("Your calm approach helps in stressful situations")
        
        # Based on resume insights
        resume_insights = profile.resume_insights
        if resume_insights.get("average_score", 0) < 70:
            recommendations["skill_development"].append("Focus on improving resume presentation and quantifying achievements")
            recommendations["career_guidance"].append("Consider working with a career counselor to strengthen your profile")
        elif resume_insights.get("average_score", 0) > 85:
            recommendations["skill_development"].append("Your resume is strong - focus on networking and interview preparation")
        
        if resume_insights.get("improvement_trend") == "Declining":
            recommendations["skill_development"].append("Review recent feedback and address highlighted weaknesses")
        elif resume_insights.get("improvement_trend") == "Improving":
            recommendations["skill_development"].append("Great progress! Keep iterating on your resume")
        
        # Based on professional interests
        if profile.professional_interests:
            interests = ", ".join(profile.professional_interests[:2])
            recommendations["career_guidance"].append(f"Explore opportunities in {interests}")
        
        # Default recommendations if none generated
        if not any(recommendations.values()):
            recommendations["learning_style"].append("Continue engaging with academic content regularly")
            recommendations["career_guidance"].append("Explore different career paths through research and networking")
            recommendations["skill_development"].append("Consider taking online courses to build new skills")
            recommendations["communication_tips"].append("Practice articulating your achievements clearly")
        
        return recommendations

# ============================
# Personalization Engine
# ============================

class PersonalizationEngine:
    """Main engine that coordinates data collection, analysis, and reporting"""
    
    def __init__(self, openai_api_key: str, shared_db_dir: str = "shared_data", external_db_url: str = None):
        self.data_collector = DataCollector(shared_db_dir)
        self.personality_analyzer = PersonalityAnalyzer(openai_api_key)
        self.shared_db = self.data_collector.shared_db
        self.external_db = ExternalDBAPIClient(base_url=external_db_url)
    
    def get_or_create_user_profile(self, username: str) -> UserProfile:
        """Get existing profile or create new one"""
        
        # Check if profile exists in database
        existing_profile = self.shared_db.get_user_profile(username)
        
        if existing_profile:
            # Merge with defaults (API/DB may return partial profile)
            merged = {**_default_profile_values(), **{k: v for k, v in existing_profile.items() if v is not None}}
            merged["username"] = username
            return UserProfile(**merged)
        
        # Create new profile
        interactions = self.data_collector.collect_all_user_data(username)
        resume_insights = self.data_collector.get_resume_insights(username)
        
        user_info = self.shared_db.get_or_create_user(username)
        
        if not interactions:
            # Return empty profile for new users
            return UserProfile(
                username=username,
                created_at=user_info["created_at"],
                updated_at=datetime.now().isoformat(),
                modules_used=user_info.get("modules_used", []),
                session_ids=user_info.get("session_ids", {"chatbot": [], "resume_analyzer": []}),
                total_interactions=0,
                personality_traits={
                    "openness": 0.5,
                    "conscientiousness": 0.5,
                    "extraversion": 0.5,
                    "agreeableness": 0.5,
                    "emotional_stability": 0.5
                },
                communication_style={
                    "formality": "mixed",
                    "verbosity": "moderate",
                    "questioning_style": "exploratory",
                    "emotional_tone": "mixed"
                },
                topics_of_interest=[],
                professional_interests=[],
                career_goals=[],
                skill_levels={
                    "technical_writing": "intermediate",
                    "career_planning": "intermediate",
                    "academic_research": "intermediate",
                    "self_promotion": "intermediate"
                },
                behavior_patterns={},
                recommendations={},
                raw_interactions=[],
                resume_insights=resume_insights,
                data_available=False
            )
        
        # Analyze interactions
        analysis = self.personality_analyzer.analyze_interactions(interactions, resume_insights)
        
        # Build profile
        profile = UserProfile(
            username=username,
            created_at=user_info["created_at"],
            updated_at=datetime.now().isoformat(),
            modules_used=user_info.get("modules_used", []),
            session_ids=user_info.get("session_ids", {"chatbot": [], "resume_analyzer": []}),
            total_interactions=len(interactions),
            personality_traits=analysis.get("personality_traits", {}),
            communication_style=analysis.get("communication_style", {}),
            topics_of_interest=analysis.get("topics_of_interest", []),
            professional_interests=analysis.get("professional_interests", []),
            career_goals=analysis.get("career_goals", []),
            skill_levels=analysis.get("skill_levels", {}),
            behavior_patterns=analysis.get("behavior_patterns", {}),
            recommendations={},
            raw_interactions=interactions,
            resume_insights=resume_insights,
            data_available=True
        )
        
        # Generate recommendations
        profile.recommendations = self.personality_analyzer.generate_recommendations(profile)
        
        # Save to external API (PUT /db/user-profile)
        self.external_db.update_user_profile(username, profile.dict())
        
        return profile
    
    def update_user_data(self, username: str) -> Dict[str, Any]:
        """Update user profile with latest data"""
        
        if not self.data_collector.user_exists_in_database(username):
            return {
                "success": False,
                "message": f"User '{username}' not found in database",
                "username": username
            }
        
        # Collect latest data
        interactions = self.data_collector.collect_all_user_data(username)
        resume_insights = self.data_collector.get_resume_insights(username)
        
        if not interactions:
            return {
                "success": True,
                "message": "User exists but has no interactions yet",
                "username": username,
                "total_interactions": 0
            }
        
        # Re-analyze
        analysis = self.personality_analyzer.analyze_interactions(interactions, resume_insights)
        
        # Update profile
        user_info = self.shared_db.get_or_create_user(username)
        profile = UserProfile(
            username=username,
            created_at=user_info["created_at"],
            updated_at=datetime.now().isoformat(),
            modules_used=user_info.get("modules_used", []),
            session_ids=user_info.get("session_ids", {"chatbot": [], "resume_analyzer": []}),
            total_interactions=len(interactions),
            personality_traits=analysis.get("personality_traits", {}),
            communication_style=analysis.get("communication_style", {}),
            topics_of_interest=analysis.get("topics_of_interest", []),
            professional_interests=analysis.get("professional_interests", []),
            career_goals=analysis.get("career_goals", []),
            skill_levels=analysis.get("skill_levels", {}),
            behavior_patterns=analysis.get("behavior_patterns", {}),
            recommendations={},
            raw_interactions=interactions,
            resume_insights=resume_insights,
            data_available=True
        )
        
        profile.recommendations = self.personality_analyzer.generate_recommendations(profile)
        
        # Save to external API (PUT /db/user-profile)
        self.external_db.update_user_profile(username, profile.dict())
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "username": username,
            "total_interactions": len(interactions),
            "updated_at": profile.updated_at
        }
    
    def generate_personality_report(self, username: str) -> PersonalityReport:
        """Generate comprehensive personality report"""
        
        profile = self.get_or_create_user_profile(username)
        
        # Generate personality type
        personality_type = self.personality_analyzer.generate_personality_type(
            profile.personality_traits,
            username,
            profile.resume_insights
        )
        
        # Build report
        report = PersonalityReport(
            username=username,
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            summary=self._generate_summary(profile, personality_type),
            personality_type=personality_type,
            detailed_analysis={
                "personality_traits": profile.personality_traits,
                "behavior_patterns": profile.behavior_patterns,
                "total_interactions": profile.total_interactions
            },
            communication_insights={
                "style": profile.communication_style,
                "topics_of_interest": profile.topics_of_interest,
                "professional_interests": profile.professional_interests
            },
            professional_assessment={
                "skill_levels": profile.skill_levels,
                "modules_used": profile.modules_used,
                "career_goals": profile.career_goals
            },
            growth_recommendations=profile.recommendations,
            strengths=self._extract_strengths(profile),
            areas_for_improvement=self._extract_improvements(profile),
            resume_performance=profile.resume_insights,
            has_data=profile.data_available
        )
        
        # Save report to external API (POST /db/personalization-report)
        report_dict = report.dict()
        self.external_db.create_personalization_report(username, report_dict)
        
        return report
    
    def _generate_summary(self, profile: UserProfile, personality_type: str) -> str:
        """Generate summary text for report"""
        
        if profile.total_interactions == 0:
            return f"New user profile for {profile.username}. No interaction data available yet."
        
        summary = f"{profile.username} is classified as a '{personality_type}' based on {profile.total_interactions} interactions across "
        summary += f"{len(profile.modules_used)} module(s). "
        
        # Add resume insights
        resume_insights = profile.resume_insights
        if resume_insights.get("total_analyses", 0) > 0:
            summary += f"Resume analysis shows an average score of {resume_insights.get('average_score', 0)}% "
            summary += f"with a {resume_insights.get('improvement_trend', 'stable')} trend. "
        
        # Add professional interests
        if profile.professional_interests:
            summary += f"Professional interests include: {', '.join(profile.professional_interests[:3])}. "
        
        # Add dominant traits
        high_traits = [k.replace('_', ' ').title() for k, v in profile.personality_traits.items() if v > 0.7]
        if high_traits:
            summary += f"Dominant traits include: {', '.join(high_traits)}."
        
        return summary
    
    def _extract_strengths(self, profile: UserProfile) -> List[str]:
        """Extract user strengths from profile"""
        strengths = []
        
        # From personality traits
        traits = profile.personality_traits
        if traits.get("conscientiousness", 0) > 0.7:
            strengths.append("Highly organized and detail-oriented")
        if traits.get("openness", 0) > 0.7:
            strengths.append("Open to new experiences and creative thinking")
        if traits.get("extraversion", 0) > 0.7:
            strengths.append("Strong social and communication skills")
        if traits.get("agreeableness", 0) > 0.7:
            strengths.append("Collaborative and team-oriented")
        if traits.get("emotional_stability", 0) > 0.7:
            strengths.append("Calm under pressure and emotionally resilient")
        
        # From resume performance
        resume_insights = profile.resume_insights
        if resume_insights.get("average_score", 0) >= 80:
            strengths.append("Excellent resume presentation and professional profile")
        if resume_insights.get("improvement_trend") == "Improving":
            strengths.append("Shows continuous improvement in professional development")
        
        # From resume strengths
        if resume_insights.get("common_strengths"):
            for s in resume_insights["common_strengths"][:2]:
                strengths.append(f"Recognized strength: {s}")
        
        # From engagement
        if profile.total_interactions > 50:
            strengths.append("Highly engaged and active learner")
        
        return strengths if strengths else ["Developing foundational skills"]
    
    def _extract_improvements(self, profile: UserProfile) -> List[str]:
        """Extract areas for improvement from profile"""
        improvements = []
        
        # From resume performance
        resume_insights = profile.resume_insights
        if resume_insights.get("average_score", 0) < 70:
            improvements.append("Resume presentation needs strengthening")
        if resume_insights.get("improvement_trend") == "Declining":
            improvements.append("Address recent performance decline in resume quality")
        
        # From resume weaknesses
        if resume_insights.get("common_weaknesses"):
            for w in resume_insights["common_weaknesses"][:2]:
                improvements.append(f"Area to improve: {w}")
        
        # From engagement
        if profile.total_interactions < 10:
            improvements.append("Increase engagement with learning platforms")
        
        # From personality traits (low traits indicate areas to develop)
        traits = profile.personality_traits
        if traits.get("conscientiousness", 0.5) < 0.4:
            improvements.append("Develop more structured approach to tasks")
        if traits.get("extraversion", 0.5) < 0.4:
            improvements.append("Practice networking and communication skills")
        
        # From skill levels
        beginner_skills = [k for k, v in profile.skill_levels.items() if v == "beginner"]
        if beginner_skills:
            improvements.append(f"Develop skills in: {', '.join(beginner_skills)}")
        
        return improvements if improvements else ["Continue current learning path"]
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """Get user statistics"""
        
        exists = self.data_collector.user_exists_in_database(username)
        
        if not exists:
            return {
                "username": username,
                "exists": False,
                "message": "User not found in database"
            }
        
        db_stats = self.shared_db.get_user_stats(username)
        profile = self.get_or_create_user_profile(username)
        
        return {
            "username": username,
            "exists": True,
            "created_at": db_stats["created_at"],
            "updated_at": db_stats.get("updated_at", db_stats["created_at"]),
            "total_sessions": db_stats["total_sessions"],
            "chatbot_sessions": db_stats["chatbot_sessions"],
            "resume_analyzer_sessions": db_stats["resume_analyzer_sessions"],
            "total_interactions": profile.total_interactions,
            "modules_used": db_stats.get("modules_used", []),
            "personality_type": self.personality_analyzer.generate_personality_type(
                profile.personality_traits,
                username,
                profile.resume_insights
            ),
            "has_profile": db_stats["has_personalization_profile"],
            "resume_insights": profile.resume_insights,
            "data_available": profile.data_available
        }
    
    def user_has_interactions(self, username: str) -> bool:
        """Check if user has any interactions"""
        interactions = self.data_collector.collect_all_user_data(username)
        return len(interactions) > 0

# ============================
# FastAPI Application
# ============================

app = FastAPI(
    title="Personalization Module API",
    description="Personality analysis for Academic Chatbot and Resume Analyzer users",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = PersonalizationEngine(
    openai_api_key=OPENAI_API_KEY,
    shared_db_dir="shared_data"
)

# ============================
# API Endpoints
# ============================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Personalization Module API",
        "version": "3.0.0",
        "description": "Personality analysis for Academic Chatbot and Resume Analyzer",
        "endpoints": {
            "/user/{username}/profile": "GET - Get user personality profile",
            "/user/{username}/report": "GET - Generate personality report",
            "/user/{username}/stats": "GET - Get user statistics",
            "/user/{username}/update": "POST - Update user data",
            "/user/{username}/exists": "GET - Check if user exists",
            "/users": "GET - List all users",
            "/health": "GET - Health check"
        },
        "features": [
            "✅ Integrated with Resume Analyzer",
            "✅ All data stored in shared_data directory",
            "✅ Personality analysis via GPT",
            "✅ Resume performance tracking",
            "✅ Comprehensive reporting",
            "✅ Dynamic profile updates"
        ]
    }

@app.get("/user/{username}/profile")
async def get_user_profile(username: str):
    """Get user personality profile"""
    try:
        profile = engine.get_or_create_user_profile(username)
        response = profile.dict()
        
        # Add data availability info
        response["data_available"] = profile.total_interactions > 0
        response["user_exists_in_database"] = engine.data_collector.user_exists_in_database(username)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/user/{username}/report")
async def generate_personality_report(username: str):
    """Generate comprehensive personality report"""
    try:
        report = engine.generate_personality_report(username)
        response = report.dict()
        
        # Add metadata
        response["has_data"] = "NO_DATA" not in response["personality_type"]
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/user/{username}/stats")
async def get_user_stats(username: str):
    """Get user statistics"""
    try:
        stats = engine.get_user_stats(username)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/user/{username}/update")
async def update_user_data(username: str):
    """Update user profile with latest data"""
    try:
        result = engine.update_user_data(username)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/user/{username}/exists")
async def check_user_exists(username: str):
    """Check if user exists in database"""
    try:
        exists = engine.data_collector.user_exists_in_database(username)
        return {
            "username": username,
            "exists_in_database": exists,
            "has_interactions": engine.user_has_interactions(username) if exists else False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/users")
async def list_all_users():
    """List all users with basic info"""
    try:
        all_users = engine.shared_db.get_all_users()
        users_list = []
        
        for username in all_users:
            stats = engine.get_user_stats(username)
            users_list.append(stats)
        
        return {
            "total_users": len(users_list),
            "users": sorted(users_list, key=lambda x: x.get("created_at", ""), reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        shared_db = engine.shared_db
        all_users = shared_db.get_all_users()
        
        openai_status = "configured" if OPENAI_API_KEY and OPENAI_API_KEY != "placeholder-key" else "missing"
        
        return {
            "status": "healthy",
            "service": "Personalization Module (Resume Analyzer + Chatbot)",
            "version": "3.0.0",
            "timestamp": datetime.now().isoformat(),
            "openai_api_key": openai_status,
            "database": {
                "type": "external_api",
                "url": EXTERNAL_DB_API_URL,
                "total_users": len(all_users)
            },
            "modules": [
                "academic_chatbot",
                "resume_analyzer"
            ],
            "features": [
                "✅ Personality analysis via GPT",
                "✅ Resume performance tracking",
                "✅ All data in external API (no local storage)",
                "✅ Comprehensive reporting",
                "✅ Bi-directional insights for modules",
                "✅ Dynamic profile updates"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# ============================
# Server Runner
# ============================

def run_server():
    """Run the FastAPI server"""
    
    print("=" * 60)
    print("🚀 Starting Personalization Module API")
    print("=" * 60)
    print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
    print(f"🧠 Modules: Academic Chatbot + Resume Analyzer")
    print(f"🔗 API URL: http://localhost:8001")
    print(f"📚 Documentation: http://localhost:8001/docs")
    print(f"🩺 Health Check: http://localhost:8001/health")
    print("=" * 60)
    
    # Check OpenAI API key
    if OPENAI_API_KEY == "placeholder-key" or not OPENAI_API_KEY:
        print("❌ WARNING: OpenAI API key not configured!")
        print("💡 Set OPENAI_API_KEY in .env file")
        print("=" * 60)
    
    uvicorn.run(
        "personalization_module:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()