"""
AI Academic Chatbot with FULL Personalization Integration & Enhanced Features
This version combines friend-like conversational style with comprehensive personalization
and dynamic resume awareness
"""

import os
import logging
import json
import re
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

# LangChain imports
from langchain_openai import ChatOpenAI
import openai

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from dotenv import load_dotenv

# Import shared database
from shared_database import SharedDatabase, EXTERNAL_DB_API_URL

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AI Academic Chatbot with Personalization & Enhanced Features",
    description="Personalized chatbot with friend-like conversations, smart intent detection, and resume awareness",
    version="6.0.0"
)

# ============================
# Personalization Integration
# ============================

class PersonalizationIntegration:
    """Handles all personalization API calls and context building"""
    
    def __init__(self, personalization_url: str = "http://localhost:8001"):
        self.api_url = personalization_url
        self.cache = {}  # Cache personalization data
        self.cache_timeout = 300  # 5 minutes
    
    def get_user_profile(self, username: str) -> Optional[Dict]:
        """Fetch user profile from personalization module"""
        try:
            # Check cache
            cache_key = f"profile_{username}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now().timestamp() - timestamp) < self.cache_timeout:
                    return cached_data
            
            response = requests.get(f"{self.api_url}/user/{username}/profile", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = (data, datetime.now().timestamp())
                return data
            else:
                logger.warning(f"Personalization API returned {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            logger.warning("Personalization module not available")
            return None
        except Exception as e:
            logger.error(f"Error fetching profile: {e}")
            return None
    
    def build_personalization_context(self, username: str) -> str:
        """Build comprehensive personalization context for LLM"""
        profile = self.get_user_profile(username)
        
        if not profile or not profile.get("data_available", False):
            return ""
        
        context_parts = ["\n=== USER PERSONALIZATION CONTEXT ==="]
        
        # Personality traits
        traits = profile.get("personality_traits", {})
        if traits:
            high_traits = [k.replace('_', ' ').title() for k, v in traits.items() if v > 0.6]
            if high_traits:
                context_parts.append(f"🧠 Personality: {', '.join(high_traits)}")
        
        # Communication style
        comm_style = profile.get("communication_style", {})
        if comm_style:
            formality = comm_style.get("formality", "mixed")
            verbosity = comm_style.get("verbosity", "moderate")
            context_parts.append(f"💬 Communication: {formality} tone, {verbosity} responses")
        
        # Topics of interest
        topics = profile.get("topics_of_interest", [])
        if topics:
            context_parts.append(f"📚 Interests: {', '.join(topics[:5])}")
        
        # Professional interests from resume
        prof_interests = profile.get("professional_interests", [])
        if prof_interests:
            context_parts.append(f"💼 Professional Interests: {', '.join(prof_interests[:5])}")
        
        # Career goals
        career_goals = profile.get("career_goals", [])
        if career_goals:
            context_parts.append(f"🎯 Career Goals: {', '.join(career_goals[:3])}")
        
        # Skill levels
        skills = profile.get("skill_levels", {})
        if skills:
            skill_info = ", ".join([f"{k}: {v}" for k, v in skills.items()])
            context_parts.append(f"🎯 Skills: {skill_info}")
        
        # Resume insights (IMPORTANT!)
        resume_insights = profile.get("resume_insights", {})
        if resume_insights and resume_insights.get("total_analyses", 0) > 0:
            avg_score = resume_insights.get("average_score", 0)
            trend = resume_insights.get("improvement_trend", "stable")
            target_roles = resume_insights.get("target_roles", [])
            strengths = resume_insights.get("common_strengths", [])
            weaknesses = resume_insights.get("common_weaknesses", [])
            
            context_parts.append(f"📄 **Resume Performance:**")
            context_parts.append(f"   - Average Score: {avg_score}%")
            context_parts.append(f"   - Trend: {trend}")
            if target_roles:
                context_parts.append(f"   - Target Roles: {', '.join(target_roles[:3])}")
            if strengths:
                context_parts.append(f"   - Key Strengths: {', '.join(strengths[:3])}")
            if weaknesses:
                context_parts.append(f"   - Areas to Improve: {', '.join(weaknesses[:3])}")
            
            # Add recent analyses
            analyses_history = resume_insights.get("analyses_history", [])
            if analyses_history:
                latest = analyses_history[0]
                context_parts.append(f"   - Latest: {latest.get('score')}% for {latest.get('role')}")
        
        # Recommendations
        recommendations = profile.get("recommendations", {})
        if recommendations:
            learning_recs = recommendations.get("learning_style", [])
            if learning_recs:
                context_parts.append(f"💡 Recommendations: {'; '.join(learning_recs[:2])}")
        
        context_parts.append("=== END PERSONALIZATION CONTEXT ===\n")
        
        return "\n".join(context_parts)
    
    def get_detailed_resume_insights(self, username: str) -> Dict[str, Any]:
        """Get detailed resume insights from personalization module"""
        profile = self.get_user_profile(username)
        
        if not profile:
            return {}
        
        return profile.get("resume_insights", {})
    
    def trigger_profile_update(self, username: str):
        """Trigger profile update in background"""
        try:
            requests.post(f"{self.api_url}/user/{username}/update", timeout=2)
        except:
            pass  # Non-critical, fail silently


# ============================
# Request/Response Models
# ============================

class ChatRequest(BaseModel):
    message: str

class CollegeRecommendation(BaseModel):
    """College recommendation model"""
    id: str
    name: str
    location: str
    type: str
    courses_offered: str
    website: str
    admission_process: str
    approximate_fees: str
    notable_features: str
    source: str

class ChatResponse(BaseModel):
    response: str
    is_recommendation: bool
    timestamp: str
    conversation_title: Optional[str] = None
    recommendations: Optional[List[CollegeRecommendation]] = []
    personalized: bool = False

class UserPreferences(BaseModel):
    """User preferences extracted from conversation"""
    location: Optional[str] = Field(None, description="Preferred city or state for college")
    state: Optional[str] = Field(None, description="Preferred state for college")
    course_type: Optional[str] = Field(None, description="Type of course like Engineering, Medicine, Arts, Commerce, etc.")
    college_type: Optional[str] = Field(None, description="Government, Private, or Deemed university")
    level: Optional[str] = Field(None, description="UG (Undergraduate) or PG (Postgraduate)")
    budget_range: Optional[str] = Field(None, description="Budget preference like low, medium, high")
    specific_course: Optional[str] = Field(None, description="Specific course like BTech, MBA, MBBS, etc.")
    specific_institution_type: Optional[str] = Field(None, description="Specific institution type like IIT, NIT, IIIT, AIIMS, etc.")

# ============================
# Conversation Memory Manager (Enhanced for Shared DB)
# ============================

class ConversationMemoryManager:
    """Manages conversation memory with Shared Database persistence - Enhanced"""
    
    def __init__(self, db: SharedDatabase):
        self.db = db
        self.active_memories = {}  # In-memory cache
        # SINGLE UNIFIED MEMORY - maintains context across ALL conversations
        self.chat_memories = defaultdict(lambda: [])  # Simple list instead of ChatMessageHistory
    
    def load_conversation(self, chat_id: str, username: str) -> dict:
        """Load conversation from database"""
        conv = self.db.get_chatbot_conversation(username, chat_id)
        if conv:
            self.active_memories[chat_id] = conv
            
            # Also load messages into memory
            for msg in conv.get('messages', []):
                if msg['role'] == 'human':
                    self.chat_memories[chat_id].append(
                        HumanMessage(content=msg['content'])
                    )
                elif msg['role'] == 'ai':
                    self.chat_memories[chat_id].append(
                        AIMessage(content=msg['content'])
                    )
            return conv
        return None
    
    def add_message(self, chat_id: str, username: str, role: str, content: str, is_recommendation: bool = False):
        """Add message to conversation"""
        if chat_id not in self.active_memories:
            conv = self.db.get_chatbot_conversation(username, chat_id)
            if conv:
                self.active_memories[chat_id] = conv
            else:
                self.active_memories[chat_id] = {
                    "title": "New Conversation",
                    "messages": [],
                    "preferences": {}
                }
        
        self.active_memories[chat_id]['messages'].append({
            'role': role,
            'content': content,
            'is_recommendation': is_recommendation,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to memory for context
        if role == 'human':
            self.chat_memories[chat_id].append(
                HumanMessage(content=content)
            )
        elif role == 'ai':
            self.chat_memories[chat_id].append(
                AIMessage(content=content)
            )
        
        # Save to shared database
        self.db.save_chatbot_conversation(username, chat_id, self.active_memories[chat_id])
    
    def get_messages(self, chat_id: str, last_n: int = None) -> List[Dict]:
        """Get messages for a chat"""
        if chat_id not in self.active_memories:
            return []
        
        messages = self.active_memories[chat_id]['messages']
        if last_n:
            return messages[-last_n:]
        return messages
    
    def set_title(self, chat_id: str, username: str, title: str):
        """Set conversation title"""
        if chat_id not in self.active_memories:
            self.load_conversation(chat_id, username)
        
        if chat_id in self.active_memories:
            self.active_memories[chat_id]['title'] = title
            self.db.save_chatbot_conversation(username, chat_id, self.active_memories[chat_id])
    
    def get_title(self, chat_id: str) -> Optional[str]:
        """Get conversation title"""
        if chat_id in self.active_memories:
            return self.active_memories[chat_id]['title']
        return None
    
    def set_preferences(self, chat_id: str, username: str, preferences: dict):
        """Set user preferences"""
        if chat_id not in self.active_memories:
            self.load_conversation(chat_id, username)
        
        if chat_id in self.active_memories:
            self.active_memories[chat_id]['preferences'].update(preferences)
            self.db.save_chatbot_conversation(username, chat_id, self.active_memories[chat_id])
    
    def get_preferences(self, chat_id: str) -> dict:
        """Get user preferences"""
        if chat_id in self.active_memories:
            return self.active_memories[chat_id]['preferences']
        return {}
    
    def get_memory_context(self, chat_id: str, max_messages: int = 15) -> List[BaseMessage]:
        """Get memory context (last N messages)"""
        if chat_id in self.chat_memories:
            all_messages = self.chat_memories[chat_id]
            return all_messages[-max_messages:] if len(all_messages) > max_messages else all_messages
        return []

# ============================
# Enhanced Academic Chatbot with Personalization
# ============================

class PersonalizedAcademicChatbot:
    """Academic chatbot with personalization, friend-like conversations, and enhanced features"""
    
    def __init__(self, openai_api_key: str, storage_dir: str = None, model_name: str = "gpt-4o-mini"):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize database (external API - no local storage)
        self.db = SharedDatabase(storage_dir)
        
        # Personalization integration
        self.personalization = PersonalizationIntegration()
        
        # Single LLM for all operations
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            max_tokens=1000,
            api_key=openai_api_key
        )
        
        # Enhanced Memory manager
        self.memory_manager = ConversationMemoryManager(self.db)
        
        # Setup enhanced chains
        self._setup_unified_chain()
        self._setup_intent_classifier()
        self._setup_preference_extraction()
    
    def _setup_unified_chain(self):
        """Setup single unified conversational chain - friend-like, with personalization"""
        unified_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a warm and friendly academic companion. You chat naturally like a supportive friend who genuinely cares.

🎯 YOUR PERSONALITY:
- Talk like a friend, not a formal assistant
- Be warm, encouraging, and relatable
- DON'T bombard with questions - just flow naturally
- Remember everything from the conversation
- Respond directly to what the user asks
- Adapt your style based on the user's personality and preferences

💬 CONVERSATION STYLE:
- If someone says "I want to study astrophysics" → Be excited! Share encouragement, maybe mention it's fascinating, and naturally weave in that you can help find colleges if they want
- If they ask for college recommendations → Jump right in with specific suggestions based on what you know
- If they ask follow-up questions about colleges you mentioned → Reference them naturally like "Oh yeah, IIT Delhi that I mentioned earlier..."
- For general questions → Just answer them warmly and directly
- For resume questions → Reference their actual resume data and provide personalized, specific feedback based on their strengths and areas for improvement

🚫 WHAT NOT TO DO:
- DON'T ask "Are you looking for college recommendations or information?" - just respond naturally
- DON'T list multiple options like "I can help you with: 1. 2. 3." unless explicitly asked
- DON'T be overly formal or robotic
- DON'T ask obvious questions - if they say they want to study something, they probably want help with it
- DON'T give generic resume advice - use their actual resume data

✅ WHAT TO DO:
- Be conversational and natural
- Show enthusiasm about their goals
- Offer help smoothly without being pushy
- If college data is in the context, integrate it naturally
- Remember and reference previous parts of the conversation
- Be encouraging and supportive
- Use personalization data when available to tailor your responses
- When discussing resumes, reference their specific strengths and areas for improvement

CONTEXT AWARENESS:
- You maintain full memory of the conversation
- If you recommended colleges earlier, you can discuss them
- If they mentioned preferences before, you remember them
- Be naturally conversational - like texting with a knowledgeable friend
- Use personalization context to adapt your communication style

PERSONALIZATION CONTEXT (if available):
{personalization_context}

Remember: You're a friend who happens to know a lot about academics and colleges, not a Q&A machine!"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create a runnable that gets chat history and personalization
        def get_chat_history_and_context(input_dict: dict) -> dict:
            chat_id = input_dict.get("chat_id", "default")
            username = input_dict.get("username", "unknown")
            chat_history = self.memory_manager.get_memory_context(chat_id, max_messages=15)
            
            # Get personalization context
            personalization_context = self.personalization.build_personalization_context(username)
            
            return {
                "chat_history": chat_history,
                "input": input_dict.get("input", ""),
                "personalization_context": personalization_context
            }
        
        self.unified_chain = (
            RunnableLambda(get_chat_history_and_context)
            | unified_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _setup_intent_classifier(self):
        """Setup intent classification to determine if user wants college recommendations"""
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier. Analyze if the user is EXPLICITLY asking for college recommendations.

RETURN "YES" ONLY IF:
1. User explicitly asks for college suggestions/recommendations/list
2. User asks "which colleges should I consider" or similar direct questions
3. User asks to "show me colleges" or "tell me about colleges for X"
4. User asks "where can I study X" expecting a list of institutions

RETURN "NO" IF:
1. User is just talking about their interests ("I want to study physics")
2. User is asking general information about a field/course
3. User is greeting or having general conversation
4. User is asking follow-up questions about already mentioned colleges (they already have recommendations)
5. User is asking about admission process, eligibility, etc. without asking for new colleges

Be strict - only return YES when user clearly wants a list of college recommendations.

Answer with just one word: YES or NO"""),
            ("human", "Message: {message}\nContext: {context}")
        ])
        
        self.intent_chain = intent_prompt | self.llm | StrOutputParser()
    
    def _setup_preference_extraction(self):
        """Setup preference extraction"""
        self.preference_parser = PydanticOutputParser(pydantic_object=UserPreferences)
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract user preferences for college search from the conversation.

Conversation History:
{conversation_history}

Current Message:
{current_message}

Extract whatever preferences you can find. If nothing specific is mentioned, return null values.

{format_instructions}

Extract preferences as JSON."""),
            ("human", "Extract preferences from the conversation above.")
        ])
        
        self.preference_chain = (
            extraction_prompt.partial(
                format_instructions=self.preference_parser.get_format_instructions()
            )
            | self.llm
            | self.preference_parser
        )
    
    def _detect_resume_question(self, message: str) -> bool:
        """Detect if user is asking about their resume"""
        resume_keywords = [
            'resume', 'cv', 'my application', 'job application',
            'my profile', 'career', 'how am i doing', 'my performance',
            'resume score', 'resume analysis', 'resume feedback',
            'my resume', 'check my resume', 'review my resume',
            'how is my resume', 'what do you think of my resume',
            'resume review', 'resume suggestions', 'improve my resume',
            'resume help', 'resume advice', 'my resume feedback'
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in resume_keywords)
    
    def _get_resume_insights_context(self, username: str) -> str:
        """Get detailed resume insights for personalization"""
        try:
            # Get resume insights from personalization module
            profile = self.personalization.get_user_profile(username)
            
            if not profile or not profile.get("resume_insights"):
                return ""
            
            resume_insights = profile.get("resume_insights", {})
            
            if resume_insights.get("total_analyses", 0) == 0:
                return ""
            
            context_parts = ["\n=== USER'S RESUME INSIGHTS ==="]
            
            # Basic stats
            context_parts.append(f"📊 Resume Performance:")
            context_parts.append(f"   - Average Score: {resume_insights.get('average_score', 0)}%")
            context_parts.append(f"   - Latest Score: {resume_insights.get('latest_score', 0)}%")
            context_parts.append(f"   - Trend: {resume_insights.get('improvement_trend', 'stable')}")
            
            # Target roles
            target_roles = resume_insights.get('target_roles', [])
            if target_roles:
                context_parts.append(f"🎯 Target Roles: {', '.join(target_roles)}")
            
            # Common strengths - THIS IS WHAT THE USER IS GOOD AT
            strengths = resume_insights.get('common_strengths', [])
            if strengths:
                context_parts.append(f"💪 Key Strengths (what user excels at):")
                for strength in strengths[:3]:
                    context_parts.append(f"   - {strength}")
            
            # Common weaknesses - AREAS WHERE USER NEEDS HELP
            weaknesses = resume_insights.get('common_weaknesses', [])
            if weaknesses:
                context_parts.append(f"📝 Areas for Improvement:")
                for weakness in weaknesses[:3]:
                    context_parts.append(f"   - {weakness}")
            
            # Experience level
            exp_levels = resume_insights.get('experience_levels', [])
            if exp_levels:
                most_common_exp = max(set(exp_levels), key=exp_levels.count) if exp_levels else ""
                context_parts.append(f"👔 Experience Level: {most_common_exp}")
            
            context_parts.append("=== END RESUME INSIGHTS ===\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting resume insights: {e}")
            return ""
    
    def _get_detailed_resume_analysis(self, username: str) -> Dict[str, Any]:
        """Get detailed resume analysis for the latest resume"""
        try:
            # Get the latest resume analysis
            analyses = self.db.get_user_resume_analyses(username)
            
            if not analyses:
                return {}
            
            latest = analyses[0]  # Most recent
            analysis_result = latest.get("analysis_result", {})
            
            # Extract detailed information
            detailed_info = {
                "overall_score": latest.get("overall_score", 0),
                "target_role": latest.get("target_role", ""),
                "strengths": [],
                "weaknesses": [],
                "improvement_plan": {},
                "job_market_analysis": {}
            }
            
            # Get strengths with detailed analysis
            strengths_analysis = analysis_result.get("strengths_analysis", [])
            for strength in strengths_analysis:
                detailed_info["strengths"].append({
                    "strength": strength.get("strength", ""),
                    "why_strong": strength.get("why_its_strong", ""),
                    "evidence": strength.get("evidence", "")
                })
            
            # Get weaknesses with specific fixes
            weaknesses_analysis = analysis_result.get("weaknesses_analysis", [])
            for weakness in weaknesses_analysis:
                detailed_info["weaknesses"].append({
                    "weakness": weakness.get("weakness", ""),
                    "priority": weakness.get("fix_priority", ""),
                    "specific_fix": weakness.get("specific_fix", ""),
                    "timeline": weakness.get("timeline", "")
                })
            
            # Get improvement plan
            improvement_plan = analysis_result.get("improvement_plan", {})
            detailed_info["improvement_plan"] = {
                "critical": improvement_plan.get("critical", []),
                "high": improvement_plan.get("high", []),
                "medium": improvement_plan.get("medium", [])
            }
            
            # Get job market analysis
            job_market = analysis_result.get("job_market_analysis", {})
            detailed_info["job_market_analysis"] = {
                "role_compatibility": job_market.get("role_compatibility", ""),
                "market_positioning": job_market.get("market_positioning", ""),
                "skill_development": job_market.get("skill_development", "")
            }
            
            return detailed_info
            
        except Exception as e:
            logger.error(f"Error getting detailed resume analysis: {e}")
            return {}
    
    def get_personalized_resume_feedback(self, username: str) -> str:
        """Get personalized feedback based on resume analysis"""
        try:
            analyses = self.db.get_user_resume_analyses(username)
            
            if not analyses:
                return "I notice you haven't uploaded your resume for analysis yet. Would you like me to guide you through the Resume Analyzer? It can help identify your strengths and areas for improvement!"
            
            latest = analyses[0]  # Most recent
            score = latest.get("overall_score", 0)
            strengths = latest.get("strengths", [])
            weaknesses = latest.get("weaknesses", [])
            target_role = latest.get("target_role", "your target role")
            
            # Get detailed analysis for deeper insights
            detailed = self._get_detailed_resume_analysis(username)
            
            # Build personalized response
            response = f"Hey! 👋 I've looked at your resume analysis. Here's my personalized feedback:\n\n"
            
            # Overall score with interpretation
            if score >= 80:
                response += f"✨ **Great news!** Your resume scored **{score}%**, which is excellent! You're in a strong position for {target_role} roles.\n\n"
            elif score >= 70:
                response += f"👍 **Good progress!** Your resume scored **{score}%**. You're on the right track for {target_role} positions.\n\n"
            elif score >= 60:
                response += f"📝 Your resume scored **{score}%**. With some improvements, you'll be in great shape for {target_role} roles.\n\n"
            else:
                response += f"📊 Your resume scored **{score}%**. Don't worry - this gives us a clear roadmap for improvement!\n\n"
            
            # Strengths section - PERSONALIZED
            if detailed.get("strengths"):
                response += "💪 **What You're Doing Well:**\n"
                for i, strength_data in enumerate(detailed["strengths"][:3]):
                    response += f"• **{strength_data['strength']}** - {strength_data['why_strong'][:100]}...\n"
                response += "\n"
            elif strengths:
                response += "💪 **Your Key Strengths:**\n"
                for strength in strengths[:3]:
                    response += f"• {strength}\n"
                response += "\n"
            
            # Weaknesses with SPECIFIC FIXES
            if detailed.get("weaknesses"):
                response += "🔧 **Areas to Work On (with specific fixes):**\n"
                for weakness_data in detailed["weaknesses"][:3]:
                    priority = weakness_data.get("priority", "MEDIUM")
                    weakness = weakness_data.get("weakness", "")
                    fix = weakness_data.get("specific_fix", "")
                    
                    priority_emoji = "🔴" if priority == "CRITICAL" else "🟡" if priority == "HIGH" else "🟢"
                    response += f"{priority_emoji} **{weakness}** ({priority} priority)\n"
                    if fix:
                        response += f"   → **Suggestion**: {fix[:150]}\n"
                    
                    timeline = weakness_data.get("timeline", "")
                    if timeline:
                        response += f"   → **Timeline**: {timeline}\n"
                response += "\n"
            elif weaknesses:
                response += "🔧 **Areas to Work On:**\n"
                for weakness in weaknesses[:3]:
                    response += f"• {weakness}\n"
                response += "\n"
            
            # Improvement plan - ACTIONABLE STEPS
            improvement_plan = detailed.get("improvement_plan", {})
            if improvement_plan.get("critical") or improvement_plan.get("high"):
                response += "📋 **Recommended Action Plan:**\n"
                
                for item in improvement_plan.get("critical", [])[:2]:
                    response += f"🔴 **Critical**: {item}\n"
                for item in improvement_plan.get("high", [])[:2]:
                    response += f"🟡 **High Priority**: {item}\n"
                for item in improvement_plan.get("medium", [])[:2]:
                    response += f"🟢 **Medium Priority**: {item}\n"
                response += "\n"
            
            # Career guidance based on resume
            job_market = detailed.get("job_market_analysis", {})
            if job_market.get("role_compatibility") or job_market.get("market_positioning"):
                response += "🎯 **Career Insights Just for You:**\n"
                if job_market.get("role_compatibility"):
                    response += f"• {job_market['role_compatibility']}\n"
                if job_market.get("market_positioning"):
                    response += f"• {job_market['market_positioning']}\n"
                if job_market.get("skill_development"):
                    response += f"• **Focus on**: {job_market['skill_development']}\n"
            
            # Encouraging close
            response += "\n✨ Want me to help you with any specific section or suggest colleges that match your profile?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating personalized feedback: {e}")
            return "I'd love to give you personalized feedback on your resume, but I'm having trouble accessing the analysis right now. Could you try uploading your resume again through the Resume Analyzer?"
    
    def should_get_college_recommendations(self, message: str, chat_id: str) -> bool:
        """Determine if we should fetch college recommendations using LLM intent classification"""
        try:
            # Get recent conversation context
            recent_messages = self.memory_manager.get_messages(chat_id, last_n=5)
            context = " | ".join([f"{msg['role']}: {msg['content'][:100]}" for msg in recent_messages[-3:]])
            
            # Use LLM to classify intent
            result = self.intent_chain.invoke({
                "message": message,
                "context": context
            })
            
            intent = result.strip().upper()
            logger.info(f"Intent classification: {intent} for message: '{message[:50]}...'")
            
            return intent == "YES"
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            # Fallback to simple keyword matching if LLM fails
            message_lower = message.lower().strip()
            fallback_indicators = [
                'recommend college', 'suggest college', 'which college should',
                'show me college', 'list of college', 'colleges for',
                'where should i study', 'where can i study', 'best college for'
            ]
            return any(indicator in message_lower for indicator in fallback_indicators)
    
    def extract_preferences(self, chat_id: str, username: str, current_message: str) -> UserPreferences:
        """Extract user preferences using LLM"""
        try:
            messages = self.memory_manager.get_messages(chat_id, last_n=10)
            conversation_history = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" for msg in messages
            ])
            
            preferences = self.preference_chain.invoke({
                "conversation_history": conversation_history,
                "current_message": current_message
            })
            
            # Save preferences to memory
            if any(value for value in preferences.dict().values()):
                self.memory_manager.set_preferences(chat_id, username, preferences.dict())
            
            return preferences
                
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            prev_prefs = self.memory_manager.get_preferences(chat_id)
            if prev_prefs:
                return UserPreferences(**prev_prefs)
            return UserPreferences()
    
    def get_openai_recommendations(self, preferences: UserPreferences, chat_history: str) -> List[Dict]:
        """Get college recommendations from OpenAI with context awareness"""
        try:
            pref_parts = []
            
            if preferences.specific_institution_type:
                pref_parts.append(f"Institution type: {preferences.specific_institution_type}")
            if preferences.location:
                pref_parts.append(f"Location: {preferences.location}")
            if preferences.state:
                pref_parts.append(f"State: {preferences.state}")
            if preferences.course_type:
                pref_parts.append(f"Course type: {preferences.course_type}")
            if preferences.specific_course:
                pref_parts.append(f"Specific course: {preferences.specific_course}")
            if preferences.college_type:
                pref_parts.append(f"College type: {preferences.college_type}")
            if preferences.budget_range:
                pref_parts.append(f"Budget: {preferences.budget_range}")
            
            # Build comprehensive prompt
            if pref_parts:
                preference_text = ", ".join(pref_parts)
                prompt = f"""Based on these preferences: {preference_text}

Conversation context:
{chat_history[-500:]}

Recommend 5 best colleges in India that match these criteria."""
            else:
                prompt = f"""Based on this conversation:
{chat_history[-500:]}

Recommend 5 diverse, well-known colleges in India that would be relevant."""
            
            prompt += """

Return as JSON array with this exact structure:
[
    {
        "name": "Full College Name",
        "location": "City, State",
        "type": "Government/Private/Deemed",
        "courses": "Main courses offered (be specific)",
        "features": "Key highlights and why it's recommended",
        "website": "Official website URL if known, otherwise 'Visit official website'",
        "admission": "Brief admission process info",
        "fees": "Approximate annual fee range"
    }
]

Return ONLY the JSON array, no additional text."""
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                colleges = json.loads(result)
                return colleges[:5]
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    colleges = json.loads(json_match.group())
                    return colleges[:5]
                return []
                
        except Exception as e:
            logger.error(f"Error getting OpenAI recommendations: {e}")
            return []
    
    def convert_openai_college_to_json(self, college_data: Dict) -> Optional[CollegeRecommendation]:
        """Convert OpenAI college to standardized JSON format"""
        try:
            return CollegeRecommendation(
                id=str(uuid.uuid4()),
                name=college_data.get('name', 'N/A'),
                location=college_data.get('location', 'N/A'),
                type=college_data.get('type', 'N/A'),
                courses_offered=college_data.get('courses', 'N/A'),
                website=college_data.get('website', 'Visit official website for details'),
                admission_process=college_data.get('admission', 'Check official website'),
                approximate_fees=college_data.get('fees', 'Contact institution for fee details'),
                notable_features=college_data.get('features', 'Quality education institution'),
                source="openai_knowledge"
            )
            
        except Exception as e:
            logger.error(f"Error converting OpenAI college: {e}")
            return None
    
    def format_college_context(self, colleges: List[Dict]) -> str:
        """Format college information as context for the LLM"""
        if not colleges:
            return ""
        
        context_parts = ["\n[COLLEGE RECOMMENDATIONS AVAILABLE:"]
        
        for i, college in enumerate(colleges, 1):
            context_parts.append(f"""
{i}. {college.get('name', 'N/A')} ({college.get('location', 'N/A')})
   Type: {college.get('type', 'N/A')}
   Courses: {college.get('courses', 'N/A')}
   Features: {college.get('features', 'N/A')}
   Fees: {college.get('fees', 'N/A')}
   Website: {college.get('website', 'N/A')}
""")
        
        context_parts.append("]")
        return "\n".join(context_parts)
    
    # ============================
    # Title generation only — topic-specific and uses full conversation context
    # ============================
    def generate_conversation_title(self, message: str, chat_id: str) -> str:
        """Generate a topic-specific title from the conversation so each chat gets a unique title."""
        try:
            messages = self.memory_manager.get_messages(chat_id, last_n=8)
            if not messages:
                return "New Conversation"
            transcript = "\n".join(
                f"{'User' if m.get('role') == 'human' else 'Assistant'}: {(m.get('content') or '')[:120]}"
                for m in messages if (m.get('content') or '').strip()
            )
            if not transcript.strip():
                return "New Conversation"

            title_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    (
                        "You generate short 3-7 word conversation titles that reflect the actual topic discussed. "
                        "Use the specific subject (e.g. 'BTech colleges in Karnataka', 'Resume feedback', 'IIT JEE prep'). "
                        "Do NOT use generic titles like 'Academic Discussion', 'New Conversation', or 'College chat'. "
                        "Return ONLY the title — no quotes, no extra text."
                    ),
                ),
                ("human", "Conversation:\n{conversation}\n\nTitle:"),
            ])
            title_chain = title_prompt | self.llm | StrOutputParser()
            title = title_chain.invoke({"conversation": transcript})

            title = (title or "").strip().replace('"', '').replace("'", "").strip(".")
            if not title or title.lower() in ("academic discussion", "academic conversation", "new conversation"):
                first_user = next((m.get("content", "") for m in messages if m.get("role") == "human"), "")
                title = (first_user[:44] + "..." if len(first_user) > 44 else first_user) or "Academic Chat"
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "Academic Conversation"

    def get_response(self, message: str, chat_id: str, username: str) -> Dict[str, Any]:
        """Main unified processing function - conversational, personalized, and context-aware"""
        timestamp = datetime.now().isoformat()
        
        # Ensure user exists in database
        self.db.get_or_create_user(username)
        
        # Load conversation if exists
        self.memory_manager.load_conversation(chat_id, username)
        
        # Save user message
        self.memory_manager.add_message(chat_id, username, 'human', message, False)
        
        # Generate or retrieve conversation title
        existing_title = self.memory_manager.get_title(chat_id)
        conversation_title = existing_title
        
        if not existing_title and len(message.strip()) > 10:
            conversation_title = self.generate_conversation_title(message, chat_id)
            self.memory_manager.set_title(chat_id, username, conversation_title)
        elif not existing_title:
            conversation_title = "New Conversation"
        
        # Check if asking about resume
        if self._detect_resume_question(message):
            logger.info(f"🎯 Resume question detected for {username}")
            
            # Get personalized resume feedback
            resume_feedback = self.get_personalized_resume_feedback(username)
            
            # Also get resume insights for context
            resume_insights_context = self._get_resume_insights_context(username)
            
            # Combine with personalization
            enhanced_message = f"{message}\n\nContext: The user is asking about their resume. Here's what I know about them:\n{resume_insights_context}"
            
            # Process through unified chain for natural response
            try:
                response = self.unified_chain.invoke({
                    "input": enhanced_message,
                    "chat_id": chat_id,
                    "username": username
                })
            except Exception as e:
                logger.error(f"Unified chain failed, using fallback: {e}")
                response = resume_feedback
            
            # Save AI response
            self.memory_manager.add_message(chat_id, username, 'ai', response, False)
            # Refresh title from conversation so it stays topic-specific
            msg_count = len(self.memory_manager.get_messages(chat_id))
            if msg_count >= 2 and (msg_count % 4 == 0 or (conversation_title or "").strip() == "New Conversation"):
                conversation_title = self.generate_conversation_title(message, chat_id)
                self.memory_manager.set_title(chat_id, username, conversation_title)
            return {
                "response": response,
                "is_recommendation": False,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": [],
                "personalized": True
            }
        
        # Check if we should fetch college recommendations
        should_recommend = self.should_get_college_recommendations(message, chat_id)
        
        logger.info(f"🎯 Recommendation triggered: {should_recommend}")
        
        # Prepare input for unified chain
        enhanced_message = message
        recommendations_data = []
        
        # Check personalization availability
        profile = self.personalization.get_user_profile(username)
        has_personalization = bool(profile and profile.get("data_available", False))
        
        # If recommendations needed, add college context
        if should_recommend:
            try:
                logger.info("📚 Fetching college recommendations...")
                
                # Extract preferences
                preferences = self.extract_preferences(chat_id, username, message)
                logger.info(f"Extracted preferences: {preferences.dict()}")
                
                # Get conversation history for context
                messages = self.memory_manager.get_messages(chat_id)
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
                # Get recommendations from OpenAI
                openai_colleges = self.get_openai_recommendations(preferences, chat_history)
                
                # Convert to standardized format
                for college in openai_colleges:
                    json_rec = self.convert_openai_college_to_json(college)
                    if json_rec:
                        recommendations_data.append(json_rec)
                
                # Add context to message
                if recommendations_data:
                    college_context = self.format_college_context(openai_colleges)
                    enhanced_message = f"{message}\n\n{college_context}"
                    logger.info(f"✅ Added {len(recommendations_data)} college recommendations to context")
                    
            except Exception as e:
                logger.error(f"Error fetching recommendations: {e}")
        
        # Process through unified chain
        try:
            response = self.unified_chain.invoke({
                "input": enhanced_message,
                "chat_id": chat_id,
                "username": username
            })
            
            # Save AI response to memory and database
            self.memory_manager.add_message(chat_id, username, 'ai', response, should_recommend)
            # Refresh title from conversation so it stays topic-specific
            msg_count = len(self.memory_manager.get_messages(chat_id))
            if msg_count >= 2 and (msg_count % 4 == 0 or (conversation_title or "").strip() == "New Conversation"):
                conversation_title = self.generate_conversation_title(message, chat_id)
                self.memory_manager.set_title(chat_id, username, conversation_title)
            # Trigger profile update occasionally (every 10 messages)
            if len(self.memory_manager.get_memory_context(chat_id)) % 10 == 0:
                self.personalization.trigger_profile_update(username)
            
            logger.info(f"✅ Response generated successfully (personalized: {has_personalization})")
            
            return {
                "response": response,
                "is_recommendation": should_recommend,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": recommendations_data,
                "personalized": has_personalization
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm having a bit of trouble right now. Could you try asking that again? 😊",
                "is_recommendation": False,
                "timestamp": timestamp,
                "conversation_title": conversation_title,
                "recommendations": [],
                "personalized": has_personalization
            }

# ============================
# Initialize
# ============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

try:
    chatbot = PersonalizedAcademicChatbot(OPENAI_API_KEY)
    logger.info("✅ Enhanced Personalized Academic Chatbot initialized with Shared Database")
    logger.info("📦 Using LATEST LangChain packages")
except Exception as e:
    logger.error(f"❌ Error initializing chatbot: {e}")
    raise

# ============================
# FastAPI Routes
# ============================

@app.get("/")
async def root():
    return {
        "message": "AI Academic Chatbot with Personalization & Enhanced Features",
        "version": "6.0.0",
        "description": "Friend-like academic chatbot with personalization, smart intent detection, and resume awareness",
        "features": {
            "unified_pipeline": "✅",
            "natural_conversations": "✅",
            "smart_intent_detection": "✅",
            "context_awareness": "✅",
            "friend_like_personality": "✅",
            "personalization": "✅",
            "resume_awareness": "✅",
            "dynamic_profile_updates": "✅",
            "personalized_resume_feedback": "✅",
            "communication_style_matching": "✅",
            "personality_adaptation": "✅",
            "college_recommendations": "✅",
            "shared_database": "✅",
            "user_tracking": "✅",
            "conversation_memory": "✅"
        },
        "enhancements": [
            "✅ Friend-like conversational style",
            "✅ Full personalization integration",
            "✅ Resume-aware responses with specific feedback",
            "✅ Dynamic profile updates based on interactions",
            "✅ Personality trait adaptation",
            "✅ Communication style matching",
            "✅ Improved intent classification",
            "✅ Better context awareness",
            "✅ Enhanced college recommendations",
            "✅ Unified memory system",
            "✅ Latest LangChain packages"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    chat_id: str = Query(..., description="Chat ID managed by backend"),
    username: str = Query(..., description="Username")
):
    """Enhanced personalized chat endpoint"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    
    if not chat_id.strip():
        raise HTTPException(status_code=400, detail="Chat ID cannot be empty")
    
    try:
        result = chatbot.get_response(
            message=request.message,
            chat_id=chat_id,
            username=username
        )
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user/{username}/conversations")
async def get_user_conversations(username: str):
    """Get all chatbot conversations for a user"""
    try:
        conversations = chatbot.db.get_user_chatbot_conversations(username)
        return {
            "username": username,
            "total_conversations": len(conversations),
            "conversations": conversations
        }
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversation/{username}/{chat_id}")
async def get_conversation(username: str, chat_id: str):
    """Get specific conversation"""
    try:
        conversation = chatbot.db.get_chatbot_conversation(username, chat_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Add memory context if available
        memory_context = chatbot.memory_manager.get_memory_context(chat_id)
        conversation["memory_context_count"] = len(memory_context)
        
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/conversation/{username}/{chat_id}")
async def delete_conversation(username: str, chat_id: str):
    """Delete a conversation"""
    try:
        chatbot.db.delete_interaction(username, "chatbot", chat_id)
        
        # Also clear from memory manager
        if chat_id in chatbot.memory_manager.active_memories:
            del chatbot.memory_manager.active_memories[chat_id]
        if chat_id in chatbot.memory_manager.chat_memories:
            del chatbot.memory_manager.chat_memories[chat_id]
        
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user/{username}/personalization")
async def get_user_personalization(username: str):
    """Get user personalization status"""
    profile = chatbot.personalization.get_user_profile(username)
    
    if not profile:
        return {
            "username": username,
            "personalization_available": False,
            "message": "Personalization module not available or user has no data"
        }
    
    return {
        "username": username,
        "personalization_available": True,
        "has_resume_data": profile.get("resume_insights", {}).get("total_analyses", 0) > 0,
        "total_interactions": profile.get("total_interactions", 0),
        "personality_traits": profile.get("personality_traits", {}),
        "communication_style": profile.get("communication_style", {}),
        "resume_insights": profile.get("resume_insights", {}),
        "topics_of_interest": profile.get("topics_of_interest", []),
        "professional_interests": profile.get("professional_interests", [])
    }

@app.post("/user/{username}/update-personalization")
async def trigger_personalization_update(username: str):
    """Manually trigger personalization update"""
    chatbot.personalization.trigger_profile_update(username)
    return {"message": f"Personalization update triggered for {username}"}

@app.get("/health")
async def health_check():
    """Health check with comprehensive status"""
    try:
        # Check personalization module
        personalization_status = "connected"
        try:
            response = requests.get("http://localhost:8001/health", timeout=2)
            if response.status_code != 200:
                personalization_status = "disconnected"
        except:
            personalization_status = "disconnected"
        
        # Get stats from shared database
        all_users = chatbot.db.get_all_users()
        
        # Get memory stats
        active_conversations = len(chatbot.memory_manager.active_memories)
        active_memories = len(chatbot.memory_manager.chat_memories)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "AI Academic Chatbot with Personalization & Enhanced Features",
            "version": "6.0.0",
            "features": {
                "unified_pipeline": "✅",
                "natural_conversations": "✅",
                "smart_intent_detection": "✅",
                "context_awareness": "✅",
                "friend_like_personality": "✅",
                "personalization": "✅",
                "resume_awareness": "✅",
                "dynamic_profile_updates": "✅",
                "communication_style_matching": "✅",
                "personality_adaptation": "✅",
                "college_recommendations": "✅",
                "shared_database": "✅",
                "user_tracking": "✅",
                "conversation_memory": "✅"
            },
            "database": {
                "type": "external_api",
                "url": EXTERNAL_DB_API_URL,
                "total_users": len(all_users)
            },
            "memory": {
                "active_conversations": active_conversations,
                "active_memories": active_memories,
                "type": "Shared Database + In-memory context"
            },
            "personalization_module": personalization_status,
            "langchain_version": "Latest (langchain-core, langchain-openai)",
            "enhancements": [
                "✅ Friend-like conversational style",
                "✅ Full personalization integration",
                "✅ Resume-aware responses with specific feedback",
                "✅ Dynamic profile updates based on interactions",
                "✅ Personality trait adaptation",
                "✅ Communication style matching",
                "✅ Improved intent classification",
                "✅ Better context awareness",
                "✅ Enhanced college recommendations",
                "✅ Unified memory system",
                "✅ Latest LangChain packages"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting Enhanced Personalized Academic Chatbot")
    print("=" * 70)
    print(f"📊 Database: External API ({EXTERNAL_DB_API_URL})")
    print(f"🧠 Personalization: Enabled")
    print(f"🎯 Version: 6.0.0 - Friend-like Conversations with Personalization")
    print(f"💬 Features:")
    print(f"   • Unified Pipeline: ✅")
    print(f"   • Natural Conversations: ✅")
    print(f"   • Smart Intent Detection: ✅")
    print(f"   • Context Awareness: ✅")
    print(f"   • Friend-like Personality: ✅")
    print(f"   • Personalization: ✅")
    print(f"   • Resume Awareness: ✅")
    print(f"   • Dynamic Profile Updates: ✅")
    print(f"   • Personalized Resume Feedback: ✅")
    print(f"   • Communication Style Matching: ✅")
    print(f"   • Personality Adaptation: ✅")
    print(f"   • College Recommendations: ✅")
    print(f"   • Shared Database: ✅")
    print(f"   • User Tracking: ✅")
    print(f"   • Conversation Memory: ✅")
    print(f"🔗 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )