"""
Personalization Integration Helper for Academic Chatbot
This module provides utilities to fetch and use personalization insights in the chatbot
"""

import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PersonalizationHelper:
    """Helper class to integrate personalization insights into chatbot"""
    
    def __init__(self, personalization_api_url: str = "http://localhost:8001"):
        self.api_url = personalization_api_url
    
    def get_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch user personality profile from personalization module"""
        try:
            response = requests.get(
                f"{self.api_url}/user/{username}/profile",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch profile for {username}: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to personalization module: {e}")
            return None
    
    def get_user_report(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive personality report"""
        try:
            response = requests.get(
                f"{self.api_url}/user/{username}/report",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch report for {username}: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to personalization module: {e}")
            return None
    
    def build_personalization_context(self, username: str) -> str:
        """Build context string with personalization insights for LLM"""
        profile = self.get_user_profile(username)
        
        if not profile or not profile.get("data_available", False):
            return ""
        
        context_parts = ["\n[PERSONALIZATION INSIGHTS:]"]
        
        # Add personality type
        personality_type = profile.get("personality_traits", {})
        if personality_type:
            traits_str = ", ".join([
                f"{k}: {v:.2f}" for k, v in personality_type.items()
                if isinstance(v, (int, float))
            ])
            if traits_str:
                context_parts.append(f"Personality traits: {traits_str}")
        
        # Add communication style
        comm_style = profile.get("communication_style", {})
        if comm_style:
            context_parts.append(f"Communication style: {comm_style.get('formality', 'mixed')}, {comm_style.get('verbosity', 'moderate')}")
        
        # Add topics of interest
        topics = profile.get("topics_of_interest", [])
        if topics:
            context_parts.append(f"Topics of interest: {', '.join(topics[:5])}")
        
        # Add skill levels
        skills = profile.get("skill_levels", {})
        if skills:
            skills_str = ", ".join([f"{k}: {v}" for k, v in skills.items()])
            context_parts.append(f"Skill levels: {skills_str}")
        
        # Add recommendations
        recs = profile.get("recommendations", {})
        if recs:
            learning_recs = recs.get("learning_style", [])
            if learning_recs:
                context_parts.append(f"Learning recommendations: {'; '.join(learning_recs[:2])}")
        
        # Add resume insights
        resume_insights = profile.get("resume_insights", {})
        if resume_insights and resume_insights.get("total_analyses", 0) > 0:
            context_parts.append(
                f"Resume performance: {resume_insights.get('average_score', 0)}% average score, "
                f"{resume_insights.get('improvement_trend', 'stable')} trend"
            )
        
        context_parts.append("[END PERSONALIZATION INSIGHTS]\n")
        
        return "\n".join(context_parts)
    
    def get_resume_summary_for_chat(self, username: str) -> str:
        """Get resume analysis summary formatted for chat response"""
        profile = self.get_user_profile(username)
        
        if not profile:
            return "I don't have access to your resume analysis yet. Please upload your resume through the Resume Analyzer module first."
        
        resume_insights = profile.get("resume_insights", {})
        
        if not resume_insights or resume_insights.get("total_analyses", 0) == 0:
            return "You haven't uploaded any resume for analysis yet. Would you like me to guide you through the Resume Analyzer module?"
        
        # Build friendly summary
        total = resume_insights.get("total_analyses", 0)
        avg_score = resume_insights.get("average_score", 0)
        latest_score = resume_insights.get("latest_score", 0)
        trend = resume_insights.get("improvement_trend", "stable")
        target_roles = resume_insights.get("target_roles", [])
        
        summary = f"Based on your {total} resume analysis/analyses:\n\n"
        summary += f"📊 **Average Score**: {avg_score}%\n"
        summary += f"📈 **Latest Score**: {latest_score}%\n"
        summary += f"📉 **Trend**: {trend}\n"
        
        if target_roles:
            summary += f"🎯 **Target Roles**: {', '.join(target_roles[:3])}\n"
        
        summary += "\n"
        
        # Add interpretation
        if avg_score >= 80:
            summary += "✨ Your resume is in excellent shape! "
        elif avg_score >= 70:
            summary += "👍 Your resume is good, with room for improvement. "
        elif avg_score >= 60:
            summary += "📝 Your resume needs some work to stand out. "
        else:
            summary += "⚠️ Your resume needs significant improvements. "
        
        if trend == "Improving":
            summary += "Great job on the improvements you've made!"
        elif trend == "Declining":
            summary += "Let's work on getting your resume back on track."
        else:
            summary += "Keep refining your resume with each application."
        
        # Get specific recommendations from latest report
        report = self.get_user_report(username)
        if report and report.get("has_data", False):
            improvements = report.get("areas_for_improvement", [])
            if improvements:
                summary += f"\n\n**Key areas to focus on**:\n"
                for imp in improvements[:3]:
                    summary += f"• {imp}\n"
        
        return summary
    
    def should_personalize_response(self, username: str) -> bool:
        """Check if we should apply personalization for this user"""
        profile = self.get_user_profile(username)
        
        if not profile:
            return False
        
        # Only personalize if user has sufficient interaction data
        total_interactions = profile.get("total_interactions", 0)
        return total_interactions >= 5  # Minimum threshold
    
    def get_personalized_greeting(self, username: str) -> str:
        """Generate personalized greeting based on user profile"""
        profile = self.get_user_profile(username)
        
        if not profile or not profile.get("data_available", False):
            return f"Hi {username}! 👋"
        
        # Get personality type
        comm_style = profile.get("communication_style", {})
        formality = comm_style.get("formality", "mixed")
        
        # Get resume insights
        resume_insights = profile.get("resume_insights", {})
        has_resume = resume_insights.get("total_analyses", 0) > 0
        
        # Customize greeting
        if formality == "formal":
            greeting = f"Hello {username},"
        else:
            greeting = f"Hey {username}! 👋"
        
        # Add context-aware message
        topics = profile.get("topics_of_interest", [])
        if topics and len(topics) > 0:
            greeting += f" Ready to explore {topics[0]} today?"
        elif has_resume:
            greeting += " How can I help you with your academic or career goals today?"
        else:
            greeting += " What would you like to learn about today?"
        
        return greeting
    
    def trigger_profile_update(self, username: str) -> bool:
        """Trigger an update of user profile in personalization module"""
        try:
            response = requests.post(
                f"{self.api_url}/user/{username}/update",
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Profile updated successfully for {username}")
                return True
            else:
                logger.warning(f"Profile update failed for {username}: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error triggering profile update: {e}")
            return False