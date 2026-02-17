"""
INTEGRATION EXAMPLE: How to Add Personalization to Your Existing Chatbot

This file shows the exact code changes needed to integrate personalization
into your existing main.py chatbot.
"""

# ============================================================================
# STEP 1: Add imports at the top of main.py
# ============================================================================

# Add these imports after your existing imports:
from personalization_helper import PersonalizationHelper
import requests

# ============================================================================
# STEP 2: Initialize PersonalizationHelper in your chatbot class
# ============================================================================

class EnhancedChatbot:
    """Your existing chatbot class with personalization added"""
    
    def __init__(self, db: SharedDatabase):
        # ... your existing initialization code ...
        
        # ADD THIS: Initialize personalization helper
        self.personalization = PersonalizationHelper(
            personalization_api_url="http://localhost:8001"
        )
        
        # ADD THIS: Cache for profiles (5 minute TTL)
        self.profile_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    # ========================================================================
    # STEP 3: Add helper methods for personalization
    # ========================================================================
    
    def _get_cached_profile(self, username: str) -> dict:
        """Get profile from cache or fetch new"""
        import time
        
        now = time.time()
        if username in self.profile_cache:
            profile, timestamp = self.profile_cache[username]
            if now - timestamp < self.cache_ttl:
                return profile
        
        # Fetch new profile
        profile = self.personalization.get_user_profile(username)
        if profile:
            self.profile_cache[username] = (profile, now)
        
        return profile
    
    def _should_update_profile(self, username: str, message_count: int) -> bool:
        """Determine if we should trigger a profile update"""
        # Update every 10 messages
        return message_count > 0 and message_count % 10 == 0
    
    # ========================================================================
    # STEP 4: Modify your main chat method to include personalization
    # ========================================================================
    
    def chat_with_personalization(self, username: str, user_message: str, chat_id: str):
        """
        ENHANCED VERSION of your chat method with personalization
        
        Replace your existing chat method with this structure
        """
        
        # Check if user is asking about their resume
        resume_keywords = ['resume', 'cv', 'my application', 'job application', 
                          'my profile', 'career', 'how am i doing']
        
        if any(keyword in user_message.lower() for keyword in resume_keywords):
            # Handle resume question
            resume_summary = self.personalization.get_resume_summary_for_chat(username)
            
            # Return response with resume information
            return {
                "response": resume_summary,
                "is_recommendation": False,
                "timestamp": datetime.now().isoformat(),
                "personalized": True
            }
        
        # Get personalization context
        personalization_context = self.personalization.build_personalization_context(username)
        
        # Get chat history (your existing code)
        chat_history = self._get_chat_history(chat_id)
        
        # BUILD ENHANCED SYSTEM PROMPT with personalization
        system_prompt = f"""You are a friendly, helpful academic advisor chatbot for Indian students.

{personalization_context}

IMPORTANT: Use the personalization insights above to:
1. Match the user's communication style (formal/casual)
2. Adjust explanation depth based on their skill levels
3. Reference their topics of interest when relevant
4. Provide recommendations aligned with their personality traits
5. Be aware of their resume performance if applicable

Guidelines:
- Be conversational and supportive
- Provide actionable advice
- Ask clarifying questions when needed
- Reference past interactions naturally

Current conversation:
{chat_history}

Respond to the user's message naturally and helpfully."""

        # YOUR EXISTING LLM CALL (modify to use enhanced prompt)
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        
        assistant_response = response.content
        
        # Adapt response based on profile (optional enhancement)
        profile = self._get_cached_profile(username)
        if profile and profile.get("data_available"):
            assistant_response = self._adapt_response_style(
                assistant_response, 
                profile
            )
        
        # Save message to database (your existing code)
        self._save_message_to_db(chat_id, username, user_message, assistant_response)
        
        # Check if we should trigger profile update
        message_count = len(self._get_messages(chat_id))
        if self._should_update_profile(username, message_count):
            # Trigger async update (non-blocking)
            try:
                self.personalization.trigger_profile_update(username)
            except Exception as e:
                logger.warning(f"Profile update failed: {e}")
        
        return {
            "response": assistant_response,
            "is_recommendation": False,
            "timestamp": datetime.now().isoformat(),
            "personalized": profile is not None and profile.get("data_available", False)
        }
    
    def _adapt_response_style(self, response: str, profile: dict) -> str:
        """Adapt response based on user's communication style"""
        
        comm_style = profile.get("communication_style", {})
        formality = comm_style.get("formality", "mixed")
        
        # Make more formal if user prefers formal communication
        if formality == "formal":
            response = response.replace("Hey", "Hello")
            response = response.replace("!", ".")
            response = response.replace(" gonna ", " going to ")
            response = response.replace(" wanna ", " want to ")
        
        # Make more casual if user prefers casual
        elif formality == "casual":
            response = response.replace("Hello", "Hey")
            # Add friendly emojis if not present
            if "!" not in response and len(response) > 50:
                response += " 😊"
        
        return response
    
    # ========================================================================
    # STEP 5: Add personalized greeting method
    # ========================================================================
    
    def get_personalized_greeting(self, username: str) -> str:
        """Generate personalized greeting for user"""
        
        # Try to get personalized greeting
        try:
            greeting = self.personalization.get_personalized_greeting(username)
            return greeting
        except Exception as e:
            logger.warning(f"Failed to get personalized greeting: {e}")
            return f"Hi {username}! 👋 How can I help you today?"
    
    # ========================================================================
    # STEP 6: Add method to get user insights for UI
    # ========================================================================
    
    def get_user_insights_for_ui(self, username: str) -> dict:
        """Get formatted insights to display in chat UI"""
        
        profile = self._get_cached_profile(username)
        
        if not profile or not profile.get("data_available"):
            return {
                "has_insights": False,
                "message": "Chat with me more to unlock personalized insights!"
            }
        
        resume_insights = profile.get("resume_insights", {})
        
        insights = {
            "has_insights": True,
            "personality_type": self._format_personality_type(profile),
            "communication_preference": profile.get("communication_style", {}).get("formality", "mixed"),
            "topics_of_interest": profile.get("topics_of_interest", [])[:3],
            "resume_score": resume_insights.get("average_score", 0) if resume_insights.get("total_analyses", 0) > 0 else None,
            "total_interactions": profile.get("total_interactions", 0)
        }
        
        return insights
    
    def _format_personality_type(self, profile: dict) -> str:
        """Format personality type for display"""
        traits = profile.get("personality_traits", {})
        
        if not traits or all(v == 0.5 for v in traits.values()):
            return "Still learning about you..."
        
        # Find dominant traits
        high_traits = [k.replace('_', ' ').title() for k, v in traits.items() if v > 0.7]
        
        if high_traits:
            return f"You show strong {', '.join(high_traits[:2])}"
        else:
            return "Balanced learner"

# ============================================================================
# STEP 7: Update FastAPI endpoints
# ============================================================================

# Add new endpoint to your FastAPI app
@app.get("/user/{username}/insights")
async def get_user_insights(username: str):
    """Get personalization insights for UI display"""
    try:
        insights = chatbot.get_user_insights_for_ui(username)
        return insights
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return {
            "has_insights": False,
            "error": str(e)
        }

@app.get("/user/{username}/greeting")
async def get_greeting(username: str):
    """Get personalized greeting"""
    try:
        greeting = chatbot.get_personalized_greeting(username)
        return {"greeting": greeting}
    except Exception as e:
        logger.error(f"Error getting greeting: {e}")
        return {"greeting": f"Hi {username}! 👋"}

# ============================================================================
# STEP 8: Modify existing chat endpoint
# ============================================================================

# BEFORE (your existing code):
@app.post("/chat")
async def chat(request: ChatRequest):
    response = chatbot.chat(request.username, request.message, request.chat_id)
    return response

# AFTER (with personalization):
@app.post("/chat")
async def chat(request: ChatRequest):
    # Use the enhanced chat method
    response = chatbot.chat_with_personalization(
        request.username, 
        request.message, 
        request.chat_id
    )
    return response

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: User asks about their progress

User: "How am I doing with my career preparation?"

Chatbot: [Checks resume insights + personality]
Response: "Great question! Based on your resume analyses, you've improved 
significantly - your score went from 65% to 82%! 📈

Your analytical personality and interest in data science align well with 
your target roles. I recommend focusing on:
1. Adding more quantified achievements
2. Highlighting your ML projects
3. Expanding your technical skills section

Would you like specific advice on any of these areas?"
"""

"""
Example 2: Personalized college recommendations

User: "Suggest some colleges for me"

Chatbot: [Checks personality + communication style + past preferences]
- High openness → Suggests innovative programs
- Formal communication → Professional tone
- Interest in tech → CS-focused recommendations
- Resume shows strong academic performance → Top-tier colleges

Response: "Based on your interest in computer science and strong academic 
profile, I recommend these institutions:

1. IIT Delhi - BTech Computer Science
   [Details...]

Your analytical approach and technical aptitude would thrive in their 
research-focused environment."
"""

"""
Example 3: Adaptive communication

User A (Casual profile): "yo what's good with ML courses?"
Chatbot: "Hey! 👋 So you're interested in ML? Awesome choice! Here are 
some beginner-friendly courses..."

User B (Formal profile): "What are good machine learning courses?"
Chatbot: "Hello. I can recommend several excellent machine learning courses.
For your skill level, I suggest..."
"""

# ============================================================================
# TESTING THE INTEGRATION
# ============================================================================

"""
To test:

1. Start all services:
   - python main.py (port 8000)
   - python personalization_module.py (port 8001)

2. Create test conversations:
   POST http://localhost:8000/chat
   {
     "username": "test_user",
     "message": "Tell me about engineering colleges"
   }

3. Check personalization:
   GET http://localhost:8001/user/test_user/profile

4. Ask about resume (after uploading one):
   POST http://localhost:8000/chat
   {
     "username": "test_user",
     "message": "How is my resume doing?"
   }

5. Check insights in UI:
   GET http://localhost:8000/user/test_user/insights
"""

# ============================================================================
# MIGRATION CHECKLIST
# ============================================================================

"""
✅ Step 1: Add personalization_helper.py to your project
✅ Step 2: Import PersonalizationHelper in main.py
✅ Step 3: Initialize helper in chatbot __init__
✅ Step 4: Add personalization context to system prompt
✅ Step 5: Add resume question handling
✅ Step 6: Add profile caching
✅ Step 7: Add periodic profile updates
✅ Step 8: Add UI insights endpoint
✅ Step 9: Test with real users
✅ Step 10: Monitor and refine
"""

# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
1. Cache profiles for 5 minutes to reduce API calls
2. Make profile updates async/non-blocking
3. Gracefully handle personalization service downtime
4. Log personalization usage for monitoring
5. Set timeouts on personalization API calls (5 seconds)
"""

# ============================================================================
# ERROR HANDLING
# ============================================================================

def safe_get_personalization(username: str) -> dict:
    """Safely get personalization with fallback"""
    try:
        profile = personalization.get_user_profile(username)
        return profile if profile else {}
    except requests.exceptions.Timeout:
        logger.warning("Personalization service timeout")
        return {}
    except requests.exceptions.ConnectionError:
        logger.warning("Personalization service unavailable")
        return {}
    except Exception as e:
        logger.error(f"Unexpected personalization error: {e}")
        return {}