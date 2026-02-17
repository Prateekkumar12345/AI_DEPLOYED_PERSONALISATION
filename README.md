# AI System with Unified Personalization

## 🎯 Overview

This system integrates three modules with a unified database and bi-directional personalization:

1. **Academic Chatbot** - Helps students with college recommendations and academic queries
2. **Resume Analyzer** - Analyzes resumes and provides detailed feedback
3. **Personalization Module** - Analyzes user behavior and provides personality insights

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED DATABASE                          │
│                    (shared_data/)                           │
│  • users.json - User accounts                              │
│  • interactions.json - All user interactions               │
│  • user_profiles.json - Personality profiles               │
│  • personalization_reports.json - Analysis reports         │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐        ┌─────┴─────┐
    │ Chatbot │         │ Resume  │        │Personal-  │
    │ :8000   │←────────│Analyzer │────────→│ization   │
    └─────────┘         │ :8002   │        │  :8001    │
         ↑              └─────────┘        └───────────┘
         │                                       │
         └───────── Personalization ─────────────┘
                      Insights Flow
```

## 📁 File Structure

```
project/
├── shared_data/                    # All data stored here
│   ├── users.json                 # User accounts
│   ├── interactions.json          # Chatbot + Resume interactions
│   ├── user_profiles.json         # Personality profiles
│   └── personalization_reports.json  # Generated reports
├── shared_database.py             # Database manager
├── personalization_module.py      # Personality analysis engine
├── personalization_helper.py      # Integration helper for modules
├── main.py                        # Academic Chatbot
├── interviewer.py                 # Resume Analyzer (your existing code)
├── setup.py                       # Setup script
├── .env                           # OpenAI API key
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install fastapi uvicorn openai langchain langchain_openai langchain_core python-dotenv pydantic requests PyPDF2

# Run setup
python setup.py
```

### 2. Configure API Key

Create `.env` file:
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Start Services

**Terminal 1 - Academic Chatbot:**
```bash
python main.py
# Access: http://localhost:8000
```

**Terminal 2 - Resume Analyzer:**
```bash
python interviewer.py
# Access: http://localhost:8002
```

**Terminal 3 - Personalization Module:**
```bash
python personalization_module.py
# Access: http://localhost:8001
```

## 🔗 Integration Guide

### Integrating Personalization into Your Existing Chatbot

#### Step 1: Import the Helper

Add to your `main.py`:

```python
from personalization_helper import PersonalizationHelper

# Initialize in your chatbot class
self.personalization = PersonalizationHelper(personalization_api_url="http://localhost:8001")
```

#### Step 2: Add Personalization Context to LLM Prompts

Before sending messages to LLM:

```python
def generate_response(self, username: str, user_message: str, chat_history: str):
    # Get personalization context
    personalization_context = self.personalization.build_personalization_context(username)
    
    # Add to your system prompt
    system_prompt = f"""You are a friendly academic chatbot.
    
{personalization_context}

Use the personalization insights above to tailor your responses to the user's:
- Communication style (formal/casual)
- Skill levels
- Topics of interest
- Learning preferences

Be natural and conversational while being helpful."""
    
    # Continue with your normal LLM call...
```

#### Step 3: Handle Resume Questions

Add this method to your chatbot:

```python
def handle_resume_question(self, username: str):
    """Handle when user asks about their resume"""
    summary = self.personalization.get_resume_summary_for_chat(username)
    return summary
```

Detect resume questions:

```python
# In your message processing
if any(keyword in user_message.lower() for keyword in ['resume', 'cv', 'my application', 'job application']):
    resume_info = self.handle_resume_question(username)
    # Include this in your response or context
```

#### Step 4: Personalized Greetings

```python
def greet_user(self, username: str):
    """Generate personalized greeting"""
    greeting = self.personalization.get_personalized_greeting(username)
    return greeting
```

#### Step 5: Trigger Profile Updates

After significant interactions:

```python
# After user has meaningful conversation
if message_count % 10 == 0:  # Every 10 messages
    self.personalization.trigger_profile_update(username)
```

### Complete Integration Example

```python
from personalization_helper import PersonalizationHelper

class AcademicChatbot:
    def __init__(self):
        # ... your existing init code ...
        self.personalization = PersonalizationHelper()
    
    def chat(self, username: str, message: str, chat_id: str):
        # Check if asking about resume
        if 'resume' in message.lower() or 'cv' in message.lower():
            resume_summary = self.personalization.get_resume_summary_for_chat(username)
            return {
                "response": resume_summary,
                "is_recommendation": False
            }
        
        # Get personalization context
        context = self.personalization.build_personalization_context(username)
        
        # Build prompt with context
        system_prompt = f"""You are a helpful academic advisor.
        
{context}

Adapt your communication style and recommendations based on the user's profile."""
        
        # ... continue with your LLM call ...
        
        # Optionally trigger profile update
        if should_update_profile:
            self.personalization.trigger_profile_update(username)
```

## 📊 Personalization Module API

### Endpoints

#### `GET /user/{username}/profile`
Get user personality profile with traits, communication style, resume insights.

**Response:**
```json
{
  "username": "john_doe",
  "total_interactions": 45,
  "personality_traits": {
    "openness": 0.8,
    "conscientiousness": 0.75,
    "extraversion": 0.6,
    "agreeableness": 0.7,
    "emotional_stability": 0.65
  },
  "communication_style": {
    "formality": "casual",
    "verbosity": "moderate",
    "questioning_style": "exploratory"
  },
  "topics_of_interest": ["engineering", "AI", "machine learning"],
  "skill_levels": {
    "technical_writing": "advanced",
    "career_planning": "intermediate"
  },
  "resume_insights": {
    "total_analyses": 3,
    "average_score": 78.5,
    "latest_score": 82,
    "improvement_trend": "Improving",
    "target_roles": ["Software Engineer", "Data Scientist"]
  },
  "recommendations": {
    "learning_style": ["Explore diverse subjects"],
    "career_guidance": ["Consider technical leadership roles"],
    "skill_development": ["Focus on project presentation"]
  }
}
```

#### `GET /user/{username}/report`
Generate comprehensive personality report.

#### `POST /user/{username}/update`
Trigger profile update with latest interaction data.

#### `GET /user/{username}/stats`
Get user statistics.

## 🎨 Personalization Features

### 1. Personality Analysis
- Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability)
- Communication style detection
- Learning preferences

### 2. Resume Performance Tracking
- Average scores across analyses
- Improvement trends
- Target role tracking
- Historical performance

### 3. Bi-directional Insights
- Chatbot gets personality insights → adapts responses
- Resume scores visible in chatbot → career advice
- Behavior patterns → personalized recommendations

### 4. Adaptive Responses
- Formal vs casual communication
- Detailed vs concise explanations
- Topic-specific suggestions
- Skill-level appropriate content

## 💡 Use Cases

### For Students

**Scenario 1: Career Guidance**
```
Student: "What should I study?"
Chatbot: [Checks personality + resume]
  - High openness → Suggests interdisciplinary fields
  - Resume shows coding skills → Recommends CS programs
  - Formal communication style → Professional tone
```

**Scenario 2: Resume Feedback**
```
Student: "How's my resume looking?"
Chatbot: [Fetches resume insights]
  "Your resume has improved from 65% to 82%! 📈
   Key strengths: Strong technical skills presentation
   Work on: Adding more quantified achievements
   
   Based on your interests in AI and your analytical personality,
   I recommend highlighting your ML projects more prominently."
```

### For the System

**Adaptive Learning:**
- Tracks which topics user explores most
- Adjusts recommendation complexity
- Learns communication preferences
- Monitors career development progress

## 🔧 Customization

### Modify Personality Analysis

Edit `personalization_module.py`:

```python
def _build_analysis_prompt(self, ...):
    # Customize the analysis criteria
    # Add domain-specific traits
    # Adjust scoring weights
```

### Add New Insights

In `PersonalizationHelper`:

```python
def get_custom_insight(self, username: str):
    profile = self.get_user_profile(username)
    # Extract and format custom insights
    return formatted_insight
```

### Customize Response Adaptation

In your chatbot:

```python
def adapt_response_style(self, response: str, profile: dict):
    formality = profile['communication_style']['formality']
    
    if formality == 'formal':
        # Make response more formal
        response = response.replace("Hey", "Hello")
        response = response.replace("!", ".")
    
    return response
```

## 🐛 Troubleshooting

### Personalization Module Not Responding

```bash
# Check if service is running
curl http://localhost:8001/health

# Restart service
python personalization_module.py
```

### No Personalization Data

```python
# User needs interactions first
# Have user:
# 1. Chat with chatbot (at least 5 messages)
# 2. Upload resume to analyzer
# 3. Trigger profile update

requests.post("http://localhost:8001/user/john_doe/update")
```

### Profile Not Updating

```python
# Manually trigger update
from personalization_helper import PersonalizationHelper
helper = PersonalizationHelper()
helper.trigger_profile_update("username")
```

## 📈 Data Flow

### 1. User Interaction
```
User → Chatbot/Resume Analyzer → SharedDatabase
```

### 2. Profile Generation
```
Personalization Module → Read from SharedDatabase → Analyze → Save Profile
```

### 3. Insight Delivery
```
Chatbot → Request Insights → Personalization Module → Return Profile
Chatbot → Adapt Response → User
```

## 🎯 Best Practices

1. **Trigger Updates Periodically**
   - After every 10 chatbot messages
   - After resume upload
   - Daily batch updates

2. **Cache Profile Data**
   - Cache for 5-10 minutes in chatbot
   - Reduce API calls
   - Improve response time

3. **Graceful Degradation**
   - If personalization unavailable → use default behavior
   - Never block user interaction
   - Log errors for debugging

4. **Privacy Considerations**
   - Store only necessary data
   - Allow profile deletion
   - Transparent about data usage

## 🔐 Security

- All data stored locally in `shared_data/`
- No external data transmission (except OpenAI API)
- User-specific data isolation
- API keys in `.env` file (never commit)

## 📝 API Integration Examples

### Python (requests)

```python
import requests

# Get profile
response = requests.get("http://localhost:8001/user/john_doe/profile")
profile = response.json()

# Update profile
requests.post("http://localhost:8001/user/john_doe/update")
```

### JavaScript (fetch)

```javascript
// Get profile
const profile = await fetch('http://localhost:8001/user/john_doe/profile')
  .then(r => r.json());

// Update profile
await fetch('http://localhost:8001/user/john_doe/update', {
  method: 'POST'
});
```

## 🚦 Status Monitoring

Check system health:

```bash
# All services
curl http://localhost:8000/health  # Chatbot
curl http://localhost:8001/health  # Personalization
curl http://localhost:8002/health  # Resume Analyzer (if applicable)
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## 🤝 Contributing

To extend the system:

1. Add new traits to personality analysis
2. Create custom insight extractors
3. Build domain-specific adapters
4. Enhance recommendation engine

## 📄 License

[Your License Here]

## 🆘 Support

For issues or questions:
- Check troubleshooting section
- Review API documentation at `/docs` endpoints
- Check logs in terminal outputs

---

**Version:** 3.0.0  
**Last Updated:** February 2026  
**Compatible Modules:** Academic Chatbot, Resume Analyzer, Personalization Module
