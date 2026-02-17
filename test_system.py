"""
Testing Script for AI System with Personalization
Tests all three modules and their integrations
"""

import requests
import json
import time
from pathlib import Path

# ===== CONFIGURATION =====
CHATBOT_URL = "http://localhost:8000"
PERSONALIZATION_URL = "http://localhost:8001"
RESUME_ANALYZER_URL = "http://localhost:8002"

TEST_USERNAME = "test_user_demo"

# ===== COLOR CODES FOR OUTPUT =====
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    """Print info message"""
    print(f"{BLUE}ℹ️  {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {text}{RESET}")

# ===== HEALTH CHECKS =====

def check_service_health(service_name, url):
    """Check if a service is running"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print_success(f"{service_name} is running")
            return True
        else:
            print_error(f"{service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"{service_name} is not running at {url}")
        return False
    except Exception as e:
        print_error(f"{service_name} health check failed: {e}")
        return False

def run_health_checks():
    """Run health checks on all services"""
    print_header("Health Checks")
    
    services = [
        ("Chatbot", CHATBOT_URL),
        ("Personalization Module", PERSONALIZATION_URL),
        ("Resume Analyzer", RESUME_ANALYZER_URL)
    ]
    
    results = []
    for name, url in services:
        result = check_service_health(name, url)
        results.append(result)
        time.sleep(0.5)
    
    if all(results):
        print_success("\nAll services are healthy! ✨")
        return True
    else:
        print_error("\nSome services are not running. Please start them first.")
        return False

# ===== CHATBOT TESTS =====

def test_chatbot_basic():
    """Test basic chatbot functionality"""
    print_header("Testing Chatbot - Basic Conversation")
    
    test_messages = [
        "Hi, I'm interested in engineering colleges",
        "What are some good colleges in Bangalore?",
        "Tell me about IIT Delhi",
        "What about private universities?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print_info(f"Message {i}: {message}")
        
        try:
            response = requests.post(
                f"{CHATBOT_URL}/chat",
                json={
                    "username": TEST_USERNAME,
                    "message": message,
                    "chat_id": "test_chat_001"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Response: {data.get('response', '')[:100]}...")
            else:
                print_error(f"Failed with status {response.status_code}")
                
        except Exception as e:
            print_error(f"Request failed: {e}")
        
        time.sleep(1)

def test_chatbot_resume_question():
    """Test chatbot's ability to answer resume questions"""
    print_header("Testing Chatbot - Resume Question")
    
    print_info("Asking about resume performance...")
    
    try:
        response = requests.post(
            f"{CHATBOT_URL}/chat",
            json={
                "username": TEST_USERNAME,
                "message": "How is my resume performing?",
                "chat_id": "test_chat_002"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Got resume information from chatbot!")
            print(f"Response: {data.get('response', '')[:200]}...")
        else:
            print_error(f"Failed with status {response.status_code}")
            
    except Exception as e:
        print_error(f"Request failed: {e}")

# ===== RESUME ANALYZER TESTS =====

def test_resume_analyzer():
    """Test resume analyzer"""
    print_header("Testing Resume Analyzer")
    
    # Check if sample resume exists
    sample_resume_path = Path("sample_resume.pdf")
    
    if not sample_resume_path.exists():
        print_warning("No sample_resume.pdf found. Creating a test placeholder...")
        print_info("In real usage, provide an actual PDF resume file")
        return False
    
    print_info("Uploading resume for analysis...")
    
    try:
        with open(sample_resume_path, 'rb') as f:
            files = {'file': ('sample_resume.pdf', f, 'application/pdf')}
            data = {
                'username': TEST_USERNAME,
                'target_role': 'Software Engineer',
                'search_jobs': False,
                'location': 'India'
            }
            
            response = requests.post(
                f"{RESUME_ANALYZER_URL}/analyze-resume",
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print_success("Resume analyzed successfully!")
                print_info(f"Analysis ID: {result.get('analysis_id')}")
                print_info(f"Overall Score: {result.get('ai_insights', {}).get('overall_score', 'N/A')}%")
                print_info(f"Recommendation: {result.get('ai_insights', {}).get('recommendation_level', 'N/A')}")
                return True
            else:
                print_error(f"Analysis failed with status {response.status_code}")
                print_error(response.text[:200])
                return False
                
    except FileNotFoundError:
        print_warning("Sample resume file not found")
        return False
    except Exception as e:
        print_error(f"Analysis request failed: {e}")
        return False

def test_get_user_analyses():
    """Test fetching user's resume analyses"""
    print_header("Testing Resume Analyzer - Get User History")
    
    print_info(f"Fetching analyses for {TEST_USERNAME}...")
    
    try:
        response = requests.get(
            f"{RESUME_ANALYZER_URL}/user/{TEST_USERNAME}/analyses",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            total = data.get('total_analyses', 0)
            print_success(f"Found {total} analyses for user")
            
            if total > 0:
                print_info("Recent analyses:")
                for analysis in data.get('analyses', [])[:3]:
                    print(f"  - {analysis.get('target_role')} (Score: {analysis.get('overall_score')}%)")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

# ===== PERSONALIZATION TESTS =====

def test_personalization_profile():
    """Test personalization profile generation"""
    print_header("Testing Personalization - Profile Generation")
    
    print_info(f"Getting profile for {TEST_USERNAME}...")
    
    try:
        response = requests.get(
            f"{PERSONALIZATION_URL}/user/{TEST_USERNAME}/profile",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Profile retrieved successfully!")
            print_info(f"Total interactions: {data.get('total_interactions', 0)}")
            print_info(f"Data available: {data.get('data_available', False)}")
            print_info(f"Modules used: {', '.join(data.get('modules_used', []))}")
            
            resume_insights = data.get('resume_insights', {})
            if resume_insights.get('total_analyses', 0) > 0:
                print_info(f"Resume avg score: {resume_insights.get('average_score', 0)}%")
                print_info(f"Improvement trend: {resume_insights.get('improvement_trend', 'N/A')}")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_personalization_report():
    """Test personalization report generation"""
    print_header("Testing Personalization - Report Generation")
    
    print_info(f"Generating report for {TEST_USERNAME}...")
    
    try:
        response = requests.get(
            f"{PERSONALIZATION_URL}/user/{TEST_USERNAME}/report",
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Report generated successfully!")
            print_info(f"Personality type: {data.get('personality_type', 'N/A')}")
            print_info(f"Has data: {data.get('has_data', False)}")
            
            strengths = data.get('strengths', [])
            if strengths:
                print_info(f"Strengths: {len(strengths)} identified")
                for strength in strengths[:3]:
                    print(f"  - {strength}")
            
            improvements = data.get('areas_for_improvement', [])
            if improvements:
                print_info(f"Areas for improvement: {len(improvements)} identified")
            
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

def test_personalization_update():
    """Test triggering profile update"""
    print_header("Testing Personalization - Profile Update")
    
    print_info(f"Triggering profile update for {TEST_USERNAME}...")
    
    try:
        response = requests.post(
            f"{PERSONALIZATION_URL}/user/{TEST_USERNAME}/update",
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Profile updated successfully!")
            print_info(f"Success: {data.get('success', False)}")
            print_info(f"Message: {data.get('message', '')}")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False

# ===== INTEGRATION TESTS =====

def test_full_integration():
    """Test full integration workflow"""
    print_header("Testing Full Integration Workflow")
    
    print_info("This test simulates a complete user journey:\n")
    print("1. User chats with chatbot")
    print("2. User uploads resume")
    print("3. Personalization analyzes behavior")
    print("4. Chatbot uses personalization insights\n")
    
    # Step 1: Chat interactions
    print_info("Step 1: Creating chat interactions...")
    test_chatbot_basic()
    time.sleep(2)
    
    # Step 2: Upload resume (if available)
    print_info("\nStep 2: Uploading resume...")
    resume_uploaded = test_resume_analyzer()
    time.sleep(2)
    
    # Step 3: Trigger personalization update
    print_info("\nStep 3: Updating personalization profile...")
    test_personalization_update()
    time.sleep(2)
    
    # Step 4: Get updated profile
    print_info("\nStep 4: Fetching updated profile...")
    test_personalization_profile()
    time.sleep(2)
    
    # Step 5: Generate report
    print_info("\nStep 5: Generating personality report...")
    test_personalization_report()
    time.sleep(2)
    
    # Step 6: Ask chatbot about resume
    if resume_uploaded:
        print_info("\nStep 6: Asking chatbot about resume...")
        test_chatbot_resume_question()
    
    print_success("\n🎉 Full integration test completed!")

# ===== MAIN TEST RUNNER =====

def run_all_tests():
    """Run all tests"""
    print_header("AI System Integration Test Suite")
    print(f"Testing with username: {TEST_USERNAME}\n")
    
    # Health checks first
    if not run_health_checks():
        print_error("\nCannot proceed - services not running")
        print_info("\nPlease start all services:")
        print("  Terminal 1: python main.py")
        print("  Terminal 2: python personalization_module.py")
        print("  Terminal 3: python resume_analyzer.py")
        return
    
    time.sleep(1)
    
    # Individual component tests
    tests = [
        ("Chatbot Basic", test_chatbot_basic),
        ("Resume Analyzer", test_resume_analyzer),
        ("User Analyses", test_get_user_analyses),
        ("Personalization Profile", test_personalization_profile),
        ("Personalization Report", test_personalization_report),
        ("Profile Update", test_personalization_update),
        ("Chatbot Resume Question", test_chatbot_resume_question)
    ]
    
    results = []
    for test_name, test_func in tests:
        print_info(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print_error(f"Test failed with exception: {e}")
            results.append((test_name, False))
        time.sleep(1)
    
    # Print summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("\n🎉 All tests passed!")
    else:
        print_warning(f"\n⚠️  {total - passed} test(s) failed")

# ===== INTERACTIVE MENU =====

def interactive_menu():
    """Interactive test menu"""
    while True:
        print_header("AI System Test Menu")
        print("1. Run all tests")
        print("2. Health checks only")
        print("3. Test chatbot")
        print("4. Test resume analyzer")
        print("5. Test personalization")
        print("6. Full integration test")
        print("7. Quick smoke test")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "0":
            print_info("Goodbye!")
            break
        elif choice == "1":
            run_all_tests()
        elif choice == "2":
            run_health_checks()
        elif choice == "3":
            test_chatbot_basic()
            test_chatbot_resume_question()
        elif choice == "4":
            test_resume_analyzer()
            test_get_user_analyses()
        elif choice == "5":
            test_personalization_profile()
            test_personalization_report()
            test_personalization_update()
        elif choice == "6":
            test_full_integration()
        elif choice == "7":
            print_header("Quick Smoke Test")
            if run_health_checks():
                print_success("System is operational!")
        else:
            print_warning("Invalid choice")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "--health":
        run_health_checks()
    elif len(sys.argv) > 1 and sys.argv[1] == "--integration":
        test_full_integration()
    else:
        interactive_menu()