"""
Shared Database Manager for AI Chatbot and Resume Analyzer
API base: http://3.7.255.54:3003 → paths: /db/users, /db/interaction, etc.

Endpoint mapping (Swagger → code):
  Users:
    POST   /db/users               → create user
    GET    /db/users               → list all users
    GET    /db/users/details       → get single user by username (?username=)

  User Profile:
    POST   /db/user-profile        → create profile
    GET    /db/user-profile        → get profile by username (?username=)
    PUT    /db/user-profile        → update profile by username (?username=)

  Personalization Report:
    POST   /db/personalization-report         → create report
    GET    /db/personalization-report         → get all reports by username (?username=)
    GET    /db/personalization-report/details → get single report by report_id

  Interaction:
    POST   /db/interaction         → create new session
    GET    /db/interaction         → get all sessions by username (?username=)
    PUT    /db/interaction         → update session data completely
    DELETE /db/interaction         → delete session
    GET    /db/interaction/details → get single session (?username=&session_id=)
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

EXTERNAL_DB_API_URL = os.getenv("EXTERNAL_DB_API_URL", "http://3.7.255.54:3003")
USE_EXTERNAL_API = os.getenv("USE_EXTERNAL_API", "true").lower() in ("true", "1", "yes")
DEFAULT_TIMEOUT = 15

# Kept for backward-compat imports in personalization_module.py
API_PATH_PREFIX = "db"


def _path(p: str) -> str:
    """Build a /db/<p> path."""
    return f"/db/{p}"


# ---------------------------------------------------------------------------
# Local fallback storage
# ---------------------------------------------------------------------------

def _project_dir() -> Path:
    """Project root = directory containing shared_database.py"""
    return Path(__file__).resolve().parent


class LocalStorageBackend:
    """Local JSON file storage used only when external API is unavailable."""

    def __init__(self, storage_dir: str = "shared_data"):
        # Always use path relative to project directory
        self.storage_dir = (_project_dir() / storage_dir).resolve()
        self.users_file = self.storage_dir / "users.json"
        self.interactions_file = self.storage_dir / "interactions.json"
        self.profiles_file = self.storage_dir / "user_profiles.json"
        self.reports_file = self.storage_dir / "personalization_reports.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._init_files()

    def _init_files(self):
        for f, default in [
            (self.users_file, {}),
            (self.interactions_file, {}),
            (self.profiles_file, {}),
            (self.reports_file, {}),
        ]:
            if not f.exists():
                self._save_json(f, default)

    def _load_json(self, filepath: Path) -> dict:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}

    def _save_json(self, filepath: Path, data: dict):
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")


# ---------------------------------------------------------------------------
# Main database class
# ---------------------------------------------------------------------------

class SharedDatabase:
    """
    Talks to the external REST API first; silently falls back to local JSON
    files on 404 / connection errors so the app keeps working.
    """

    def __init__(self, storage_dir: str = None):
        self.storage_dir_str = storage_dir or "shared_data"
        self.base_url = EXTERNAL_DB_API_URL.rstrip("/")
        self._local = LocalStorageBackend(self.storage_dir_str)
        self._use_external = USE_EXTERNAL_API and requests is not None

        # Expose paths for legacy code that may reference them directly
        self.storage_dir = self._local.storage_dir
        self.users_file = self._local.users_file
        self.interactions_file = self._local.interactions_file
        self.profiles_file = self._local.profiles_file
        self.reports_file = self._local.reports_file

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict = None) -> Optional[Any]:
        if not self._use_external:
            return None
        try:
            url = f"{self.base_url}{path}"
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return response.json() if response.text.strip() else {}
            if response.status_code == 404:
                logger.debug(f"GET {path} returned 404 – using local fallback")
            else:
                logger.warning(f"GET {path} returned {response.status_code}")
            return None
        except Exception as e:
            logger.debug(f"GET {path} failed: {e} – using local fallback")
            return None

    def _post(self, path: str, data: dict) -> Optional[dict]:
        """Returns parsed response body on success, None on failure."""
        if not self._use_external:
            return None
        try:
            url = f"{self.base_url}{path}"
            response = requests.post(
                url, json=data, timeout=DEFAULT_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code in (200, 201):
                try:
                    return response.json()
                except Exception:
                    return {}
            if response.status_code == 404:
                logger.info(
                    f"External API 404 – using local storage | URL: {url} | "
                    "Check EXTERNAL_DB_API_URL and that backend exposes POST /db/interaction"
                )
            else:
                logger.warning(f"POST {path} returned {response.status_code}: {response.text[:300]}")
            return None
        except Exception as e:
            logger.debug(f"POST {path} failed: {e} – using local fallback")
            return None

    def _put(self, path: str, data: dict, params: dict = None) -> bool:
        if not self._use_external:
            return False
        try:
            url = f"{self.base_url}{path}"
            response = requests.put(
                url, json=data, params=params, timeout=DEFAULT_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code in (200, 201, 204):
                return True
            if response.status_code == 404:
                logger.info(f"External API {path} not found (404) – using local storage")
            else:
                logger.warning(f"PUT {path} returned {response.status_code}: {response.text[:300]}")
            return False
        except Exception as e:
            logger.debug(f"PUT {path} failed: {e} – using local fallback")
            return False

    def _delete(self, path: str, params: dict = None) -> bool:
        if not self._use_external:
            return False
        try:
            response = requests.delete(
                f"{self.base_url}{path}", params=params, timeout=DEFAULT_TIMEOUT
            )
            return response.status_code in (200, 204)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # User Management
    #   POST /db/users              – create
    #   GET  /db/users              – list all
    #   GET  /db/users/details      – get single by ?username=
    # ------------------------------------------------------------------

    def get_or_create_user(self, username: str) -> dict:
        # Try to fetch existing user via /db/users/details?username=
        result = self._get(_path("users/details"), params={"username": username})
        if result:
            data = result.get("data", result) if isinstance(result, dict) else result
            if isinstance(data, dict) and "username" in data:
                return data
            if isinstance(result, dict) and "username" in result:
                return result

        # Not found – create via POST /db/users
        user_payload = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "modules_used": [],
            "session_ids": {"chatbot": [], "resume_analyzer": []},
            "total_interactions": 0,
        }
        resp = self._post(_path("users"), user_payload)
        if resp is not None:
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            if isinstance(data, dict) and "username" in data:
                logger.info(f"Created user {username} via external API")
                return data

        # Fall back to local
        users = self._local._load_json(self._local.users_file)
        if username not in users:
            users[username] = {
                "username": username,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "modules_used": [],
                "session_ids": {"chatbot": [], "resume_analyzer": []},
                "total_interactions": 0,
            }
            self._local._save_json(self._local.users_file, users)
        return users[username]

    def update_user_modules(self, username: str, module: str):
        users = self._local._load_json(self._local.users_file)
        if username in users and module not in users[username].get("modules_used", []):
            users[username].setdefault("modules_used", []).append(module)
            users[username]["updated_at"] = datetime.now().isoformat()
            self._local._save_json(self._local.users_file, users)
        return True

    def add_session_to_user(self, username: str, module: str, session_id: str):
        users = self._local._load_json(self._local.users_file)
        if username in users:
            session_ids = users[username].setdefault(
                "session_ids", {"chatbot": [], "resume_analyzer": []}
            )
            module_sessions = session_ids.setdefault(module, [])
            if session_id not in module_sessions:
                module_sessions.append(session_id)
                users[username]["total_interactions"] = (
                    users[username].get("total_interactions", 0) + 1
                )
                users[username]["updated_at"] = datetime.now().isoformat()
                self.update_user_modules(username, module)
                self._local._save_json(self._local.users_file, users)
        return True

    def get_user_sessions(self, username: str, module: Optional[str] = None) -> List[str]:
        user = self.get_or_create_user(username)
        if module:
            return user.get("session_ids", {}).get(module, [])
        return [s for sessions in user.get("session_ids", {}).values() for s in sessions]

    def get_all_users(self) -> List[str]:
        result = self._get(_path("users"))
        if result is not None:
            if isinstance(result, list):
                return [u.get("username", u) if isinstance(u, dict) else u for u in result]
            data = result.get("data", result.get("users", []))
            if isinstance(data, list):
                return [u.get("username", u) if isinstance(u, dict) else u for u in data]
        return list(self._local._load_json(self._local.users_file).keys())

    # ------------------------------------------------------------------
    # Interaction (Session) Management
    #   POST   /db/interaction         – create new session
    #   GET    /db/interaction         – get all by username (?username=&module=)
    #   PUT    /db/interaction         – update session (?username=&session_id=)
    #   DELETE /db/interaction         – delete session (?username=&session_id=)
    #   GET    /db/interaction/details – get single (?username=&session_id=)
    # ------------------------------------------------------------------

    def save_interaction(self, username: str, module: str, session_id: str, interaction_data: dict):
        """Create or update an interaction/session via the external API."""
        payload = {
            "username": username,
            "module": module,
            "session_id": session_id,
            "data": interaction_data,
        }

        # Check whether the session already exists so we choose PUT vs POST
        existing = self._get(
            _path("interaction/details"),
            params={"username": username, "session_id": session_id},
        )

        if existing:
            api_ok = self._put(
                _path("interaction"), payload,
                params={"username": username, "session_id": session_id},
            )
            verb = "updated"
        else:
            api_ok = self._post(_path("interaction"), payload) is not None
            verb = "created"

        if api_ok:
            logger.info(
                f"Interaction {verb} for {username}/{module}/{session_id} via external API"
            )
            return  # Done – no local write needed

        # External API unavailable → local fallback (no exception raised)
        interactions = self._local._load_json(self._local.interactions_file)
        key = f"{username}:{module}:{session_id}"
        if key not in interactions:
            interactions[key] = {
                "username": username,
                "module": module,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "data": {},
            }
        interactions[key]["updated_at"] = datetime.now().isoformat()
        interactions[key]["data"] = interaction_data
        self._local._save_json(self._local.interactions_file, interactions)
        self.add_session_to_user(username, module, session_id)
        logger.info(
            f"Saved {module} interaction for {username} via local storage (API unavailable)"
        )

    def get_interaction(self, username: str, module: str, session_id: str) -> Optional[dict]:
        result = self._get(
            _path("interaction/details"),
            params={"username": username, "session_id": session_id},
        )
        if result is not None:
            return result.get("data", result) if isinstance(result, dict) else result

        interactions = self._local._load_json(self._local.interactions_file)
        key = f"{username}:{module}:{session_id}"
        return interactions.get(key)

    def get_user_interactions(self, username: str, module: Optional[str] = None) -> List[dict]:
        params: dict = {"username": username}
        if module:
            params["module"] = module
        result = self._get(_path("interaction"), params=params)
        if result is not None:
            items = (
                result if isinstance(result, list)
                else result.get("data", result.get("interactions", []))
            )
            if isinstance(items, list):
                return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

        # Local fallback
        interactions = self._local._load_json(self._local.interactions_file)
        user_interactions = [
            v for v in interactions.values()
            if v.get("username") == username
            and (module is None or v.get("module") == module)
        ]
        return sorted(user_interactions, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_interaction(self, username: str, module: str, session_id: str):
        self._delete(
            _path("interaction"),
            params={"username": username, "session_id": session_id},
        )
        interactions = self._local._load_json(self._local.interactions_file)
        users = self._local._load_json(self._local.users_file)
        key = f"{username}:{module}:{session_id}"
        if key in interactions:
            del interactions[key]
            self._local._save_json(self._local.interactions_file, interactions)
        if username in users:
            module_sessions = users[username].get("session_ids", {}).get(module, [])
            if session_id in module_sessions:
                module_sessions.remove(session_id)
                users[username]["total_interactions"] = max(
                    0, users[username].get("total_interactions", 0) - 1
                )
                self._local._save_json(self._local.users_file, users)

    # ------------------------------------------------------------------
    # Chatbot helpers
    # ------------------------------------------------------------------

    def save_chatbot_conversation(self, username: str, chat_id: str, conversation_data: dict):
        self.save_interaction(username, "chatbot", chat_id, {
            "title": conversation_data.get("title", "New Conversation"),
            "messages": conversation_data.get("messages", []),
            "preferences": conversation_data.get("preferences", {}),
            "message_count": len(conversation_data.get("messages", [])),
            "last_message_at": datetime.now().isoformat(),
        })

    def get_chatbot_conversation(self, username: str, chat_id: str) -> Optional[dict]:
        interaction = self.get_interaction(username, "chatbot", chat_id)
        if interaction:
            return interaction.get("data", interaction)
        return None

    def get_user_chatbot_conversations(self, username: str) -> List[dict]:
        interactions = self.get_user_interactions(username, "chatbot")
        return [{
            "chat_id": i.get("session_id", ""),
            "username": username,
            "created_at": i.get("created_at", ""),
            "updated_at": i.get("updated_at", ""),
            "title": i.get("data", {}).get("title", ""),
            "messages": i.get("data", {}).get("messages", []),
            "preferences": i.get("data", {}).get("preferences", {}),
            "message_count": i.get("data", {}).get("message_count", 0),
        } for i in interactions]

    # ------------------------------------------------------------------
    # Resume Analyzer helpers
    # ------------------------------------------------------------------

    def save_resume_analysis(self, username: str, analysis_id: str, analysis_data: dict):
        ar = analysis_data.get("analysis_result", {})
        strengths = [s.get("strength", "") for s in ar.get("strengths_analysis", [])[:5]]
        weaknesses = [w.get("weakness", "") for w in ar.get("weaknesses_analysis", [])[:5]]
        pp = ar.get("executive_summary", {}).get("professional_profile", {})
        self.save_interaction(username, "resume_analyzer", analysis_id, {
            "target_role": analysis_data.get("target_role", ""),
            "overall_score": analysis_data.get("overall_score", 0),
            "recommendation_level": analysis_data.get("recommendation_level", ""),
            "analysis_result": ar,
            "uploaded_at": analysis_data.get("uploaded_at", datetime.now().isoformat()),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "professional_profile": pp,
            "technical_skills_count": pp.get("technical_skills_count", 0),
            "experience_level": pp.get("experience_level", ""),
            "achievement_metrics": pp.get("achievement_metrics", ""),
        })

    def get_resume_analysis(self, username: str, analysis_id: str) -> Optional[dict]:
        interaction = self.get_interaction(username, "resume_analyzer", analysis_id)
        return interaction.get("data", interaction) if interaction else None

    def get_user_resume_analyses(self, username: str) -> List[dict]:
        interactions = self.get_user_interactions(username, "resume_analyzer")
        return [{
            "analysis_id": i.get("session_id", ""),
            "username": username,
            "created_at": i.get("created_at", ""),
            "updated_at": i.get("updated_at", ""),
            "target_role": i.get("data", {}).get("target_role", ""),
            "overall_score": i.get("data", {}).get("overall_score", 0),
            "recommendation_level": i.get("data", {}).get("recommendation_level", ""),
            "analysis_result": i.get("data", {}).get("analysis_result", {}),
            "uploaded_at": i.get("data", {}).get("uploaded_at", ""),
            "strengths": i.get("data", {}).get("strengths", []),
            "weaknesses": i.get("data", {}).get("weaknesses", []),
        } for i in interactions]

    # ------------------------------------------------------------------
    # User Profile
    #   POST /db/user-profile  – create (?username=)
    #   GET  /db/user-profile  – get by username (?username=)
    #   PUT  /db/user-profile  – update by username (?username=)
    # ------------------------------------------------------------------

    def save_user_profile(self, username: str, profile_data: dict):
        payload = {
            "username": username,
            "profile": {**profile_data, "updated_at": datetime.now().isoformat()},
        }
        # Try PUT first (update existing); fall back to POST (create new)
        ok = self._put(_path("user-profile"), payload, params={"username": username})
        if not ok:
            ok = self._post(_path("user-profile"), payload) is not None
        if not ok:
            profiles = self._local._load_json(self._local.profiles_file)
            profiles[username] = {
                "username": username,
                "updated_at": datetime.now().isoformat(),
                "profile": profile_data,
            }
            self._local._save_json(self._local.profiles_file, profiles)

    def get_user_profile(self, username: str) -> Optional[dict]:
        result = self._get(_path("user-profile"), params={"username": username})
        if result is not None:
            data = result.get("data", result) if isinstance(result, dict) else result
            if isinstance(data, dict):
                return data.get("profile", data)
        profiles = self._local._load_json(self._local.profiles_file)
        return profiles.get(username, {}).get("profile")

    # ------------------------------------------------------------------
    # Personalization Report
    #   POST /db/personalization-report         – create
    #   GET  /db/personalization-report         – all by username (?username=)
    #   GET  /db/personalization-report/details – single by report_id
    # ------------------------------------------------------------------

    def save_personalization_report(self, username: str, report_data: dict):
        report_id = report_data.get(
            "report_id",
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        payload = {
            "username": username,
            "report_id": report_id,
            "generated_at": report_data.get("generated_at", datetime.now().isoformat()),
            "report": report_data,
        }
        ok = self._post(_path("personalization-report"), payload) is not None
        if not ok:
            reports = self._local._load_json(self._local.reports_file)
            if username not in reports:
                reports[username] = []
            reports[username].insert(0, {
                "report_id": report_id,
                "generated_at": payload["generated_at"],
                "report": report_data,
            })
            reports[username] = reports[username][:10]
            self._local._save_json(self._local.reports_file, reports)

    def get_latest_personalization_report(self, username: str) -> Optional[dict]:
        result = self._get(_path("personalization-report"), params={"username": username})
        if result is not None:
            items = (
                result if isinstance(result, list)
                else result.get("data", result.get("reports", []))
            )
            if isinstance(items, list) and items:
                first = items[0]
                return first.get("report", first) if isinstance(first, dict) else first
        reports = self._local._load_json(self._local.reports_file)
        user_reports = reports.get(username, [])
        return user_reports[0]["report"] if user_reports else None

    def get_all_personalization_reports(self, username: str) -> List[dict]:
        result = self._get(_path("personalization-report"), params={"username": username})
        if result is not None:
            return (
                result if isinstance(result, list)
                else result.get("data", result.get("reports", []))
            )
        return self._local._load_json(self._local.reports_file).get(username, [])

    # ------------------------------------------------------------------
    # Resume Insights (derived from interactions)
    # ------------------------------------------------------------------

    def get_resume_insights(self, username: str) -> Dict[str, Any]:
        analyses = self.get_user_resume_analyses(username)
        if not analyses:
            return {
                "total_analyses": 0, "average_score": 0, "target_roles": [],
                "improvement_trend": "No data", "common_strengths": [],
                "common_weaknesses": [], "technical_skills_trend": 0,
                "experience_levels": [], "analyses_history": [],
            }
        total = len(analyses)
        scores = [a.get("overall_score", 0) for a in analyses]
        avg = sum(scores) / len(scores) if scores else 0
        target_roles = list({a.get("target_role", "") for a in analyses if a.get("target_role")})
        all_s = [x for a in analyses for x in a.get("strengths", [])]
        all_w = [x for a in analyses for x in a.get("weaknesses", [])]
        trend = (
            "Improving" if len(scores) >= 2 and scores[0] > scores[-1]
            else "Declining" if len(scores) >= 2 and scores[0] < scores[-1]
            else "Insufficient data"
        )
        return {
            "total_analyses": total,
            "average_score": round(avg, 1),
            "latest_score": scores[0] if scores else 0,
            "target_roles": target_roles,
            "improvement_trend": trend,
            "common_strengths": [s for s, _ in Counter(all_s).most_common(5)],
            "common_weaknesses": [w for w, _ in Counter(all_w).most_common(5)],
            "technical_skills_trend": sum(
                a.get("technical_skills_count", 0) for a in analyses
            ) / total,
            "experience_levels": [a.get("experience_level", "") for a in analyses],
            "analyses_history": [
                {
                    "date": a.get("created_at", ""),
                    "role": a.get("target_role", ""),
                    "score": a.get("overall_score", 0),
                }
                for a in analyses[:5]
            ],
        }

    # ------------------------------------------------------------------
    # Export & Stats
    # ------------------------------------------------------------------

    def export_user_data_for_personalization(self, username: str) -> Dict[str, Any]:
        user = self.get_or_create_user(username)
        chatbot_interactions = self.get_user_interactions(username, "chatbot")
        resume_analyses = self.get_user_interactions(username, "resume_analyzer")
        all_messages = []
        for i in chatbot_interactions:
            for msg in i.get("data", {}).get("messages", []):
                all_messages.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp", ""),
                    "is_recommendation": msg.get("is_recommendation", False),
                })
        return {
            "username": username,
            "user_info": user,
            "all_messages": all_messages,
            "chatbot_interactions": chatbot_interactions,
            "resume_analyses": resume_analyses,
            "resume_insights": self.get_resume_insights(username),
            "existing_profile": self.get_user_profile(username),
            "latest_report": self.get_latest_personalization_report(username),
            "total_messages": len(all_messages),
            "total_analyses": len(resume_analyses),
            "modules_used": user.get("modules_used", []),
        }

    def get_user_stats(self, username: str) -> Dict[str, Any]:
        user = self.get_or_create_user(username)
        chatbot_sessions = len(self.get_user_sessions(username, "chatbot"))
        resume_sessions = len(self.get_user_sessions(username, "resume_analyzer"))
        resume_insights = self.get_resume_insights(username)
        latest_report = self.get_latest_personalization_report(username)
        return {
            "username": username,
            "created_at": user.get("created_at", ""),
            "updated_at": user.get("updated_at", user.get("created_at", "")),
            "modules_used": user.get("modules_used", []),
            "total_sessions": chatbot_sessions + resume_sessions,
            "chatbot_sessions": chatbot_sessions,
            "resume_analyzer_sessions": resume_sessions,
            "total_interactions": user.get("total_interactions", 0),
            "has_personalization_profile": self.get_user_profile(username) is not None,
            "latest_report_date": latest_report.get("generated_at") if latest_report else None,
            "resume_insights": resume_insights,
        }