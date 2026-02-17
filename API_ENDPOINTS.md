# Expected External API Endpoints

The chatbot and resume analyzer store data in the external API at **http://3.7.255.54:3003/db** (direct `/db` paths, no `/api`). When the API returns 404 or fails, the app **falls back to local `shared_data/`** so it keeps working.

## Confirmed Endpoints (from Swagger docs)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /db/interaction | Create/update interaction (chatbot conversations, resume analyses) |
| PUT | /db/user-profile | Update user profile |
| POST | /db/personalization-report | Create personalization report |

## Expected GET Endpoints (required for full functionality)

The SharedDatabase also calls these endpoints for reads. If your backend uses different paths, update `shared_database.py`:

| Method | Endpoint | Query Params | Purpose |
|--------|----------|--------------|---------|
| GET | /db/interactions | username, module? | Get user interactions |
| GET | /db/interaction | username, module, session_id | Get single interaction |
| GET | /db/users | - | List all users |
| GET | /db/user | username | Get or create user |
| GET | /db/user-profile | username | Get user profile |
| GET | /db/personalization-reports | username | Get personalization reports |
| DELETE | /db/interaction | username, module, session_id | Delete interaction |
| POST | /db/user | body: user object | Create user (optional) |
| POST | /db/user/session | body: username, module, session_id | Add session (optional) |
| PUT | /db/user/{username}/modules | body: module | Update modules (optional) |
| GET | /db/user/sessions | username, module? | Get sessions (optional) |

## Configuration

Set in `.env` (no spaces around `=`):
```
EXTERNAL_DB_API_URL=http://3.7.255.54:3003
USE_EXTERNAL_API=true   # Set to false to skip API and use local storage only
API_PATH_PREFIX=db      # Path prefix; use empty string if your API uses /interaction instead of /db/interaction
```

**If user profiles or personalization reports are not stored on the external API:**  
Use the base URL **without** `/api` (e.g. `http://3.7.255.54:3003`). The app calls `/db/user-profile` and `/db/personalization-report`; if your backend serves these at the root (e.g. `http://host:3003/db/user-profile`), do not add `/api` to the base URL. Only add `/api` if your backend is mounted under `/api` (e.g. `http://host:3003/api/db/user-profile`).

## Fallback Behavior

- If the external API returns **404** or is unreachable, data is stored locally in `shared_data/` (next to the project files).
- This ensures the app continues to work while the backend is being configured.
- Once the backend is reachable and exposes the correct endpoints, data will flow there automatically.

## Troubleshooting 404 Errors

If you see `External API 404 – using local storage`:

1. **Verify the API is reachable** from the machine running the chatbot:
   ```bash
   curl -X POST "http://3.7.255.54:3003/db/interaction" -H "Content-Type: application/json" -d "{\"username\":\"test\",\"module\":\"chatbot\",\"session_id\":\"test-1\",\"data\":{}}"
   ```

2. **Base URL** – Default is `http://3.7.255.54:3003` (no `/api`). Paths are `/db/interaction`, `/db/users`, etc.

3. **Ensure the backend is running** and that the routes in your Swagger docs match what the app calls.

## Response Formats

- **GET /db/interactions**: Expects `[{username, module, session_id, created_at, updated_at, data}, ...]` or `{interactions: [...]}`
- **GET /db/interaction**: Expects single interaction object with `data` field
- **GET /db/user-profile**: Expects `{profile: {...}}` or the profile object directly
- **GET /db/personalization-reports**: Expects `[{report_id, generated_at, report}, ...]` or `{reports: [...]}`
