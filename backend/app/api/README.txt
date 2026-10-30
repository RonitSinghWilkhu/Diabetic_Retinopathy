RetinaAI backend JWT protection changes

Files to replace in your project:
- predictions.py -> backend/app/api/predictions.py
- explanations.py -> backend/app/api/explanations.py
- chat.py -> backend/app/api/chat.py

Before replacing:
1. Stop the FastAPI server (Ctrl+C).
2. Back up the three existing files.
3. Copy these files to the exact locations above.
4. Restart FastAPI.

These changes reuse app.api.auth.get_current_user and require a valid Bearer token for prediction, Grad-CAM explanation, and educational chat endpoints. Registration, login, and /api/auth/me are unchanged.

Important frontend follow-up:
Your frontend must include `Authorization: Bearer <access_token>` on requests to:
- POST /api/predictions/
- POST /api/explanations/
- POST /api/chat/

If the frontend doesn't send that header, these features will return 401 even after a successful login. Send your current frontend App.jsx and RAGChat.jsx so the API calls can be updated consistently.

Validation performed: Python AST syntax parsing passed for the three modified files. This is not a full runtime/integration test.
