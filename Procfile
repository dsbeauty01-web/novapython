# Nova v200 — Render deployment
# Two processes:
#   web: FastAPI HTTP server (browser → /v2/create-session, /v2/vision-observe, /v2/memory)
#   worker: LiveKit Agent worker (joins rooms, speaks as Nova)

web: uvicorn server:app --host 0.0.0.0 --port $PORT
worker: python agent.py start
