# Nova v200 — Render deployment
# Two processes:
#   web: FastAPI HTTP server (browser → /v2/create-session, /v2/vision-observe, /v2/memory)
#   worker: LiveKit Agent worker (joins rooms, speaks as Nova)
#           + maya_watch.py sidecar (Refael 2026-11-13: Maya pod watchdog + page keep-alive on the paid worker;
#             background, stdlib-only, touches nothing of Nova's, its death never takes the worker down)

web: uvicorn server:app --host 0.0.0.0 --port $PORT
worker: sh -c "python maya_watch.py & exec python agent.py start"
