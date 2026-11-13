#!/usr/bin/env python3
"""maya_watch.py — Maya pod watchdog + page keep-alive, riding the always-on Nova worker (Refael, 2026-11-13:
"I pay Render already, use the worker for Maya"). Runs BESIDE agent.py (Procfile: `maya_watch.py & exec agent.py`),
touches nothing of Nova's, and dies silently without ever taking the worker down.

Every WATCH_EVERY_S (default 600):
  1. RunPod terminator, same rules as maya-lt's watchdog, by EXPLICIT pod id, never a blanket sweep:
     only pods whose name starts with one of MAYA_DOG_PREFIXES; RUNNING and older than MAYA_DOG_MAX_MIN,
     or costing more than MAYA_DOG_MAX_RATE/h, or anything RUNNING when the balance is under MAYA_DOG_FLOOR
     -> DELETE by id. Nova pods are NOT in the prefix list. No RUNPOD_API_KEY in env -> this part is skipped.
  2. GET MAYA_GW_PING (the widget gateway /health) so the free page never sleeps.
Stdlib only. Never prints keys."""
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

REST = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY", "")
EVERY = int(os.environ.get("WATCH_EVERY_S", "600"))
PREFIXES = tuple(x.strip() for x in os.environ.get("MAYA_DOG_PREFIXES", "maya,zp-,zb-,zv-,zf-,zc-,w3-,smslive-").split(",") if x.strip())
MAX_MIN = float(os.environ.get("MAYA_DOG_MAX_MIN", "240"))
MAX_RATE = float(os.environ.get("MAYA_DOG_MAX_RATE", "1.50"))
FLOOR = float(os.environ.get("MAYA_DOG_FLOOR", "2.0"))
PING = os.environ.get("MAYA_GW_PING", "https://refael-live-widget.onrender.com/health")


def req(method, url, body=None):
    r = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as f:
        return json.loads(f.read() or "{}")


def age_min(pod):
    for k in ("createdAt", "lastStartedAt"):
        v = str(pod.get(k) or "").replace(" UTC", "").replace("Z", "+00:00")
        m = re.match(r"^(\S+)[ T](\d\d:\d\d:\d\d)(\.\d+)?\s*([+-]\d\d):?(\d\d)?", v)
        if not m:
            continue
        t = datetime.fromisoformat(f"{m[1]}T{m[2]}{m[4] or '+00'}:{m[5] or '00'}")
        return (datetime.now(timezone.utc) - t.replace(tzinfo=t.tzinfo or timezone.utc)).total_seconds() / 60
    return None


def round_once():
    if KEY:
        bal = None
        try:
            bal = (req("POST", "https://api.runpod.io/graphql",
                       {"query": "query { myself { clientBalance } }"}).get("data") or {}).get("myself", {}).get("clientBalance")
        except Exception as e:
            print("maya_watch balance:", type(e).__name__, flush=True)
        try:
            # graphql, not rest.runpod.io: REST answered every round with an HTTPError from Render's network
            # (2026-11-13) while graphql went through, and runtime.uptimeInSeconds beats parsing createdAt anyway.
            d = req("POST", "https://api.runpod.io/graphql",
                    {"query": "query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }"})
            pods = ((d.get("data") or {}).get("myself") or {}).get("pods") or []
            for p in pods:
                name, pid = p.get("name") or "", p.get("id")
                if not pid or not name.startswith(PREFIXES):
                    continue
                if (p.get("desiredStatus") or p.get("status")) not in ("RUNNING", None):
                    continue
                up = (p.get("runtime") or {}).get("uptimeInSeconds")
                a = up / 60 if isinstance(up, (int, float)) else age_min(p)
                rate = float(p.get("costPerHr") or 0)
                why = ("over %d min" % MAX_MIN if a is not None and a > MAX_MIN else
                       "rate %.2f > %.2f" % (rate, MAX_RATE) if rate > MAX_RATE else
                       "balance %.2f under %.2f" % (bal, FLOOR) if isinstance(bal, (int, float)) and bal < FLOOR else "")
                if why:
                    print("maya_watch KILL", pid, name, why, flush=True)
                    try:
                        req("POST", "https://api.runpod.io/graphql",
                            {"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pid})
                    except Exception as e:
                        print("maya_watch kill failed", pid, type(e).__name__, flush=True)
                else:
                    print("maya_watch ok", pid, name, "age_min", None if a is None else int(a), flush=True)
        except Exception as e:
            print("maya_watch pods:", type(e).__name__, flush=True)
    if PING:
        try:
            with urllib.request.urlopen(PING, timeout=25) as f:
                f.read(200)
        except Exception as e:
            print("maya_watch ping:", type(e).__name__, flush=True)


if __name__ == "__main__":
    print("maya_watch up: every", EVERY, "s, prefixes", ",".join(PREFIXES), "runpod" if KEY else "NO RUNPOD KEY (ping only)", flush=True)
    while True:
        try:
            round_once()
        except Exception as e:
            print("maya_watch round:", type(e).__name__, flush=True)
        time.sleep(EVERY)
