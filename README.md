# Job Queue Architecture

This folder documents the two-tier inference routing and queue management system for notmyvoice.ai. This was infrastructure used in production for 1,000+ users.

This is not the full code for NotMyVoice.ai, but rather excerpts relevant to this specific architecture.

Any questions? hi[!]samir.cv

---

## What it does

Every voice conversion request passes through a routing layer that decides whether to run inference on a **Modal serverless GPU** (paid tier) or a **self-hosted VPS** (free tier). The VPS uses a multiprocessing queue to serialize jobs, while Modal handles paid requests in parallel with no queue wait.

---

## Files

| File | Source | Purpose |
|---|---|---|
| `modal_worker.py` | `lol/RVC MODAL.py` | Modal serverless GPU worker — spawned for paid-tier jobs |
| `vps_worker.py` | `lol/RVC VPS.py` | Self-hosted VPS worker with multiprocessing queue — handles free-tier jobs |
| `routing_and_callbacks.py` | `site/app.py` (excerpts) | Credit accounting, routing decision, and Flask callback endpoints |

---

## Architecture overview

```
User submits audio
        │
        ▼
 User.charge(1)
        │
        ├─ conversion_type = "paid"  ──────────────────────────────────────────►  Modal
        │   (permanent or expiring credits)                                        modal.Function.lookup('rvc', 'interface').spawn(...)
        │                                                                           modal_entity = FunctionCall.object_id
        │
        └─ conversion_type = "free"  ──► uuid gets "free-" prefix
            (daily free credits)
                    │
                    ▼
        POST https://california.notmyvoice.ai/interface
                    │
                    ├─ success ──────────────────────────────────────────────────►  VPS queue
                    │           modal_entity = "httprequest"                        multiprocessing.Queue
                    │                                                               serial worker process
                    │
                    └─ connection error ─────────────────────────────────────────►  Modal (fallback)
                                        modal_entity = FunctionCall.object_id
```

---

## Callback / status update flow

Both Modal and VPS workers receive an `updateLink` — a URL pointing to the Flask server's `POST /conversionUpdate/<id>` endpoint. Workers POST progress updates as JSON throughout the job lifecycle:

```
Worker (Modal or VPS)
    │
    │  POST updateLink
    │  {"update": {"status": "🚀 Our AI model is doing its work...", "description": "..."}}
    │
    ▼
Flask: POST /conversionUpdate/<id>
    Writes JSON to Conversion.status in DB
    If status == "Done": writes CDN URLs to Conversion.downloads
    │
    ▼
Client polls: GET /conversionUpdate/<id>
    Returns current Conversion.status as JSON
    Also directly probes Modal FunctionCall for silent ERROR returns
    On error: refunds 1 credit to user
```

---

## Cost optimization logic

The routing decision is purely based on which credit bucket was charged:

- **Paid credits** (`_credits` or `expiring_credits`) → Modal (billed per-second GPU time, but no queue delay for the user)
- **Free credits** (`FreeCredits` table, 3/day per user) → VPS first, Modal as fallback

The VPS is significantly cheaper to run — it's a single always-on server. Free users share its single worker queue. Queue position is broadcast to all waiting jobs after each job completes.

**Anti-gaming:** Free credits are tracked across user ID, IP, session cookie, and browser fingerprint. A user creating multiple accounts doesn't get extra free credits.

---

## VPS queue details

- Implemented with `multiprocessing.Queue` + a single `multiprocessing.Process` worker
- Worker is lazily started on first request and auto-restarts if it dies
- Timeout: **300s** if the model isn't cached locally (includes S3 download), **90s** otherwise
- After each job: broadcasts updated queue positions to all remaining jobs via their `updateLink`
- After each job: deletes the completed session folder and any `queue-*` temp files older than 1 hour

---

## Key identifiers

| Value | Meaning |
|---|---|
| `unique` starting with `"free-"` | Job was charged to free credits → goes to VPS |
| `modal_entity = "httprequest"` | Job was successfully queued on the VPS |
| `modal_entity = <object_id>` | Job is a Modal FunctionCall — can be probed directly |
