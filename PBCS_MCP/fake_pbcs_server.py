from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)

# -----------------------------
# In-memory "PBCS" state
# -----------------------------
JOB_DEFS = [
    {"jobType": "RULES", "jobName": "RollupUSSales", "description": "Aggregate US Sales"},
    {"jobType": "IMPORT_DATA", "jobName": "Import_GL", "description": "Import GL data file"},
    {"jobType": "EXPORT_DATA", "jobName": "Export_GL", "description": "Export GL data file"},
    {"jobType": "REFRESH_CUBE", "jobName": "RefreshPlan1", "description": "Refresh cube Plan1"},
]

JOBS: Dict[str, Dict[str, Any]] = {}
JOB_DETAILS: Dict[str, List[Dict[str, Any]]] = {}

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def require_auth() -> Optional[Any]:
    """
    Very light auth simulation:
    - If header X-Auth-Mode=failure401 -> return 401
    - If header X-Auth-Mode=failure403 -> return 403
    Otherwise allow.
    """
    mode = (request.headers.get("X-Auth-Mode") or "").lower()
    if mode == "failure401":
        return jsonify({"message": "Unauthorized (simulated)"}), 401
    if mode == "failure403":
        return jsonify({"message": "Forbidden (simulated)"}), 403
    return None

@app.get("/HyperionPlanning/rest/")
def versions():
    deny = require_auth()
    if deny: return deny
    return jsonify({
        "items": [
            {"version": "v3", "links": [{"rel": "self", "href": "/HyperionPlanning/rest/v3"}]}
        ]
    })

@app.get("/HyperionPlanning/rest/v3/applications/<appname>/jobdefinitions")
def jobdefinitions(appname: str):
    deny = require_auth()
    if deny: return deny
    return jsonify({"items": JOB_DEFS, "application": appname})

@app.post("/HyperionPlanning/rest/v3/applications/<appname>/jobs")
def execute_job(appname: str):
    deny = require_auth()
    if deny: return deny

    body = request.get_json(force=True) or {}
    job_type = body.get("jobType")
    job_name = body.get("jobName")
    params = body.get("parameters", {}) or {}

    # Validate like a real service would (basic)
    if not job_type or not job_name:
        return jsonify({"message": "jobType and jobName are required"}), 400

    # Simulate 429 throttling if asked
    if (request.headers.get("X-RateLimit") or "").lower() == "429":
        return jsonify({"message": "Too Many Requests (simulated)"}), 429

    # Create job
    job_id = str(int(time.time() * 1000))
    JOBS[job_id] = {
        "jobId": job_id,
        "jobType": job_type,
        "jobName": job_name,
        "status": "RUNNING",
        "descriptiveStatus": "RUNNING",
        "percentComplete": 0,
        "startTime": now_iso(),
        "application": appname,
        "parameters": params,
    }

    JOB_DETAILS[job_id] = [
        {"severity": "INFO", "type": "MESSAGE", "row": None, "message": f"Job {job_name} started."},
        {"severity": "INFO", "type": "MESSAGE", "row": None, "message": f"jobType={job_type} parameters={params}"},
    ]

    return jsonify({"jobId": job_id, "status": "RUNNING", "descriptiveStatus": "RUNNING"}), 201

@app.get("/HyperionPlanning/rest/v3/applications/<appname>/jobs/<job_id>")
def job_status(appname: str, job_id: str):
    deny = require_auth()
    if deny: return deny

    job = JOBS.get(job_id)
    if not job:
        return jsonify({"message": "Job not found"}), 404

    # Simulate progress each time someone polls
    if job["status"] == "RUNNING":
        job["percentComplete"] = min(100, int(job["percentComplete"]) + 35)

        # Optional: simulate failure if requested
        if (request.headers.get("X-Fail-Job") or "").lower() == "true" and job["percentComplete"] >= 70:
            job["status"] = "FAILED"
            job["descriptiveStatus"] = "FAILED"
            job["endTime"] = now_iso()
            JOB_DETAILS[job_id].append({"severity": "ERROR", "type": "MESSAGE", "row": None, "message": "Simulated failure occurred."})
        elif job["percentComplete"] >= 100:
            job["status"] = "SUCCEEDED"
            job["descriptiveStatus"] = "SUCCEEDED"
            job["endTime"] = now_iso()
            JOB_DETAILS[job_id].append({"severity": "INFO", "type": "MESSAGE", "row": None, "message": "Job completed successfully."})

    return jsonify(job)

@app.get("/HyperionPlanning/rest/v3/applications/<appname>/jobs/<job_id>/details")
def job_details(appname: str, job_id: str):
    deny = require_auth()
    if deny: return deny

    items = JOB_DETAILS.get(job_id)
    if items is None:
        return jsonify({"message": "Job not found"}), 404

    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 200))
    page = items[offset: offset + limit]

    return jsonify({
        "items": page,
        "offset": offset,
        "limit": limit,
        "count": len(page),
        "hasMore": (offset + limit) < len(items),
    })

if __name__ == "__main__":
    # Local fake PBCS REST
    app.run(host="127.0.0.1", port=9010, debug=False)
