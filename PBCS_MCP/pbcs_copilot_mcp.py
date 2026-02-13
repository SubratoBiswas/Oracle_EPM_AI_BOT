from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional
import requests

# -----------------------------
# Config
# -----------------------------
def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def load_cfg() -> Dict[str, Any]:
    cfg = {
        "base_url": (os.getenv("PBCS_BASE_URL") or "").rstrip("/"),
        "application": os.getenv("PBCS_APPLICATION") or "Vision",
        "api_version": os.getenv("PBCS_API_VERSION") or "v3",
        "verify_ssl": env_bool("PBCS_VERIFY_SSL", True),
        # Optional key to prevent random callers when running remotely
        "client_api_key": os.getenv("MCP_CLIENT_API_KEY"),
        # Optional knobs to drive fake server behaviors
        "fake_auth_mode": os.getenv("FAKE_AUTH_MODE"),   # failure401 / failure403 / None
        "fake_rate_limit": os.getenv("FAKE_RATE_LIMIT"), # 429 / None
        "fake_fail_job": os.getenv("FAKE_FAIL_JOB"),     # true / None
    }
    if not cfg["base_url"]:
        raise RuntimeError("Missing env var PBCS_BASE_URL (use http://127.0.0.1:9010 for the fake server)")
    return cfg

def pbcs_request(cfg: Dict[str, Any], method: str, path: str, params=None, body=None) -> Dict[str, Any]:
    url = cfg["base_url"] + path
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    # Drive fake failure modes via headers (purely for POC)
    if cfg.get("fake_auth_mode"):
        headers["X-Auth-Mode"] = cfg["fake_auth_mode"]
    if cfg.get("fake_rate_limit"):
        headers["X-RateLimit"] = cfg["fake_rate_limit"]
    if cfg.get("fake_fail_job"):
        headers["X-Fail-Job"] = cfg["fake_fail_job"]

    try:
        r = requests.request(
            method.upper(),
            url,
            headers=headers,
            params=params,
            data=None if body is None else json.dumps(body),
            timeout=30,
            verify=cfg["verify_ssl"],
        )
        payload = r.json() if r.content else {}
    except Exception as e:
        return {"ok": False, "error": "NETWORK_OR_PARSE_ERROR", "message": str(e), "url": url}

    if r.status_code >= 300:
        return {"ok": False, "error": "HTTP_ERROR", "status_code": r.status_code, "response": payload}
    return {"ok": True, "status_code": r.status_code, "response": payload}

# -----------------------------
# Transform helpers (LLM-friendly)
# -----------------------------
def compact_job_defs(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = payload.get("items") or []
    defs = [{
        "jobType": i.get("jobType"),
        "jobName": i.get("jobName"),
        "description": i.get("description"),
    } for i in items]
    return {"count": len(defs), "jobDefinitions": defs}

def compact_job_submit(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"jobId": payload.get("jobId"), "status": payload.get("status") or payload.get("descriptiveStatus")}

def compact_job_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": payload.get("status") or payload.get("descriptiveStatus"),
        "percentComplete": payload.get("percentComplete"),
        "startTime": payload.get("startTime"),
        "endTime": payload.get("endTime"),
        "jobType": payload.get("jobType"),
        "jobName": payload.get("jobName"),
    }

def compact_job_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = payload.get("items") or []
    out = [{
        "severity": i.get("severity"),
        "type": i.get("type"),
        "row": i.get("row"),
        "message": i.get("message"),
    } for i in items]
    return {"count": len(out), "items": out, "hasMore": payload.get("hasMore", False)}

# -----------------------------
# Tools
# -----------------------------
def check_client_key(cfg: Dict[str, Any], args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if cfg.get("client_api_key") and args.get("client_api_key") != cfg["client_api_key"]:
        return {"ok": False, "error": "UNAUTHORIZED_CLIENT", "message": "Invalid MCP client API key."}
    return None

def tool_discover_versions(cfg: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    deny = check_client_key(cfg, args)
    if deny: return deny
    return pbcs_request(cfg, "GET", "/HyperionPlanning/rest/")

def tool_list_job_definitions(cfg: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    deny = check_client_key(cfg, args)
    if deny: return deny
    v = args.get("api_version") or cfg["api_version"]
    app = args.get("application") or cfg["application"]
    res = pbcs_request(cfg, "GET", f"/HyperionPlanning/rest/{v}/applications/{app}/jobdefinitions")
    if not res.get("ok"): return res
    return {"ok": True, **compact_job_defs(res["response"])}

def tool_execute_job(cfg: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    deny = check_client_key(cfg, args)
    if deny: return deny
    v = args.get("api_version") or cfg["api_version"]
    app = args.get("application") or cfg["application"]

    job_type = args.get("job_type")
    job_name = args.get("job_name")
    parameters = args.get("parameters") or {}

    if not job_type or not job_name:
        return {"ok": False, "error": "BAD_ARGUMENTS", "message": "job_type and job_name are required"}

    body = {"jobType": job_type, "jobName": job_name, "parameters": parameters}
    res = pbcs_request(cfg, "POST", f"/HyperionPlanning/rest/{v}/applications/{app}/jobs", body=body)
    if not res.get("ok"): return res
    return {"ok": True, **compact_job_submit(res["response"])}

def tool_get_job_status(cfg: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    deny = check_client_key(cfg, args)
    if deny: return deny
    v = args.get("api_version") or cfg["api_version"]
    app = args.get("application") or cfg["application"]
    job_id = args.get("job_id")
    if not job_id:
        return {"ok": False, "error": "BAD_ARGUMENTS", "message": "job_id is required"}
    res = pbcs_request(cfg, "GET", f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}")
    if not res.get("ok"): return res
    return {"ok": True, "jobId": job_id, **compact_job_status(res["response"])}

def tool_get_job_details(cfg: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    deny = check_client_key(cfg, args)
    if deny: return deny
    v = args.get("api_version") or cfg["api_version"]
    app = args.get("application") or cfg["application"]
    job_id = args.get("job_id")
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", 200))
    if not job_id:
        return {"ok": False, "error": "BAD_ARGUMENTS", "message": "job_id is required"}
    res = pbcs_request(
        cfg,
        "GET",
        f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}/details",
        params={"offset": offset, "limit": limit},
    )
    if not res.get("ok"): return res
    return {"ok": True, "jobId": job_id, "offset": offset, "limit": limit, **compact_job_details(res["response"])}

TOOLS = {
    "planning_discover_versions": tool_discover_versions,
    "planning_list_job_definitions": tool_list_job_definitions,
    "planning_execute_job": tool_execute_job,
    "planning_get_job_status": tool_get_job_status,
    "planning_get_job_details": tool_get_job_details,
}

# -----------------------------
# Minimal MCP stdio loop
# -----------------------------
def send(msg: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def main() -> None:
    cfg = load_cfg()

    # Advertise tools (simple)
    send({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {"name": "planning_discover_versions", "description": "Discover Planning REST versions"},
                {"name": "planning_list_job_definitions", "description": "List Planning job definitions"},
                {"name": "planning_execute_job", "description": "Execute a Planning job"},
                {"name": "planning_get_job_status", "description": "Get Planning job status"},
                {"name": "planning_get_job_details", "description": "Get Planning job details/messages"},
            ]
        }
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception:
            send({"jsonrpc": "2.0", "id": None, "error": {"message": "Invalid JSON"}})
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method != "tools/call":
            send({"jsonrpc": "2.0", "id": req_id, "error": {"message": f"Unsupported method: {method}"}})
            continue

        tool_name = params.get("name")
        args = params.get("arguments") or {}

        fn = TOOLS.get(tool_name)
        if not fn:
            send({"jsonrpc": "2.0", "id": req_id, "result": {"ok": False, "error": "UNKNOWN_TOOL", "tool": tool_name}})
            continue

        try:
            result = fn(cfg, args)
        except Exception as e:
            result = {"ok": False, "error": "TOOL_EXCEPTION", "message": str(e)}

        send({"jsonrpc": "2.0", "id": req_id, "result": result})

if __name__ == "__main__":
    main()
