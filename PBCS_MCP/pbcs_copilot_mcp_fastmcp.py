from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP  # FastMCP implements the MCP handshake correctly


mcp = FastMCP("PBCS Copilot (Fake PBCS via REST)")

def cfg() -> Dict[str, str]:
    base_url = (os.getenv("PBCS_BASE_URL") or "").rstrip("/")
    app = os.getenv("PBCS_APPLICATION") or "Vision"
    v = os.getenv("PBCS_API_VERSION") or "v3"
    if not base_url:
        raise RuntimeError("Missing PBCS_BASE_URL (for fake server use http://127.0.0.1:9010)")
    return {"base_url": base_url, "app": app, "v": v}

def req(method: str, path: str, *, params=None, body=None) -> Dict[str, Any]:
    c = cfg()
    url = f"{c['base_url']}{path}"
    r = requests.request(
        method=method,
        url=url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        params=params,
        data=None if body is None else json.dumps(body),
        timeout=30,
    )
    try:
        payload = r.json() if r.content else {}
    except Exception:
        payload = {"raw_text": r.text}

    if r.status_code >= 300:
        return {"ok": False, "status_code": r.status_code, "error": "HTTP_ERROR", "response": payload}
    return {"ok": True, "status_code": r.status_code, "response": payload}

def compact_job_defs(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = payload.get("items") or []
    defs = [{"jobType": i.get("jobType"), "jobName": i.get("jobName"), "description": i.get("description")} for i in items]
    return {"count": len(defs), "jobDefinitions": defs}

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
    out = [{"severity": i.get("severity"), "type": i.get("type"), "row": i.get("row"), "message": i.get("message")} for i in items]
    return {"count": len(out), "items": out, "hasMore": payload.get("hasMore", False)}


@mcp.tool
def planning_discover_versions() -> Dict[str, Any]:
    return req("GET", "/HyperionPlanning/rest/")


@mcp.tool
def planning_list_job_definitions(api_version: Optional[str] = None, application: Optional[str] = None) -> Dict[str, Any]:
    c = cfg()
    v = api_version or c["v"]
    app = application or c["app"]
    res = req("GET", f"/HyperionPlanning/rest/{v}/applications/{app}/jobdefinitions")
    if not res["ok"]:
        return res
    return {"ok": True, **compact_job_defs(res["response"])}


@mcp.tool
def planning_execute_job(
    job_type: str,
    job_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    api_version: Optional[str] = None,
    application: Optional[str] = None,
) -> Dict[str, Any]:
    c = cfg()
    v = api_version or c["v"]
    app = application or c["app"]
    body = {"jobType": job_type, "jobName": job_name, "parameters": parameters or {}}
    res = req("POST", f"/HyperionPlanning/rest/{v}/applications/{app}/jobs", body=body)
    if not res["ok"]:
        return res
    payload = res["response"]
    return {"ok": True, "jobId": payload.get("jobId"), "status": payload.get("status") or payload.get("descriptiveStatus")}


@mcp.tool
def planning_get_job_status(job_id: str, api_version: Optional[str] = None, application: Optional[str] = None) -> Dict[str, Any]:
    c = cfg()
    v = api_version or c["v"]
    app = application or c["app"]
    res = req("GET", f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}")
    if not res["ok"]:
        return res
    return {"ok": True, "jobId": job_id, **compact_job_status(res["response"])}


@mcp.tool
def planning_get_job_details(
    job_id: str,
    offset: int = 0,
    limit: int = 200,
    api_version: Optional[str] = None,
    application: Optional[str] = None,
) -> Dict[str, Any]:
    c = cfg()
    v = api_version or c["v"]
    app = application or c["app"]
    res = req(
        "GET",
        f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}/details",
        params={"offset": offset, "limit": limit},
    )
    if not res["ok"]:
        return res
    return {"ok": True, "jobId": job_id, "offset": offset, "limit": limit, **compact_job_details(res["response"])}


if __name__ == "__main__":
    # Stdio transport for Claude Desktop
    mcp.run()
