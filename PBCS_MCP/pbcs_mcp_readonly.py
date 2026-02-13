from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import requests
from fastmcp import FastMCP


@dataclass(frozen=True)
class Config:
    base_url: str          # should be https://<host> (no /epmcloud)
    application: str
    api_version: str
    username: str
    password: str
    verify_ssl: bool


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _normalize_base_url(url: str) -> str:
    """
    User may paste a URL ending with /epmcloud. Planning REST base URL should be the host part.
    We strip trailing /epmcloud if present.
    """
    url = (url or "").strip().rstrip("/")
    if url.lower().endswith("/epmcloud"):
        url = url[: -len("/epmcloud")]
    return url.rstrip("/")


def load_config() -> Config:
    base_url = _normalize_base_url(os.getenv("PBCS_BASE_URL") or "")
    application = os.getenv("PBCS_APPLICATION") or ""
    api_version = os.getenv("PBCS_API_VERSION") or "v3"
    username = os.getenv("PBCS_USERNAME") or ""
    password = os.getenv("PBCS_PASSWORD") or ""
    verify_ssl = _env_bool("PBCS_VERIFY_SSL", True)

    missing = [k for k, v in {
        "PBCS_BASE_URL": base_url,
        "PBCS_APPLICATION": application,
        "PBCS_USERNAME": username,
        "PBCS_PASSWORD": password,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return Config(
        base_url=base_url,
        application=application,
        api_version=api_version,
        username=username,
        password=password,
        verify_ssl=verify_ssl,
    )


REDACT_KEYS = {"password", "authorization", "access_token", "client_secret", "secret"}


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.lower() in REDACT_KEYS:
                out[k] = "***REDACTED***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def pbcs_get(cfg: Config, path: str, *, params: Optional[Dict[str, Any]] = None, timeout_s: int = 60) -> Dict[str, Any]:
    """
    READ-ONLY HTTP GET wrapper.
    """
    url = f"{cfg.base_url}{path}"
    try:
        r = requests.get(
            url,
            headers={"Accept": "application/json"},
            params=params,
            auth=(cfg.username, cfg.password),
            timeout=timeout_s,
            verify=cfg.verify_ssl,
        )
    except requests.RequestException as e:
        return {"ok": False, "error": "NETWORK_ERROR", "message": str(e), "url": url}

    try:
        payload = r.json() if r.content else {}
    except Exception:
        payload = {"raw_text": r.text}

    if r.status_code >= 300:
        return {
            "ok": False,
            "error": "HTTP_ERROR",
            "status_code": r.status_code,
            "message": (payload.get("message") or payload.get("details") or r.reason),
            "response": redact(payload),
            "url": url,
        }

    return {"ok": True, "status_code": r.status_code, "response": redact(payload), "url": url}


def terminal(status: Optional[str]) -> bool:
    s = (status or "").upper()
    return s in ("SUCCEEDED", "FAILED", "ERROR", "CANCELED", "CANCELLED")


mcp = FastMCP("Oracle Planning (PBCS) MCP - Read Only Job Observer")


@mcp.tool
def planning_discover_versions() -> Dict[str, Any]:
    """
    GET /HyperionPlanning/rest/
    """
    cfg = load_config()
    return pbcs_get(cfg, "/HyperionPlanning/rest/")


@mcp.tool
def planning_list_job_definitions(application: Optional[str] = None, api_version: Optional[str] = None) -> Dict[str, Any]:
    """
    GET /HyperionPlanning/rest/{api_version}/applications/{application}/jobdefinitions
    """
    cfg = load_config()
    app = application or cfg.application
    v = api_version or cfg.api_version

    res = pbcs_get(cfg, f"/HyperionPlanning/rest/{v}/applications/{app}/jobdefinitions")
    if not res.get("ok"):
        return res

    items = (res["response"].get("items") or [])
    defs = [{
        "jobType": it.get("jobType"),
        "jobName": it.get("jobName"),
        "description": it.get("description"),
    } for it in items]

    return {"ok": True, "count": len(defs), "jobDefinitions": defs}


@mcp.tool
def planning_get_job_status(job_id: str, application: Optional[str] = None, api_version: Optional[str] = None) -> Dict[str, Any]:
    """
    GET /HyperionPlanning/rest/{api_version}/applications/{application}/jobs/{jobIdentifier}
    """
    cfg = load_config()
    app = application or cfg.application
    v = api_version or cfg.api_version

    res = pbcs_get(cfg, f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}")
    if not res.get("ok"):
        return res

    p = res["response"]
    status = p.get("status") or p.get("descriptiveStatus") or p.get("state")

    return {
        "ok": True,
        "jobId": job_id,
        "status": status,
        "percentComplete": p.get("percentComplete"),
        "startTime": p.get("startTime"),
        "endTime": p.get("endTime"),
        "raw": p,
    }


@mcp.tool
def planning_get_job_details(
    job_id: str,
    offset: int = 0,
    limit: int = 200,
    application: Optional[str] = None,
    api_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    GET /HyperionPlanning/rest/{api_version}/applications/{application}/jobs/{jobIdentifier}/details

    Oracle notes details are returned for certain job types (e.g., IMPORT_DATA / EXPORT_DATA / IMPORT_METADATA / EXPORT_METADATA),
    and supports paging via offset/limit. :contentReference[oaicite:1]{index=1}
    """
    cfg = load_config()
    app = application or cfg.application
    v = api_version or cfg.api_version

    res = pbcs_get(
        cfg,
        f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}/details",
        params={"offset": offset, "limit": limit},
    )
    if not res.get("ok"):
        return res

    items = (res["response"].get("items") or [])
    compact = [{
        "severity": it.get("severity"),
        "type": it.get("type"),
        "row": it.get("row"),
        "message": it.get("message"),
    } for it in items]

    return {"ok": True, "jobId": job_id, "offset": offset, "limit": limit, "count": len(compact), "items": compact}


@mcp.tool
def planning_watch_job(
    job_id: str,
    poll_interval_seconds: int = 10,
    timeout_seconds: int = 600,
    include_details: bool = True,
    details_offset: int = 0,
    details_limit: int = 200,
    application: Optional[str] = None,
    api_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    SAFE one-tool experience (read-only):
    Poll job status until terminal state or timeout; optionally fetch details at end.
    """
    start = time.time()
    polls: List[Dict[str, Any]] = []
    last_status: Optional[Dict[str, Any]] = None

    while True:
        st = planning_get_job_status(job_id, application=application, api_version=api_version)
        if not st.get("ok"):
            return {"ok": False, "error": "POLL_FAILED", "jobId": job_id, "status_call": st}

        polls.append({
            "ts": int(time.time()),
            "status": st.get("status"),
            "percentComplete": st.get("percentComplete"),
        })
        last_status = st

        if terminal(st.get("status")):
            break

        if (time.time() - start) >= timeout_seconds:
            return {
                "ok": False,
                "error": "TIMEOUT",
                "message": f"Job did not reach terminal state within {timeout_seconds}s",
                "jobId": job_id,
                "lastStatus": last_status,
                "polls": polls,
            }

        time.sleep(max(1, int(poll_interval_seconds)))

    details = None
    if include_details:
        details = planning_get_job_details(
            job_id,
            offset=details_offset,
            limit=details_limit,
            application=application,
            api_version=api_version,
        )

    return {
        "ok": True,
        "job": {
            "jobId": job_id,
            "finalStatus": (last_status or {}).get("status"),
            "pollCount": len(polls),
            "polls": polls,
        },
        "details": details,
    }


if __name__ == "__main__":
    mcp.run()
