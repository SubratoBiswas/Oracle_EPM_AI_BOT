from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import requests
from fastmcp import FastMCP


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Config:
    base_url: str
    application: str
    api_version: str
    verify_ssl: bool

    auth_mode: str  # "basic" or "oauth2"
    username: Optional[str]
    password: Optional[str]

    idcs_token_url: Optional[str]
    oauth_client_id: Optional[str]
    oauth_client_secret: Optional[str]
    oauth_scope: Optional[str]

    client_api_key: Optional[str]  # optional local gating
    allowlist_jobs: Optional[str]  # comma-separated "TYPE:NAME,TYPE:NAME"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def load_config() -> Config:
    base_url = (os.getenv("PBCS_BASE_URL") or "").rstrip("/")
    application = os.getenv("PBCS_APPLICATION") or ""
    api_version = os.getenv("PBCS_API_VERSION") or "v3"
    verify_ssl = _env_bool("PBCS_VERIFY_SSL", True)

    auth_mode = (os.getenv("PBCS_AUTH_MODE") or "basic").strip().lower()
    username = os.getenv("PBCS_USERNAME")
    password = os.getenv("PBCS_PASSWORD")

    idcs_token_url = os.getenv("PBCS_IDCS_TOKEN_URL")
    oauth_client_id = os.getenv("PBCS_OAUTH_CLIENT_ID")
    oauth_client_secret = os.getenv("PBCS_OAUTH_CLIENT_SECRET")
    oauth_scope = os.getenv("PBCS_OAUTH_SCOPE")

    client_api_key = os.getenv("MCP_CLIENT_API_KEY")
    allowlist_jobs = os.getenv("PBCS_ALLOWLIST_JOBS")  # optional

    missing = []
    if not base_url:
        missing.append("PBCS_BASE_URL")
    if not application:
        missing.append("PBCS_APPLICATION")

    if auth_mode not in ("basic", "oauth2"):
        raise RuntimeError("PBCS_AUTH_MODE must be 'basic' or 'oauth2'")

    if auth_mode == "basic":
        if not username:
            missing.append("PBCS_USERNAME")
        if not password:
            missing.append("PBCS_PASSWORD")
    else:
        if not idcs_token_url:
            missing.append("PBCS_IDCS_TOKEN_URL")
        if not oauth_client_id:
            missing.append("PBCS_OAUTH_CLIENT_ID")
        if not oauth_client_secret:
            missing.append("PBCS_OAUTH_CLIENT_SECRET")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return Config(
        base_url=base_url,
        application=application,
        api_version=api_version,
        verify_ssl=verify_ssl,
        auth_mode=auth_mode,
        username=username,
        password=password,
        idcs_token_url=idcs_token_url,
        oauth_client_id=oauth_client_id,
        oauth_client_secret=oauth_client_secret,
        oauth_scope=oauth_scope,
        client_api_key=client_api_key,
        allowlist_jobs=allowlist_jobs,
    )


# -----------------------------
# Security helpers
# -----------------------------

REDACT_KEYS = {"password", "secret", "client_secret", "authorization", "access_token"}


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


def enforce_client_key(cfg: Config, client_api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if cfg.client_api_key and client_api_key != cfg.client_api_key:
        return {"ok": False, "error": "UNAUTHORIZED_CLIENT", "message": "Invalid MCP client API key."}
    return None


def allowlist_check(cfg: Config, job_type: str, job_name: str) -> Optional[Dict[str, Any]]:
    """
    Optional allowlist: PBCS_ALLOWLIST_JOBS="RULES:RollupUSSales,REFRESH_CUBE:RefreshPlan1"
    """
    if not cfg.allowlist_jobs:
        return None
    allowed = set(x.strip() for x in cfg.allowlist_jobs.split(",") if x.strip())
    key = f"{job_type}:{job_name}"
    if key not in allowed:
        return {"ok": False, "error": "JOB_NOT_ALLOWED", "message": f"Job not allowlisted: {key}"}
    return None


# -----------------------------
# OAuth2 token cache (IDCS)
# -----------------------------

class TokenCache:
    def __init__(self) -> None:
        self.token: Optional[str] = None
        self.expires_at: float = 0.0

    def get(self) -> Optional[str]:
        if self.token and time.time() < (self.expires_at - 30):
            return self.token
        return None

    def set(self, token: str, expires_in: int) -> None:
        self.token = token
        self.expires_at = time.time() + max(0, int(expires_in))


_TOKEN_CACHE = TokenCache()


def get_oauth_token(cfg: Config) -> Tuple[bool, Dict[str, Any]]:
    cached = _TOKEN_CACHE.get()
    if cached:
        return True, {"access_token": cached, "cached": True}

    data = {"grant_type": "client_credentials"}
    if cfg.oauth_scope:
        data["scope"] = cfg.oauth_scope

    try:
        resp = requests.post(
            cfg.idcs_token_url,
            data=data,
            auth=(cfg.oauth_client_id, cfg.oauth_client_secret),
            timeout=30,
            verify=cfg.verify_ssl,
        )
    except requests.RequestException as e:
        return False, {"error": "OAUTH_NETWORK_ERROR", "message": str(e)}

    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": resp.text}

    if resp.status_code >= 300:
        return False, {"error": "OAUTH_HTTP_ERROR", "status_code": resp.status_code, "response": payload}

    token = payload.get("access_token")
    expires_in = payload.get("expires_in", 3600)
    if not token:
        return False, {"error": "OAUTH_BAD_RESPONSE", "response": payload}

    _TOKEN_CACHE.set(token, int(expires_in))
    return True, {"access_token": token, "cached": False}


# -----------------------------
# Real PBCS REST calls
# -----------------------------

def pbcs_request(
    cfg: Config,
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    url = f"{cfg.base_url}{path}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    auth = None
    if cfg.auth_mode == "basic":
        auth = (cfg.username, cfg.password)
    else:
        ok, tok = get_oauth_token(cfg)
        if not ok:
            return {"ok": False, **redact(tok)}
        headers["Authorization"] = f"Bearer {tok['access_token']}"

    try:
        r = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            data=None if json_body is None else json.dumps(json_body),
            auth=auth,
            timeout=timeout_s,
            verify=cfg.verify_ssl,
        )
    except requests.RequestException as e:
        return {"ok": False, "error": "NETWORK_ERROR", "message": str(e)}

    try:
        payload = r.json() if r.content else {}
    except Exception:
        payload = {"raw_text": r.text}

    if r.status_code >= 300:
        return {
            "ok": False,
            "error": "HTTP_ERROR",
            "status_code": r.status_code,
            "message": payload.get("message") or payload.get("details") or r.reason,
            "response": redact(payload),
        }

    return {"ok": True, "status_code": r.status_code, "response": redact(payload)}


def terminal(status: Optional[str]) -> bool:
    s = (status or "").upper()
    return s in ("SUCCEEDED", "FAILED", "ERROR", "CANCELED", "CANCELLED")


# -----------------------------
# MCP server + tools
# -----------------------------

mcp = FastMCP("PBCS Copilot (Real Oracle EPM APIs)")


@mcp.tool
def planning_discover_versions(client_api_key: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config()
    deny = enforce_client_key(cfg, client_api_key)
    if deny:
        return deny
    return pbcs_request(cfg, "GET", "/HyperionPlanning/rest/")


@mcp.tool
def planning_list_job_definitions(
    application: Optional[str] = None,
    api_version: Optional[str] = None,
    client_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config()
    deny = enforce_client_key(cfg, client_api_key)
    if deny:
        return deny

    app = application or cfg.application
    v = api_version or cfg.api_version
    res = pbcs_request(cfg, "GET", f"/HyperionPlanning/rest/{v}/applications/{app}/jobdefinitions")
    if not res.get("ok"):
        return res

    payload = res["response"]
    items = payload.get("items") or payload.get("jobDefinitions") or []
    compact = [{
        "jobType": it.get("jobType") or it.get("type"),
        "jobName": it.get("jobName") or it.get("name"),
        "description": it.get("description"),
    } for it in items]

    return {"ok": True, "count": len(compact), "jobDefinitions": compact}


@mcp.tool
def planning_execute_job(
    job_type: str,
    job_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    application: Optional[str] = None,
    api_version: Optional[str] = None,
    client_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config()
    deny = enforce_client_key(cfg, client_api_key)
    if deny:
        return deny

    guard = allowlist_check(cfg, job_type, job_name)
    if guard:
        return guard

    app = application or cfg.application
    v = api_version or cfg.api_version

    body: Dict[str, Any] = {"jobType": job_type, "jobName": job_name}
    if parameters:
        body["parameters"] = parameters

    res = pbcs_request(cfg, "POST", f"/HyperionPlanning/rest/{v}/applications/{app}/jobs", json_body=body)
    if not res.get("ok"):
        return res

    p = res["response"]
    job_id = p.get("jobId") or p.get("jobID") or p.get("id") or p.get("jobIdentifier")
    status = p.get("status") or p.get("descriptiveStatus") or p.get("state")
    return {"ok": True, "jobId": job_id, "status": status, "raw": p}


@mcp.tool
def planning_get_job_status(
    job_id: str,
    application: Optional[str] = None,
    api_version: Optional[str] = None,
    client_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config()
    deny = enforce_client_key(cfg, client_api_key)
    if deny:
        return deny

    app = application or cfg.application
    v = api_version or cfg.api_version

    res = pbcs_request(cfg, "GET", f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}")
    if not res.get("ok"):
        return res

    p = res["response"]
    status = p.get("status") or p.get("descriptiveStatus") or p.get("state")
    return {
        "ok": True,
        "jobId": job_id,
        "status": status,
        "percentComplete": p.get("percentComplete") or p.get("progress"),
        "startTime": p.get("startTime") or p.get("startedAt"),
        "endTime": p.get("endTime") or p.get("endedAt"),
        "raw": p,
    }


@mcp.tool
def planning_get_job_details(
    job_id: str,
    offset: int = 0,
    limit: int = 200,
    application: Optional[str] = None,
    api_version: Optional[str] = None,
    client_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config()
    deny = enforce_client_key(cfg, client_api_key)
    if deny:
        return deny

    app = application or cfg.application
    v = api_version or cfg.api_version

    res = pbcs_request(
        cfg,
        "GET",
        f"/HyperionPlanning/rest/{v}/applications/{app}/jobs/{job_id}/details",
        params={"offset": offset, "limit": limit},
    )
    if not res.get("ok"):
        return res

    p = res["response"]
    items = p.get("items") or p.get("details") or p.get("messages") or []
    compact = [{
        "severity": it.get("severity"),
        "type": it.get("type"),
        "row": it.get("row"),
        "message": it.get("message") or it.get("details") or it.get("text"),
    } for it in items]

    return {"ok": True, "jobId": job_id, "offset": offset, "limit": limit, "count": len(compact), "items": compact}


@mcp.tool
def planning_run_job_and_wait(
    job_type: str,
    job_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    poll_interval_seconds: int = 5,
    timeout_seconds: int = 900,
    application: Optional[str] = None,
    api_version: Optional[str] = None,
    details_offset: int = 0,
    details_limit: int = 200,
    client_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-tool experience:
    execute -> poll until terminal/timeout -> fetch details -> return summary
    """
    # Execute
    exec_res = planning_execute_job(
        job_type=job_type,
        job_name=job_name,
        parameters=parameters,
        application=application,
        api_version=api_version,
        client_api_key=client_api_key,
    )
    if not exec_res.get("ok"):
        return exec_res

    job_id = exec_res.get("jobId")
    if not job_id:
        return {"ok": False, "error": "NO_JOB_ID", "message": "Execute did not return jobId", "raw": exec_res}

    start = time.time()
    polls: List[Dict[str, Any]] = []

    last_status: Optional[Dict[str, Any]] = None
    while True:
        st = planning_get_job_status(
            job_id=job_id,
            application=application,
            api_version=api_version,
            client_api_key=client_api_key,
        )
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
                "message": f"Job did not complete within {timeout_seconds}s",
                "jobId": job_id,
                "lastStatus": last_status,
                "polls": polls,
            }

        time.sleep(max(1, int(poll_interval_seconds)))

    # Details (best-effort)
    details = planning_get_job_details(
        job_id=job_id,
        offset=details_offset,
        limit=details_limit,
        application=application,
        api_version=api_version,
        client_api_key=client_api_key,
    )

    items = details.get("items") if details.get("ok") else []
    errors = [x for x in items if (x.get("severity") or "").upper() == "ERROR"]
    warns = [x for x in items if (x.get("severity") or "").upper() in ("WARN", "WARNING")]

    return {
        "ok": True,
        "job": {
            "jobId": job_id,
            "jobType": job_type,
            "jobName": job_name,
            "finalStatus": (last_status or {}).get("status"),
            "pollCount": len(polls),
            "polls": polls,
        },
        "summary": {
            "errors": errors[:25],
            "warnings": warns[:25],
            "detailsSample": items[:25],
        },
        "raw": {
            "execute": exec_res,
            "finalStatus": last_status,
            "details": details if details.get("ok") else details,
        }
    }


if __name__ == "__main__":
    mcp.run()
