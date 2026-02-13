from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP


@dataclass(frozen=True)
class Config:
    service_url: str   # MUST include /epmcloud (per your requirement)
    username: str
    password: str
    verify_ssl: bool


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def load_config() -> Config:
    service_url = (os.getenv("EPM_SERVICE_URL") or "").strip().rstrip("/")
    username = os.getenv("EPM_USERNAME") or ""
    password = os.getenv("EPM_PASSWORD") or ""
    verify_ssl = _env_bool("EPM_VERIFY_SSL", True)

    missing = [k for k, v in {
        "EPM_SERVICE_URL": service_url,
        "EPM_USERNAME": username,
        "EPM_PASSWORD": password,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    # Guardrail: you explicitly want /epmcloud
    if not service_url.lower().endswith("/epmcloud"):
        raise RuntimeError("EPM_SERVICE_URL must end with '/epmcloud' (example: https://<host>/epmcloud)")

    return Config(service_url=service_url, username=username, password=password, verify_ssl=verify_ssl)


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


def epm_get(cfg: Config, path: str, *, params: Optional[Dict[str, Any]] = None, timeout_s: int = 60) -> Dict[str, Any]:
    """
    READ-ONLY GET wrapper.
    Builds final URL as: <EPM_SERVICE_URL> + <path>
    Example: https://<host>/epmcloud + /interop/rest/v2/files/list
    """
    url = f"{cfg.service_url}{path}"
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
            "message": payload.get("message") or payload.get("details") or r.reason,
            "response": redact(payload),
            "url": url,
        }

    return {"ok": True, "status_code": r.status_code, "response": redact(payload), "url": url}


mcp = FastMCP("Oracle EPM (interop) MCP - Read Only")


@mcp.tool
def epm_list_files() -> Dict[str, Any]:
    """
    Lists files and snapshots in the EPM repository (read-only).
    Docs: GET /interop/rest/v2/files/list
    """
    cfg = load_config()
    return epm_get(cfg, "/interop/rest/v2/files/list")


@mcp.tool
def epm_list_backups() -> Dict[str, Any]:
    """
    Lists OCI backups (read-only).
    Docs: GET /interop/rest/v2/backups/list
    """
    cfg = load_config()
    return epm_get(cfg, "/interop/rest/v2/backups/list")


@mcp.tool
def epm_get_daily_maintenance_time() -> Dict[str, Any]:
    """
    Returns build version + daily maintenance start time (read-only).
    Docs: GET /interop/rest/v2/maintenance/getdailymaintenancestarttime
    """
    cfg = load_config()
    return epm_get(cfg, "/interop/rest/v2/maintenance/getdailymaintenancestarttime")


@mcp.tool
def epm_get_status(operation: str, job_id: str) -> Dict[str, Any]:
    """
    Generic status poller for async EPM interop operations (read-only).
    Many v2 operations return a link like:
      GET /interop/rest/v2/status/<operation>/<jobId>
    Example from docs: /interop/rest/v2/status/download/<id>

    operation examples (depends on what created the job):
      download, upload, import, export, refresh, etc.
    """
    cfg = load_config()
    op = (operation or "").strip().lower()
    if not op or "/" in op or ".." in op:
        return {"ok": False, "error": "INVALID_OPERATION", "message": "operation must be a simple token like 'download'."}
    return epm_get(cfg, f"/interop/rest/v2/status/{op}/{job_id}")


if __name__ == "__main__":
    mcp.run()
