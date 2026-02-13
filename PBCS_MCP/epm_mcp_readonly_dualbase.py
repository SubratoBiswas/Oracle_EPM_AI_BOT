from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP


@dataclass(frozen=True)
class Config:
    # User can paste either https://<host>/epmcloud or https://<host>
    service_url: str
    base_url: str          # derived host-only base (no /epmcloud)
    username: str
    password: str
    verify_ssl: bool


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _derive_base_url(service_url: str) -> str:
    url = (service_url or "").strip().rstrip("/")
    if url.lower().endswith("/epmcloud"):
        url = url[: -len("/epmcloud")]
    return url.rstrip("/")


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

    base_url = _derive_base_url(service_url)

    return Config(
        service_url=service_url,
        base_url=base_url,
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


def _http_get(url: str, cfg: Config, params: Optional[Dict[str, Any]] = None, timeout_s: int = 60) -> Dict[str, Any]:
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


def epm_get_dualbase(cfg: Config, interop_path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Try canonical base first:
      https://<BASE-URL>/interop/rest/...
    If 404, try service context:
      https://<BASE-URL>/epmcloud/interop/rest/...
    """
    # 1) canonical (per Oracle docs)
    url1 = f"{cfg.base_url}{interop_path}"
    r1 = _http_get(url1, cfg, params=params)
    if r1.get("ok"):
        r1["base_used"] = "base_url"
        return r1

    # If not found, try the /epmcloud context variant
    if r1.get("status_code") == 404:
        url2 = f"{cfg.base_url}/epmcloud{interop_path}"
        r2 = _http_get(url2, cfg, params=params)
        if r2.get("ok"):
            r2["base_used"] = "epmcloud_context"
            return r2
        return {"ok": False, "attempts": [r1, r2]}

    # For non-404 errors (401/403/500), return the first response
    return r1


mcp = FastMCP("Oracle EPM Interop MCP (Read-only, Dual-base)")


@mcp.tool
def epm_list_files() -> Dict[str, Any]:
    """
    GET /interop/rest/v2/files/list  (Migration repository list files)
    """
    cfg = load_config()
    return epm_get_dualbase(cfg, "/interop/rest/v2/files/list")


@mcp.tool
def epm_list_backups() -> Dict[str, Any]:
    """
    GET /interop/rest/v2/backups/list  (OCI backups list)
    """
    cfg = load_config()
    return epm_get_dualbase(cfg, "/interop/rest/v2/backups/list")


@mcp.tool
def epm_get_daily_maintenance_time() -> Dict[str, Any]:
    """
    GET /interop/rest/v2/maintenance/getdailymaintenancestarttime
    """
    cfg = load_config()
    return epm_get_dualbase(cfg, "/interop/rest/v2/maintenance/getdailymaintenancestarttime")


@mcp.tool
def epm_get_status(operation: str, job_id: str) -> Dict[str, Any]:
    """
    GET /interop/rest/v2/status/<operation>/<jobId>
    Read-only status check for async operations.
    """
    cfg = load_config()
    op = (operation or "").strip().lower()
    if not op or "/" in op or ".." in op:
        return {"ok": False, "error": "INVALID_OPERATION", "message": "operation must be a simple token like 'download'."}
    return epm_get_dualbase(cfg, f"/interop/rest/v2/status/{op}/{job_id}")


if __name__ == "__main__":
    mcp.run()
