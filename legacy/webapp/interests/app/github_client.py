"""Read-only GitHub panel via the REST API.

Shows pull requests awaiting your review, your own open PRs, and the unread
notification count. Token-authenticated; never raises (errors surface as data).
"""
import json
import logging
import urllib.request

from . import config

log = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_API = "https://api.github.com"


def _get(path: str):
    req = urllib.request.Request(
        _API + path,
        headers={
            "Authorization": "Bearer " + config.GITHUB_TOKEN,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "interests-dashboard",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as r:
        return json.load(r)


def _repo_item(r: dict) -> dict:
    return {
        "name": r.get("full_name") or r.get("name", ""),
        "url": r.get("html_url", ""),
        "pushed": r.get("pushed_at", ""),
        "language": r.get("language") or "",
        "stars": r.get("stargazers_count", 0),
        "private": bool(r.get("private")),
        "fork": bool(r.get("fork")),
        "description": r.get("description") or "",
    }


def fetch_github() -> dict:
    """Return {configured, login, repos, notifications, public_repos, followers, error}.

    `repos` is the 5 most-recently-active repositories the token can see.
    """
    if not config.GITHUB_TOKEN:
        return {"configured": False}

    result = {
        "configured": True,
        "login": "",
        "repos": [],
        "notifications": 0,
        "public_repos": 0,
        "followers": 0,
        "error": None,
    }
    try:
        me = _get("/user")
        result["login"] = me.get("login", "")
        result["public_repos"] = me.get("public_repos", 0)
        result["followers"] = me.get("followers", 0)

        repos = _get("/user/repos?sort=pushed&direction=desc&per_page=5")
        if isinstance(repos, list):
            result["repos"] = [_repo_item(r) for r in repos]

        try:
            notif = _get("/notifications")  # unread only, by default
            result["notifications"] = len(notif) if isinstance(notif, list) else 0
        except Exception:  # noqa: BLE001 - notifications are best-effort
            log.exception("GitHub notifications fetch failed")
    except Exception as exc:  # noqa: BLE001 - surface as data, not a 500
        log.exception("GitHub fetch failed")
        result["error"] = str(exc)
    return result
