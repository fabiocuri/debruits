"""Read-only agenda from one or more secret iCal (ICS) feeds.

Supports a Google Calendar "secret address in iCal format" and/or an Outlook /
Microsoft 365 published calendar (which includes Teams meetings). No OAuth:
each feed is just a secret URL. Events from all feeds are merged, tagged with a
source label, and returned with ISO timestamps; the browser groups them into
Today / Upcoming in local time, so the server stays timezone-agnostic.
"""
import logging
import re
import urllib.request
from datetime import datetime, timedelta, timezone

from . import config

log = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_TEAMS_RE = re.compile(r"https://teams\.microsoft\.com/l/meetup-join/[^\s>\"']+")


def _feeds() -> list[tuple[str, str]]:
    """Configured (label, url) calendar feeds, in display priority order."""
    feeds = []
    if config.CALENDAR_ICS_URL:
        feeds.append((config.CALENDAR_LABEL or "Calendar", config.CALENDAR_ICS_URL))
    if config.TEAMS_ICS_URL:
        feeds.append((config.TEAMS_LABEL or "Teams", config.TEAMS_ICS_URL))
    return feeds


def _to_iso(value) -> str | None:
    """Normalise an icalendar date/datetime to an ISO string (UTC-aware)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value.isoformat()  # a date (all-day event)


def _teams_link(event) -> str | None:
    """Pull a Teams join URL out of an event's location or description, if any."""
    for field in ("LOCATION", "DESCRIPTION"):
        val = event.get(field)
        if not val:
            continue
        m = _TEAMS_RE.search(str(val))
        if m:
            return m.group(0)
    return None


def _read(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "interests-dashboard"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        return resp.read()


def fetch_agenda() -> dict:
    """Return merged upcoming events across all feeds, or an unconfigured marker.

    Shape: {configured, events: [{summary, location, start, end, all_day,
    source, join}], error}. Never raises — failures surface in `error`.
    """
    feeds = _feeds()
    if not feeds:
        return {"configured": False}

    result = {"configured": True, "events": [], "error": None}
    items: list[dict] = []
    errors: list[str] = []

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=12)
    end = now + timedelta(days=config.AGENDA_DAYS)

    for label, url in feeds:
        try:
            import icalendar
            import recurring_ical_events

            cal = icalendar.Calendar.from_ical(_read(url))
            for e in recurring_ical_events.of(cal).between(start, end):
                dtstart = e.get("DTSTART")
                if dtstart is None:
                    continue
                dtstart = dtstart.dt
                dtend = e.get("DTEND")
                items.append({
                    "summary": str(e.get("SUMMARY", "(no title)")),
                    "location": (str(e.get("LOCATION")) or None) if e.get("LOCATION") else None,
                    "start": _to_iso(dtstart),
                    "end": _to_iso(dtend.dt) if dtend is not None else None,
                    "all_day": not isinstance(dtstart, datetime),
                    "source": label,
                    "join": _teams_link(e),
                })
        except Exception as exc:  # noqa: BLE001 - one bad feed shouldn't blank the rest
            log.exception("Agenda fetch failed for %s", label)
            errors.append(f"{label}: {exc}")

    items.sort(key=lambda x: x["start"] or "")
    result["events"] = items[: config.AGENDA_MAX]
    # Only report an error if we got nothing usable.
    if errors and not result["events"]:
        result["error"] = "; ".join(errors)
    return result
