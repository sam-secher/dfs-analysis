import re
from datetime import datetime, timezone

ISO_Z_MILLIS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

def to_iso_z_ms(dt: datetime) -> str:
    """UTC → 'YYYY-MM-DDTHH:MM:SS.mmmZ' (validated)."""
    dt_utc = dt.astimezone(timezone.utc)
    s = dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if not ISO_Z_MILLIS.fullmatch(s):
        msg = f"Bad timestamp format: {s!r}"
        raise ValueError(msg)
    return s
