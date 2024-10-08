import os


TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "true").lower() in (
    "true",
    "1",
    "t",
    "on",
)