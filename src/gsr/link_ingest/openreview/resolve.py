from urllib.parse import urlparse, parse_qs


def resolve_openreview_url(url: str) -> str:
    """
    Extract forum id from:
      - https://openreview.net/forum?id=XXX
      - https://openreview.net/pdf?id=XXX
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if "id" not in qs:
        raise ValueError("Cannot resolve forum id from URL")

    return qs["id"][0]
