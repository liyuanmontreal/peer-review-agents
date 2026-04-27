import openreview

from gsr.config import API_V2_BASE_URL, get_credentials


def build_client(username: str | None = None, password: str | None = None) -> openreview.api.OpenReviewClient:
    """Create an authenticated OpenReview API v2 client.

    Falls back to environment credentials when *username*/*password* are None.
    """
    if username is None or password is None:
        username, password = get_credentials()

    return openreview.api.OpenReviewClient(
        baseurl=API_V2_BASE_URL,
        username=username,
        password=password,
    )
