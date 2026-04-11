"""Thin mail.tm client for receiving verification emails."""
import json
import time
import urllib.parse
import urllib.request
import secrets
import string

BASE = "https://api.mail.tm"


def _request(method, path, data=None, token=None):
    url = BASE + path
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read().decode()), "status": e.code}
        except:
            return {"error": str(e), "status": e.code}


def get_domains():
    return _request("GET", "/domains")


def create_inbox():
    """Create a new mail.tm inbox. Returns (address, password, token, account_id)."""
    domains = get_domains()
    # Newest API returns a list directly
    if isinstance(domains, dict):
        domains = domains.get("hydra:member", [])
    domain = domains[0]["domain"]

    local = "mcgillnlp" + secrets.token_hex(4)
    address = f"{local}@{domain}"
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for _ in range(16))

    # Create account
    res = _request("POST", "/accounts", {"address": address, "password": password})
    if "error" in res:
        raise RuntimeError(f"Account creation failed: {res}")
    account_id = res["id"]

    # Get token
    token_res = _request("POST", "/token", {"address": address, "password": password})
    if "token" not in token_res:
        raise RuntimeError(f"Token creation failed: {token_res}")
    return address, password, token_res["token"], account_id


def list_messages(token):
    return _request("GET", "/messages", token=token)


def get_message(token, msg_id):
    return _request("GET", f"/messages/{msg_id}", token=token)


def wait_for_message(token, subject_hint=None, timeout=120):
    """Poll the inbox until a message arrives."""
    deadline = time.time() + timeout
    seen_ids = set()
    while time.time() < deadline:
        res = list_messages(token)
        messages = res.get("hydra:member", []) if isinstance(res, dict) else res
        for msg in messages:
            if msg["id"] in seen_ids:
                continue
            if subject_hint and subject_hint.lower() not in msg.get("subject", "").lower():
                seen_ids.add(msg["id"])
                continue
            # Get full message body
            full = get_message(token, msg["id"])
            return full
        time.sleep(5)
    return None


if __name__ == "__main__":
    addr, pw, tok, aid = create_inbox()
    print(f"Address: {addr}")
    print(f"Password: {pw}")
    print(f"Token: {tok[:30]}...")
    print("\nWaiting for any message (timeout 60s)...")
    msg = wait_for_message(tok, timeout=60)
    print(f"Got: {msg}")
