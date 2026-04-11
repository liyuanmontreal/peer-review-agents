"""
FPT AI Factory end-to-end signup:
1. Create a mail.tm inbox for a verifiable email address
2. Fill signup form, bypass reCAPTCHA v2 via audio challenge
3. Wait for verification email, click the link
4. Log in and apply referral code
"""
import asyncio
import json
import os
import random
import re
import secrets
import string
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path

import speech_recognition as sr
from pydub import AudioSegment
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

sys.path.insert(0, str(Path(__file__).parent))
from mailtm import create_inbox, wait_for_message

SKILL_DIR = Path(__file__).parent
STATE_FILE = SKILL_DIR / "fpt_state.json"
STORAGE_FILE = SKILL_DIR / "fpt_browser_state.json"
FPT_URL = "https://ai.fptcloud.com"
REFERRAL_CODE = "SIVAREDDYPROF-BNI8WX1W5X"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def generate_password():
    alphabet = string.ascii_letters + string.digits
    pw_body = ''.join(secrets.choice(alphabet) for _ in range(10))
    return f"Gpu!{pw_body}9"


async def human_type(el, text):
    for char in text:
        await el.type(char, delay=random.randint(50, 150))


def transcribe_audio(url):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        with open(mp3_path, "wb") as f:
            f.write(resp.read())
    wav_path = mp3_path.replace(".mp3", ".wav")
    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)
    try:
        return r.recognize_google(audio_data)
    except Exception as e:
        print(f"[captcha] transcription failed: {e}")
        return None
    finally:
        for p in (mp3_path, wav_path):
            try:
                os.remove(p)
            except:
                pass


async def solve_recaptcha_v2_audio(page):
    anchor_frame = None
    for frame in page.frames:
        if "recaptcha" in frame.url and "anchor" in frame.url:
            anchor_frame = frame
            break
    if not anchor_frame:
        return False

    print("[captcha] Clicking 'I'm not a robot'...")
    await anchor_frame.click("#recaptcha-anchor", timeout=10000)
    await page.wait_for_timeout(3000)

    # Check for auto-pass
    is_checked = await anchor_frame.evaluate(
        "() => document.querySelector('#recaptcha-anchor')?.getAttribute('aria-checked') === 'true'"
    )
    if is_checked:
        print("[captcha] Auto-passed")
        return True

    # Find challenge iframe
    bframe = None
    for _ in range(10):
        for frame in page.frames:
            if "recaptcha" in frame.url and "bframe" in frame.url:
                bframe = frame
                break
        if bframe:
            break
        await page.wait_for_timeout(500)

    if not bframe:
        return False

    print("[captcha] Switching to audio challenge...")
    await bframe.click("#recaptcha-audio-button", timeout=10000)
    await page.wait_for_timeout(3000)

    blocked = await bframe.evaluate(
        "() => document.querySelector('.rc-doscaptcha-header-text')?.textContent || null"
    )
    if blocked:
        print(f"[captcha] Google blocked: {blocked}")
        return False

    audio_url = None
    for attempt in range(5):
        audio_url = await bframe.evaluate("""
            () => {
                const src = document.querySelector('#audio-source');
                if (src) return src.src;
                const link = document.querySelector('.rc-audiochallenge-tdownload-link');
                if (link) return link.href;
                return null;
            }
        """)
        if audio_url:
            break
        await page.wait_for_timeout(2000)

    if not audio_url:
        return False

    text = transcribe_audio(audio_url)
    if not text:
        return False

    print(f"[captcha] Transcription: {text}")
    await bframe.fill("#audio-response", text)
    await page.wait_for_timeout(500)
    await bframe.click("#recaptcha-verify-button")
    await page.wait_for_timeout(4000)

    is_checked = await anchor_frame.evaluate(
        "() => document.querySelector('#recaptcha-anchor')?.getAttribute('aria-checked') === 'true'"
    )
    return bool(is_checked)


async def signup_and_verify():
    # 1. Create mail.tm inbox
    print("[mail] Creating mail.tm inbox...")
    email, mail_pw, mail_token, mail_id = create_inbox()
    print(f"[mail] Address: {email}")

    # Generate other credentials
    tag = uuid.uuid4().hex[:6]
    username = f"mcgillnlp{tag}"
    fpt_password = generate_password()
    creds = {
        "username": username,
        "email": email,
        "password": fpt_password,
        "first_name": "McGill",
        "last_name": "NLP",
        "mail_tm_password": mail_pw,
        "mail_tm_token": mail_token,
    }
    print(f"[fpt] Username: {username}")
    print(f"[fpt] Password: {fpt_password}")

    stealth = Stealth()
    async with stealth.use_async(async_playwright()) as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            channel="chromium",
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
        )
        page = await context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            window.chrome = {runtime: {}};
        """)

        # 2. Navigate to signup
        print("\n[signup] Navigating to FPT...")
        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_timeout(3000)
        await page.click("button:has-text('Sign in')")
        await page.wait_for_timeout(2000)
        await page.click("button:has-text('Continue with FPT ID')")
        await page.wait_for_timeout(5000)
        await page.click("a:has-text('Sign up')")
        await page.wait_for_timeout(5000)

        # 3. Fill form
        print("[signup] Filling form...")
        for field_id, value in [
            ("#firstName", creds["first_name"]),
            ("#lastName", creds["last_name"]),
            ("#email", creds["email"]),
            ("#username", creds["username"]),
            ("#password", creds["password"]),
            ("#password-confirm", creds["password"]),
        ]:
            el = await page.query_selector(field_id)
            await human_type(el, value)
            await page.wait_for_timeout(random.randint(300, 700))

        await page.check("#termsAccepted")
        await page.wait_for_timeout(1000)
        await page.mouse.move(400, 300)
        await page.mouse.move(600, 500, steps=10)
        await page.wait_for_timeout(500)

        # 4. Solve reCAPTCHA
        if not await solve_recaptcha_v2_audio(page):
            print("[signup] CAPTCHA failed")
            await page.screenshot(path="/tmp/fpt_captcha_failed.png")
            await browser.close()
            return False

        # 5. Submit
        print("[signup] Submitting form...")
        await page.click("button:has-text('Sign up'):not(a)")
        await page.wait_for_timeout(10000)
        await page.screenshot(path="/tmp/fpt_after_signup.png")

        page_text = await page.inner_text("body")
        if "invalid recaptcha" in page_text.lower():
            print("[signup] CAPTCHA rejected after submit")
            await browser.close()
            return False

        print(f"[signup] URL: {page.url}")

        # 6. Wait for verification email
        if "verify" in page_text.lower() or "verify" in page.url.lower():
            print("\n[mail] Waiting for verification email (up to 180s)...")
            msg = wait_for_message(mail_token, timeout=180)
            if not msg:
                print("[mail] No verification email received")
                await browser.close()
                return False

            print(f"[mail] Got: {msg.get('subject', '(no subject)')}")

            # Extract verification link
            body = msg.get("html", [""])[0] if isinstance(msg.get("html"), list) else msg.get("html", "")
            if not body:
                body = msg.get("text", "")
            print(f"[mail] Body preview: {body[:500]}")

            # Find verification URL — must be the Keycloak action-token URL.
            # Keycloak verify-email links look like:
            # https://id.fptcloud.com/auth/realms/FptSmartCloud/login-actions/action-token?key=...
            links = re.findall(r'https?://[^\s"\'<>]+', body)
            verify_link = None
            # Highest priority: Keycloak action-token URL
            for link in links:
                if "id.fptcloud.com" in link and ("action-token" in link or "login-actions" in link):
                    verify_link = link
                    break
            # Fallback: any id.fptcloud.com link
            if not verify_link:
                for link in links:
                    if "id.fptcloud.com" in link:
                        verify_link = link
                        break

            if not verify_link:
                print(f"[mail] No Keycloak link found. All links:\n{links}")
                print(f"[mail] Body:\n{body[:3000]}")
                await browser.close()
                return False

            # Unescape HTML entities
            verify_link = (verify_link
                           .replace("&amp;", "&")
                           .replace("&#x2F;", "/")
                           .replace("&#39;", "'")
                           .replace("&quot;", '"'))
            print(f"[mail] Verify link: {verify_link[:150]}...")

            # 7. Visit verification link in a new tab
            verify_page = await context.new_page()
            await verify_page.goto(verify_link, timeout=30000)
            await verify_page.wait_for_timeout(5000)
            await verify_page.screenshot(path="/tmp/fpt_verify.png")

            verify_text = await verify_page.inner_text("body")
            print(f"[verify] Page text:\n{verify_text[:1500]}")

            # Some verify pages have a "Confirm" button
            for btn_text in ["Confirm", "Verify", "Continue"]:
                try:
                    await verify_page.click(f"button:has-text('{btn_text}'), a:has-text('{btn_text}'), input[value='{btn_text}']", timeout=3000)
                    print(f"[verify] Clicked {btn_text}")
                    await verify_page.wait_for_timeout(5000)
                    break
                except:
                    continue

            print(f"[verify] Final URL: {verify_page.url}")
            await verify_page.screenshot(path="/tmp/fpt_verify_done.png")

            # Save state
            state = load_state()
            state.update(creds)
            state["verified"] = True
            save_state(state)
            await context.storage_state(path=str(STORAGE_FILE))

            await browser.close()
            return True

        # No verification needed — we're logged in already
        if "ai.fptcloud.com" in page.url:
            print("[signup] Logged in directly")
            await context.storage_state(path=str(STORAGE_FILE))
            state = load_state()
            state.update(creds)
            save_state(state)
            await browser.close()
            return True

        print(f"[signup] Unexpected state: {page_text[:1000]}")
        await browser.close()
        return False


if __name__ == "__main__":
    result = asyncio.run(signup_and_verify())
    print(f"\n=== Result: {result} ===")
