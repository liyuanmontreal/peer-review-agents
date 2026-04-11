"""
FPT AI Factory GPU Container Manager.

Automates login, container creation (2xH100), SSH testing, and cleanup.
Account creation requires reCAPTCHA and must be done manually at https://ai.fptcloud.com.

Usage:
    python fpt_manage.py login <username> <password>
    python fpt_manage.py apply-referral
    python fpt_manage.py create [--gpu 1|2|3|4] [--template pytorch|jupyter|ubuntu]
    python fpt_manage.py list
    python fpt_manage.py ssh-cmd
    python fpt_manage.py test-ssh
    python fpt_manage.py delete [container_name]

State is saved to fpt_state.json. Browser session is saved to fpt_browser_state.json.
SSH keys are stored in .ssh/ within this skill directory.
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from playwright.async_api import async_playwright

SKILL_DIR = Path(__file__).parent
STATE_FILE = SKILL_DIR / "fpt_state.json"
STORAGE_FILE = SKILL_DIR / "fpt_browser_state.json"
SSH_DIR = SKILL_DIR / ".ssh"
SSH_KEY = SSH_DIR / "id_ed25519"
SSH_PUBKEY = SSH_DIR / "id_ed25519.pub"
FPT_URL = "https://ai.fptcloud.com"
REFERRAL_CODE = "SIVAREDDYPROF-BNI8WX1W5X"

TEMPLATE_MAP = {
    "pytorch": "NVIDIA Pytorch",
    "jupyter": "Jupyter Notebook",
    "ubuntu": "Ubuntu 24.04",
    "tensorflow": "Tensorflow",
    "cuda": "Nvidia Cuda",
    "vllm": "vllm-openai",
    "ollama": "ollama",
    "code-server": "Code Server",
}


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_ssh_key():
    """Generate an SSH key pair in .ssh/ if one doesn't exist. Returns the public key."""
    SSH_DIR.mkdir(exist_ok=True)
    if not SSH_KEY.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(SSH_KEY),
             "-N", "", "-C", "fpt-cloud-skill"],
            check=True, capture_output=True,
        )
        print(f"Generated SSH key: {SSH_KEY}")
    return SSH_PUBKEY.read_text().strip()


async def get_org_slug(page):
    """Navigate to FPT and return the org slug from the URL."""
    if FPT_URL not in page.url:
        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
    url = page.url.rstrip("/")
    parts = url.split("/")
    if len(parts) >= 4 and parts[3].startswith("AI-"):
        return parts[3]
    return None


# ── Login ──────────────────────────────────────────────────────────────

async def cmd_login(username, password):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"Logging in as {username}...")
        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)

        await page.click("button:has-text('Sign in')")
        await page.wait_for_timeout(3000)
        await page.click("button:has-text('Continue with FPT ID')")
        await page.wait_for_timeout(5000)

        await page.fill("#username", username)
        await page.fill("#password", password)
        await page.click("button:has-text('Sign in')")
        await page.wait_for_timeout(8000)

        if "ai.fptcloud.com" in page.url and "id.fptcloud.com" not in page.url:
            org_slug = await get_org_slug(page)
            await context.storage_state(path=str(STORAGE_FILE))
            state = load_state()
            state["username"] = username
            state["password"] = password
            state["org_slug"] = org_slug
            save_state(state)
            print(f"Login successful. Org: {org_slug}")
            await browser.close()
            return True
        else:
            page_text = await page.inner_text("body")
            print(f"Login failed. Page:\n{page_text[:500]}")
            await browser.close()
            return False


# ── Apply referral ─────────────────────────────────────────────────────

async def cmd_apply_referral():
    """Apply the referral code via Billing → Add Voucher section."""
    if not STORAGE_FILE.exists():
        print("No session. Run 'login' first.")
        return False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(STORAGE_FILE))
        page = await context.new_page()

        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)

        org_slug = await get_org_slug(page)
        if not org_slug:
            print("Not logged in. Run 'login' first.")
            await browser.close()
            return False

        # Billing page has "Add Voucher" section with an input + "Apply Code" button
        await page.goto(f"{FPT_URL}/{org_slug}/billing", timeout=30000)
        await page.wait_for_timeout(5000)

        print(f"Applying referral code: {REFERRAL_CODE}")

        # The voucher input is a text input near "Apply Code" button.
        # Find it by locating the Apply Code button and going to the nearest input.
        try:
            # Scroll voucher section into view
            await page.evaluate("""
                const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Apply Code');
                if (btn) btn.scrollIntoView({behavior: 'instant', block: 'center'});
            """)
            await page.wait_for_timeout(1000)

            # Fill the voucher input (the one paired with "Apply Code" button)
            voucher_input = page.locator("input.semi-input-large").nth(0)
            # Better: find the input that's in the same row as Apply Code button
            filled = await page.evaluate(f"""
                (code) => {{
                    const btns = Array.from(document.querySelectorAll('button'));
                    const applyBtn = btns.find(b => b.textContent.trim() === 'Apply Code');
                    if (!applyBtn) return false;
                    // Walk up to find a common parent with an input
                    let parent = applyBtn.parentElement;
                    for (let i = 0; i < 5; i++) {{
                        const input = parent.querySelector('input[type="text"]');
                        if (input) {{
                            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                            setter.call(input, code);
                            input.dispatchEvent(new Event('input', {{bubbles: true}}));
                            input.dispatchEvent(new Event('change', {{bubbles: true}}));
                            return true;
                        }}
                        if (parent.parentElement) parent = parent.parentElement;
                        else break;
                    }}
                    return false;
                }}
            """, REFERRAL_CODE)

            if not filled:
                print("Could not fill voucher input")
                await page.screenshot(path="/tmp/fpt_voucher_fail.png")
                await browser.close()
                return False

            print(f"  Filled voucher input")
            await page.wait_for_timeout(1000)

            # Click Apply Code
            await page.click("button:has-text('Apply Code')")
            await page.wait_for_timeout(5000)
            await page.screenshot(path="/tmp/fpt_voucher_applied.png")

            page_text = await page.inner_text("body")
            # Check for success/error toast
            if "success" in page_text.lower()[:1500] or "applied" in page_text.lower()[:1500] or "$100" in page_text or "100.00" in page_text:
                print("  Referral code applied successfully!")
            else:
                print(f"  Page after apply:\n{page_text[:1500]}")

            state = load_state()
            state["referral_applied"] = True
            save_state(state)
            await context.storage_state(path=str(STORAGE_FILE))
            await browser.close()
            return True

        except Exception as e:
            print(f"Error applying referral: {e}")
            await page.screenshot(path="/tmp/fpt_voucher_error.png")
            await context.storage_state(path=str(STORAGE_FILE))
            await browser.close()
            return False


# ── Create container ───────────────────────────────────────────────────

async def cmd_create(gpu_count=2, template="pytorch"):
    if not STORAGE_FILE.exists():
        print("No session. Run 'login' first.")
        return False

    ssh_pubkey = ensure_ssh_key()
    template_name = TEMPLATE_MAP.get(template, template)
    container_name = f"claude-{uuid.uuid4().hex[:6]}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(STORAGE_FILE))
        page = await context.new_page()

        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)

        org_slug = await get_org_slug(page)
        if not org_slug:
            print("Not logged in. Run 'login' first.")
            await browser.close()
            return False

        print(f"Creating container: {container_name}")
        print(f"  GPU: {gpu_count}xH100 | Template: {template_name}")

        await page.goto(f"{FPT_URL}/{org_slug}/gpu-containers/new", timeout=30000)
        await page.wait_for_timeout(5000)

        # 1. Set container name
        name_input = await page.wait_for_selector("input[name='name']", timeout=5000)
        await name_input.fill("")
        await name_input.fill(container_name)
        print(f"  Container name: {container_name}")

        # 2. Select GPU count
        await page.click(f"text={gpu_count}xH100")
        await page.wait_for_timeout(1000)
        print(f"  Selected {gpu_count}xH100")

        # 3. Change template to one that supports SSH (e.g. NVIDIA Pytorch)
        await page.click("button:has-text('Change template')")
        await page.wait_for_timeout(3000)
        try:
            await page.click(f"span:has-text('{template_name}')")
            await page.wait_for_timeout(2000)
            print(f"  Template: {template_name}")
        except:
            print(f"  WARNING: Could not find template '{template_name}', using default")

        # 4. Enable SSH Terminal Access and add SSH key
        print("  Enabling SSH Terminal Access...")
        await page.evaluate("""
            const el = document.querySelector('input[name="ssh_terminal_access"]');
            if (el) el.scrollIntoView({behavior: 'instant', block: 'center'});
        """)
        await page.wait_for_timeout(500)
        await page.locator("input[name='ssh_terminal_access']").click(force=True)
        await page.wait_for_timeout(2000)

        # 5. Click the "+" button inside #ssh-key to open "Create SSH Key" modal
        print("  Opening SSH key dialog...")
        await page.click("#ssh-key div.mt-4.flex span.cursor-pointer")
        await page.wait_for_timeout(3000)

        # Fill the "Create SSH Key" modal
        # - SSH Key Name input (placeholder: "Input SSH key name")
        # - SSH Public Key textarea (placeholder: "Input SSH public key")
        try:
            modal = page.locator(".semi-modal")
            await modal.locator("input[placeholder*='SSH key name' i]").fill("claude-code")
            await page.wait_for_timeout(500)
            await modal.locator("textarea[placeholder*='SSH public key' i], textarea").first.fill(ssh_pubkey)
            await page.wait_for_timeout(500)
            await modal.locator("button:has-text('Add SSH Key')").click()
            await page.wait_for_timeout(3000)
            print("  SSH key added successfully")
        except Exception as e:
            print(f"  WARNING: Could not add SSH key via modal: {e}")
            # Try broader selectors
            try:
                await page.fill("input[placeholder*='key name' i]", "claude-code")
                await page.fill("textarea", ssh_pubkey)
                await page.click("button:has-text('Add SSH Key')")
                await page.wait_for_timeout(3000)
                print("  SSH key added (fallback)")
            except Exception as e2:
                print(f"  ERROR: SSH key add failed: {e2}")

        # 6. Set persistent volume
        try:
            vol_input = await page.wait_for_selector("input[name='persistant_volume']", timeout=3000)
            await vol_input.fill("300")
            print("  Volume: 300GB")

            path_input = await page.wait_for_selector("input[name='mount_path']", timeout=3000)
            await path_input.fill("/workspace")
            print("  Mount path: /workspace")
        except Exception as e:
            print(f"  WARNING: Could not set volume: {e}")

        await page.wait_for_timeout(1000)
        await page.screenshot(path="/tmp/fpt_before_create.png")

        # 7. Click Create Container
        print("  Submitting...")
        await page.click("button:has-text('Create Container')")
        await page.wait_for_timeout(10000)

        current_url = page.url
        print(f"  URL after create: {current_url}")
        await page.screenshot(path="/tmp/fpt_after_create.png")

        # If still on /new, form validation failed
        if current_url.endswith("/new"):
            page_text = await page.inner_text("body")
            print(f"  Form may have validation errors. Page text:\n{page_text[:1000]}")

        # Navigate to container list to find our container
        await page.goto(f"{FPT_URL}/{org_slug}/gpu-containers", timeout=30000)
        await page.wait_for_timeout(5000)

        page_text = await page.inner_text("body")
        if container_name in page_text:
            print("  Container creation initiated!")

            # Wait for container to be Running (poll)
            print("  Waiting for container to start...")
            container_full_name = None
            for attempt in range(30):  # up to ~2.5 minutes
                page_text = await page.inner_text("body")

                # Find our container's row and status
                match = re.search(rf'({re.escape(container_name)}-\w+)\s.*?(Running|Creating|Pending|Failed)', page_text, re.DOTALL)
                if not match:
                    # Try simpler: find full name, then find status nearby
                    name_match = re.search(rf'({re.escape(container_name)}-\w+)', page_text)
                    if name_match:
                        container_full_name = name_match.group(1)
                        if "Running" in page_text:
                            print(f"  Container {container_full_name} is RUNNING!")
                            break
                        elif "Creating" in page_text:
                            print(f"  Status: Creating (attempt {attempt + 1}/30)")
                        else:
                            print(f"  Waiting... (attempt {attempt + 1}/30)")
                    else:
                        print(f"  Waiting... (attempt {attempt + 1}/30)")
                else:
                    container_full_name = match.group(1)
                    status = match.group(2)
                    if status == "Running":
                        print(f"  Container {container_full_name} is RUNNING!")
                        break
                    elif status == "Failed":
                        print(f"  Container FAILED!")
                        await browser.close()
                        return False
                    else:
                        print(f"  Status: {status} (attempt {attempt + 1}/30)")

                await page.wait_for_timeout(5000)
                await page.reload()
                await page.wait_for_timeout(3000)

            if not container_full_name:
                match = re.search(rf'({re.escape(container_name)}-\w+)', page_text)
                if match:
                    container_full_name = match.group(1)

            # Click into container detail to get SSH port
            if container_full_name:
                await page.click(f"text={container_full_name}")
                await page.wait_for_timeout(5000)

                page_text = await page.inner_text("body")
                port_match = re.search(r'tcp-endpoint\.serverless\.fptcloud\.com:(\d+)', page_text)
                if port_match:
                    port = port_match.group(1)
                    print(f"\n  SSH port: {port}")
                    print(f"  SSH command: ssh root@tcp-endpoint.serverless.fptcloud.com -p {port} -i {SSH_KEY}")

                    state = load_state()
                    state["container_name"] = container_full_name
                    state["ssh_port"] = port
                    state["org_slug"] = org_slug
                    save_state(state)
                else:
                    print("  Could not extract SSH port from detail page.")
                    state = load_state()
                    state["container_name"] = container_full_name
                    state["org_slug"] = org_slug
                    save_state(state)
        else:
            print(f"  Container '{container_name}' not found in container list. Creation may have failed.")

        await context.storage_state(path=str(STORAGE_FILE))
        await browser.close()
        return True


# ── List containers ────────────────────────────────────────────────────

async def cmd_list():
    if not STORAGE_FILE.exists():
        print("No session. Run 'login' first.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(STORAGE_FILE))
        page = await context.new_page()

        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)

        org_slug = await get_org_slug(page)
        if not org_slug:
            print("Not logged in.")
            await browser.close()
            return

        await page.goto(f"{FPT_URL}/{org_slug}/gpu-containers", timeout=30000)
        await page.wait_for_timeout(5000)

        page_text = await page.inner_text("body")
        await page.screenshot(path="/tmp/fpt_container_list.png")
        print(page_text[:4000])

        await context.storage_state(path=str(STORAGE_FILE))
        await browser.close()


# ── Get SSH command ────────────────────────────────────────────────────

async def cmd_ssh_cmd():
    state = load_state()
    if "ssh_port" in state:
        port = state["ssh_port"]
        print(f"ssh root@tcp-endpoint.serverless.fptcloud.com -p {port} -i {SSH_KEY} -o ProxyJump=none -o StrictHostKeyChecking=no")
        return

    if not STORAGE_FILE.exists():
        print("No session and no saved SSH port. Run 'login' then 'create' first.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(STORAGE_FILE))
        page = await context.new_page()

        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)

        org_slug = await get_org_slug(page)
        await page.goto(f"{FPT_URL}/{org_slug}/gpu-containers", timeout=30000)
        await page.wait_for_timeout(5000)

        # Click on the first container
        try:
            rows = await page.query_selector_all("tr")
            for row in rows[1:]:
                text = await row.inner_text()
                if text.strip():
                    await row.click()
                    await page.wait_for_timeout(5000)
                    break
        except:
            pass

        page_text = await page.inner_text("body")
        port_match = re.search(r'tcp-endpoint\.serverless\.fptcloud\.com:(\d+)', page_text)
        if port_match:
            port = port_match.group(1)
            print(f"ssh root@tcp-endpoint.serverless.fptcloud.com -p {port} -i {SSH_KEY} -o ProxyJump=none -o StrictHostKeyChecking=no")
            state["ssh_port"] = port
            save_state(state)
        else:
            print("Could not find SSH port. Container may not be running.")

        await context.storage_state(path=str(STORAGE_FILE))
        await browser.close()


# ── Test SSH ───────────────────────────────────────────────────────────

def cmd_test_ssh():
    state = load_state()
    port = state.get("ssh_port")
    if not port:
        print("No SSH port saved. Run 'ssh-cmd' first.")
        return False

    ensure_ssh_key()

    cmd = [
        "ssh", "root@tcp-endpoint.serverless.fptcloud.com",
        "-p", str(port),
        "-i", str(SSH_KEY),
        "-o", "ProxyJump=none",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "nvidia-smi",
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"SUCCESS!\n{result.stdout}")
            return True
        else:
            print(f"FAILED (exit {result.returncode})")
            if result.stderr:
                print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT: SSH connection timed out after 30s")
        return False


# ── Delete container ───────────────────────────────────────────────────

async def cmd_delete(container_name=None):
    if not STORAGE_FILE.exists():
        print("No session. Run 'login' first.")
        return False

    state = load_state()
    if not container_name:
        container_name = state.get("container_name")
    if not container_name:
        print("No container name. Pass it as argument or run 'create' first.")
        return False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(STORAGE_FILE))
        page = await context.new_page()

        await page.goto(FPT_URL, timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)

        org_slug = await get_org_slug(page)
        if not org_slug:
            print("Not logged in.")
            await browser.close()
            return False

        print(f"Deleting container: {container_name}")
        await page.goto(f"{FPT_URL}/{org_slug}/gpu-containers", timeout=30000)
        await page.wait_for_timeout(5000)

        page_text = await page.inner_text("body")
        if container_name not in page_text:
            print(f"Container '{container_name}' not found on page.")
            await browser.close()
            return False

        # Click into container detail
        await page.click(f"text={container_name}")
        await page.wait_for_timeout(3000)

        # Click Delete button on detail page
        try:
            await page.click("button:has-text('Delete')", timeout=5000)
            print("Clicked 'Delete'")
            await page.wait_for_timeout(3000)
            await page.screenshot(path="/tmp/fpt_delete_confirm.png")

            # Confirm deletion
            for confirm in ["Confirm", "Yes", "Delete", "OK"]:
                try:
                    await page.click(f"button:has-text('{confirm}')", timeout=3000)
                    print(f"Confirmed with '{confirm}'")
                    break
                except:
                    continue

            await page.wait_for_timeout(5000)
            print("Container deletion initiated.")

            for key in ["container_name", "ssh_port"]:
                state.pop(key, None)
            save_state(state)

            await context.storage_state(path=str(STORAGE_FILE))
            await browser.close()
            return True
        except Exception as e:
            print(f"Could not delete: {e}")
            await context.storage_state(path=str(STORAGE_FILE))
            await browser.close()
            return False


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FPT AI Factory GPU Container Manager")
    sub = parser.add_subparsers(dest="command")

    login_p = sub.add_parser("login")
    login_p.add_argument("username")
    login_p.add_argument("password")

    sub.add_parser("apply-referral")

    create_p = sub.add_parser("create")
    create_p.add_argument("--gpu", type=int, default=2, choices=[1, 2, 3, 4])
    create_p.add_argument("--template", default="pytorch",
                          choices=list(TEMPLATE_MAP.keys()))

    sub.add_parser("list")
    sub.add_parser("ssh-cmd")
    sub.add_parser("test-ssh")

    delete_p = sub.add_parser("delete")
    delete_p.add_argument("container_name", nargs="?")

    args = parser.parse_args()

    if args.command == "login":
        asyncio.run(cmd_login(args.username, args.password))
    elif args.command == "apply-referral":
        asyncio.run(cmd_apply_referral())
    elif args.command == "create":
        asyncio.run(cmd_create(gpu_count=args.gpu, template=args.template))
    elif args.command == "list":
        asyncio.run(cmd_list())
    elif args.command == "ssh-cmd":
        asyncio.run(cmd_ssh_cmd())
    elif args.command == "test-ssh":
        cmd_test_ssh()
    elif args.command == "delete":
        asyncio.run(cmd_delete(args.container_name))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
