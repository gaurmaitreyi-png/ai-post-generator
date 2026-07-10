"""
One-time helper to get a LinkedIn access token + your person URN for the bot.

Prerequisites (the few clicks only you can do):
  1. Go to https://developer.linkedin.com/ -> Create app (link it to any Company Page).
  2. On the app's "Products" tab, request "Share on LinkedIn" and
     "Sign In with LinkedIn using OpenID Connect" (both are usually instant).
  3. On the "Auth" tab, under "Authorized redirect URLs", add EXACTLY:
         http://localhost:8080/callback
  4. Copy the app's Client ID and Client Secret into your .env:
         LINKEDIN_CLIENT_ID=xxxxxxxx
         LINKEDIN_CLIENT_SECRET=xxxxxxxx

Then just run:  python get_linkedin_token.py
A browser opens, you click "Allow", and this script prints the lines to paste
into .env (LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN).
"""

import os
import sys
import secrets
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8080/callback"
# openid + profile let us fetch your person URN; w_member_social lets the bot post.
SCOPES = "openid profile w_member_social"

STATE = secrets.token_urlsafe(16)
_auth_code = {"code": None, "error": None}


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("state", [""])[0] != STATE:
            _auth_code["error"] = "State mismatch (possible CSRF). Aborted."
        elif "error" in params:
            _auth_code["error"] = params.get("error_description", params["error"])[0]
        else:
            _auth_code["code"] = params.get("code", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        msg = "✅ Got it! You can close this tab and return to the terminal."
        if _auth_code["error"]:
            msg = f"❌ {_auth_code['error']}"
        self.wfile.write(f"<html><body style='font-family:sans-serif'><h2>{msg}</h2></body></html>".encode())

    def log_message(self, *args):
        pass  # keep the terminal clean


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        sys.exit(
            "❌ LINKEDIN_CLIENT_ID / LINKEDIN_CLIENT_SECRET are not set in .env.\n"
            "   Create the app first (see the instructions at the top of this file)."
        )

    auth_url = "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "state": STATE,
        }
    )

    print("🌐 Opening LinkedIn in your browser. Click 'Allow' to authorize...")
    print(f"   (If it doesn't open, paste this URL manually:)\n   {auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for the single callback request.
    server = HTTPServer(("localhost", 8080), CallbackHandler)
    while _auth_code["code"] is None and _auth_code["error"] is None:
        server.handle_request()
    server.server_close()

    if _auth_code["error"]:
        sys.exit(f"❌ Authorization failed: {_auth_code['error']}")

    print("🔑 Exchanging code for an access token...")
    token_res = requests.post(
        "https://www.linkedin.com/oauth/v2/accessToken",
        data={
            "grant_type": "authorization_code",
            "code": _auth_code["code"],
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if token_res.status_code != 200:
        sys.exit(f"❌ Token exchange failed: {token_res.status_code} {token_res.text}")
    access_token = token_res.json().get("access_token")
    if not access_token:
        sys.exit(f"❌ No access_token in response: {token_res.text}")

    print("👤 Fetching your LinkedIn member id...")
    me = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    if me.status_code != 200:
        sys.exit(f"❌ Could not fetch userinfo: {me.status_code} {me.text}")
    sub = me.json().get("sub")
    if not sub:
        sys.exit(f"❌ No 'sub' (member id) in userinfo: {me.text}")
    person_urn = f"urn:li:person:{sub}"

    print("\n" + "=" * 60)
    print("✅ SUCCESS — paste these two lines into your .env file:\n")
    print(f"LINKEDIN_ACCESS_TOKEN={access_token}")
    print(f"LINKEDIN_PERSON_URN={person_urn}")
    print("=" * 60)
    print("\nNote: LinkedIn access tokens expire (~60 days). Re-run this script to refresh.")


if __name__ == "__main__":
    main()
