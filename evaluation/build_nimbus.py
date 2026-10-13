"""Build the Nimbus evaluation set (BEIR format) from the source below.

Nimbus Drive is a fictional cloud-storage product. The 31 help-center articles,
40 questions and relevance judgements were written for this repository. Grades:
2 = the document answers the question, 1 = related and useful.

Questions deliberately mix wordings close to the article ("How much does the Plus
plan cost?") with paraphrases that share few words with it ("Can I open my
documents on the plane without wifi?"), so lexical and semantic retrieval can be
told apart.

    uv run python evaluation/build_nimbus.py
"""

from __future__ import annotations

import json
from pathlib import Path

ARTICLES: dict[str, tuple[str, str]] = {
    "restore-deleted": (
        "Recovering deleted files",
        "Files you delete are moved to the Trash, where they stay for 30 days on the Free and "
        "Plus plans and for 180 days on the Business plan.\n\nTo bring a file back, open Trash in "
        "the web app, select the file and choose Restore. The file returns to its original "
        "folder. If that folder no longer exists, Nimbus recreates it.\n\nAfter the retention "
        "period the file is purged permanently and support cannot recover it.",
    ),
    "version-history": (
        "Version history",
        "Every time a file changes, Nimbus keeps the previous revision. You can open the Versions "
        "panel from the file menu to compare revisions and roll back to an earlier one.\n\nPlus "
        "keeps 100 revisions per file for 90 days. Business keeps unlimited revisions for one "
        "year. Rolling back creates a new revision, so no history is lost.",
    ),
    "plans-pricing": (
        "Plans and storage limits",
        "Nimbus offers three plans. Free includes 5 GB of storage. Plus includes 2 TB and costs "
        "9.99 euros per month. Business includes 5 TB per user, with a minimum of three users, "
        "and costs 15 euros per user per month.\n\nAnnual billing gives two months free on Plus "
        "and Business.",
    ),
    "quota-full": (
        "What happens when your storage is full",
        "When you reach your quota, existing files stay available but syncing stops: new or "
        "changed files are not uploaded until you free up space or upgrade.\n\nShared folders "
        "count against the owner's quota only. Files in the Trash still count until they are "
        "purged, so emptying the Trash is the quickest way to recover space.",
    ),
    "upgrade-downgrade": (
        "Changing your plan",
        "You can upgrade at any time from Settings > Billing; the price difference is prorated "
        "for the current billing period.\n\nDowngrading takes effect at the end of the billing "
        "period. If your files exceed the new quota, your account becomes read-only until you "
        "are back under the limit.",
    ),
    "cancel-refund": (
        "Cancelling your subscription and refunds",
        "Cancel from Settings > Billing > Cancel subscription. Your paid features remain active "
        "until the end of the period you have paid for, then the account reverts to Free.\n\n"
        "Annual subscriptions cancelled within 14 days of purchase are refunded in full. Monthly "
        "subscriptions are not refunded.",
    ),
    "two-factor": (
        "Two-step verification",
        "Two-step verification protects your account with a second factor in addition to your "
        "password. Nimbus supports authenticator apps (TOTP) and hardware security keys "
        "(FIDO2/WebAuthn). SMS codes are not offered because they are vulnerable to SIM "
        "swapping.\n\nWhen you turn it on, Nimbus shows ten single-use recovery codes. Store them "
        "somewhere safe: they are the only way back in if you lose your device.",
    ),
    "lost-2fa-device": (
        "Lost your phone or security key",
        "If you can no longer produce a verification code, sign in with one of your recovery "
        "codes and register a new device.\n\nWithout recovery codes, an identity check by "
        "support is required. It takes up to five business days and cannot be shortened, "
        "because it protects you against account takeover.",
    ),
    "password-reset": (
        "Resetting a forgotten password",
        "Use Forgot password on the sign-in page. Nimbus emails a reset link that expires after "
        "one hour.\n\nIf end-to-end encrypted vaults are enabled, resetting the password without "
        "your recovery key makes the vault contents unreadable, since Nimbus never holds the key.",
    ),
    "encryption": (
        "How your data is encrypted",
        "All files are encrypted in transit with TLS 1.3 and at rest with AES-256. Encryption "
        "keys are managed by Nimbus in a hardware security module.\n\nFor sensitive documents "
        "you can create an encrypted Vault folder. Vault files are encrypted on your device "
        "before upload (end-to-end encryption), so Nimbus staff and infrastructure cannot read "
        "them. Vault files cannot be previewed in the browser or searched by content.",
    ),
    "data-location": (
        "Where your data is stored",
        "Business customers choose the region where their files are stored: European Union "
        "(Frankfurt and Paris) or United States. The choice is made when the workspace is "
        "created and applies to files and backups.\n\nFree and Plus accounts are stored in the "
        "EU region. Nimbus does not move data between regions without the customer's request.",
    ),
    "gdpr-export": (
        "Exporting or deleting your personal data",
        "You can download a copy of all your files and account data from Settings > Privacy > "
        "Export. The archive is prepared within 48 hours and the download link stays valid for "
        "seven days.\n\nTo erase your account permanently, use Delete account. Files are removed "
        "immediately from active storage and from backups within 30 days.",
    ),
    "sharing-links": (
        "Sharing files with a link",
        "Select a file or folder and choose Share > Copy link. Anyone with the link can view the "
        "content by default. You can restrict the link to people in your organization, protect "
        "it with a password, or set an expiry date.\n\nPassword-protected and expiring links are "
        "available on Plus and Business.",
    ),
    "sharing-permissions": (
        "Inviting people to a folder",
        "Invite collaborators by email to a shared folder and choose a role: Viewer can open and "
        "download, Commenter can also comment, Editor can add, change and delete files.\n\nOnly "
        "the owner can delete the shared folder itself or transfer ownership. Removing someone "
        "from a folder revokes access immediately on all their devices.",
    ),
    "external-sharing-policy": (
        "Controlling external sharing (Business)",
        "Administrators can limit sharing outside the organization from the admin console: allow "
        "it, allow it only with password-protected links, or block it entirely. A list of "
        "trusted domains can be exempted.\n\nThe policy applies to new links; existing public "
        "links can be revoked in bulk from the Sharing report.",
    ),
    "selective-sync": (
        "Choosing which folders sync to a computer",
        "Selective sync lets you keep some folders only in the cloud to save disk space. Open "
        "the desktop app preferences, go to Sync and untick the folders you do not need "
        "locally.\n\nFiles On-Demand goes further: everything appears in your file explorer, but "
        "content downloads only when you open a file.",
    ),
    "sync-conflicts": (
        "Understanding conflicted copies",
        "If the same file is edited on two devices before they sync, Nimbus keeps both versions. "
        "The second one is saved next to the original with the name of the device and the date, "
        "for example report (laptop's conflicted copy 2026-03-02).\n\nOpen both versions, merge "
        "the changes manually, then delete the copy you no longer need.",
    ),
    "sync-troubleshooting": (
        "The desktop app is not syncing",
        "First check the status icon: a pause symbol means sync was paused, a red cross means an "
        "error. Common causes are a full quota, a file name with characters not allowed on the "
        "other operating system, a path longer than 260 characters on Windows, or a file larger "
        "than 50 GB.\n\nIf the app is stuck, quit it and start it again; it will rescan your "
        "folders without uploading everything again.",
    ),
    "bandwidth": (
        "Limiting upload and download speed",
        "On slow or metered connections you can cap bandwidth in the desktop app preferences "
        "under Network. By default uploads are limited automatically to leave room for other "
        "applications, and downloads are not limited.",
    ),
    "supported-platforms": (
        "Supported operating systems",
        "The desktop app runs on Windows 10 and 11, macOS 13 or later, and Ubuntu 22.04 or "
        "later. Mobile apps are available for iOS 16+ and Android 10+.\n\nThe web app works in "
        "the last two versions of Chrome, Firefox, Safari and Edge.",
    ),
    "mobile-camera-upload": (
        "Backing up photos from your phone",
        "Turn on Camera uploads in the mobile app to back up new photos and videos "
        "automatically. You can choose to upload only on Wi-Fi and only while charging.\n\n"
        "Photos are stored in the Camera Uploads folder and count against your storage quota.",
    ),
    "offline-mobile": (
        "Using files without an internet connection on mobile",
        "Mark a file or folder as Available offline in the mobile app. Nimbus downloads it and "
        "keeps it updated whenever the phone is online. Offline files are stored encrypted in "
        "the app's private storage.",
    ),
    "file-size-limits": (
        "File size and type limits",
        "Through the desktop and mobile apps you can upload files up to 50 GB each. Uploads in "
        "the web browser are limited to 10 GB per file.\n\nThere is no restriction on file "
        "types, but executable files shared by public link are scanned for malware before "
        "others can download them.",
    ),
    "admin-console": (
        "Getting started with the admin console",
        "Business administrators manage users, groups, billing and security policies from the "
        "admin console. You can add users one by one, import a CSV file, or provision them "
        "automatically with SCIM from your identity provider.\n\nRemoved users' files are "
        "transferred to an administrator you choose, so nothing is lost when someone leaves.",
    ),
    "sso": (
        "Single sign-on with SAML or OpenID Connect",
        "Business workspaces can require single sign-on through any SAML 2.0 or OpenID Connect "
        "identity provider, such as Microsoft Entra ID, Okta, Google Workspace or Keycloak.\n\n"
        "Once SSO is enforced, passwords stored by Nimbus are disabled and two-step verification "
        "is handled by your identity provider.",
    ),
    "audit-log": (
        "Audit log",
        "The audit log records sign-ins, sharing changes, downloads, deletions and admin actions "
        "with user, IP address and time. Business administrators can search it in the admin "
        "console, export it as CSV, or stream it to a SIEM through the events API.\n\nEntries "
        "are kept for 12 months.",
    ),
    "api-access": (
        "Developer API",
        "The Nimbus REST API lets applications upload, download and search files. Requests are "
        "authenticated with OAuth 2.0; personal access tokens are available for scripts.\n\nThe "
        "default rate limit is 600 requests per minute per user. When it is exceeded the API "
        "answers with HTTP 429 and a Retry-After header indicating how many seconds to wait.",
    ),
    "webhooks": (
        "Webhooks",
        "Webhooks notify your application when files are created, modified, deleted or shared. "
        "Nimbus sends a signed POST request to your endpoint; verify the HMAC-SHA256 signature "
        "in the X-Nimbus-Signature header before trusting the payload.\n\nFailed deliveries are "
        "retried with exponential backoff for up to 24 hours.",
    ),
    "search": (
        "Searching your files",
        "Search looks at file names and, for documents, PDFs and images with text, at their "
        "content. Results appear as you type and can be filtered by type, owner and "
        "modification date.\n\nNew files are searchable within a few minutes. Files in an "
        "encrypted Vault are searchable by name only.",
    ),
    "support-hours": (
        "Contacting support",
        "Free users can get help from the community forum and the help center. Plus subscribers "
        "can open email tickets, answered within one business day. Business customers also have "
        "phone support from 8:00 to 20:00 CET on weekdays and a 4-hour response target for "
        "urgent issues.\n\nThe service status and incident history are published on "
        "status.nimbus.example.",
    ),
    "uptime-sla": (
        "Service level agreement",
        "Nimbus commits to 99.9 % monthly availability for Business workspaces. If availability "
        "falls below the commitment, administrators can claim service credits from 10 % to 30 % "
        "of the monthly fee, depending on the downtime.\n\nScheduled maintenance is announced at "
        "least 72 hours in advance and does not count as downtime.",
    ),
}

QUESTIONS: list[tuple[str, str, dict[str, int]]] = [
    ("q1", "How long are deleted files kept before they are gone for good?",
     {"restore-deleted": 2}),
    ("q2", "I removed a document by mistake, can I get it back?",
     {"restore-deleted": 2, "version-history": 1}),
    ("q3", "How do I go back to an older version of a spreadsheet?", {"version-history": 2}),
    ("q4", "How much does the Plus plan cost?", {"plans-pricing": 2}),
    ("q5", "What happens if I run out of space?", {"quota-full": 2, "upgrade-downgrade": 1}),
    ("q6", "Do shared folders use my storage?", {"quota-full": 2}),
    ("q7", "What happens to my files if I switch to a cheaper plan?", {"upgrade-downgrade": 2}),
    ("q8", "Can I get my money back after buying a yearly subscription?", {"cancel-refund": 2}),
    ("q9", "Why can't I receive login codes by text message?", {"two-factor": 2}),
    ("q10", "I lost my phone and cannot log in because of 2FA",
     {"lost-2fa-device": 2, "two-factor": 1}),
    ("q11", "The password reset email link does not work anymore", {"password-reset": 2}),
    ("q12", "Can Nimbus employees read my files?", {"encryption": 2}),
    ("q13", "Is my data stored in Europe?", {"data-location": 2}),
    ("q14", "How do I download all my data for a GDPR request?", {"gdpr-export": 2}),
    ("q15", "How long does it take for deleted account data to disappear from backups?",
     {"gdpr-export": 2}),
    ("q16", "Can I put a password on a link I send to a client?",
     {"sharing-links": 2, "external-sharing-policy": 1}),
    ("q17", "What is the difference between viewer and editor?", {"sharing-permissions": 2}),
    ("q18", "How can an admin stop employees from sharing files with outsiders?",
     {"external-sharing-policy": 2}),
    ("q19", "My laptop disk is almost full, can I keep some folders only online?",
     {"selective-sync": 2}),
    ("q20", "Why is there a file called conflicted copy in my folder?", {"sync-conflicts": 2}),
    ("q21", "The sync icon shows a red cross", {"sync-troubleshooting": 2}),
    ("q22", "Files with very long paths are not uploading on Windows",
     {"sync-troubleshooting": 2}),
    ("q23", "Nimbus uses all my internet bandwidth", {"bandwidth": 2}),
    ("q24", "Does the desktop client work on Linux?", {"supported-platforms": 2}),
    ("q25", "How do I automatically back up pictures from my iPhone?",
     {"mobile-camera-upload": 2}),
    ("q26", "Can I open my documents on the plane without wifi?", {"offline-mobile": 2}),
    ("q27", "What is the maximum size of a video I can upload from the browser?",
     {"file-size-limits": 2}),
    ("q28", "How do we create accounts for all our employees at once?", {"admin-console": 2}),
    ("q29", "Can we log in with our company Okta accounts?", {"sso": 2}),
    ("q30", "Who downloaded a confidential file last week?", {"audit-log": 2}),
    ("q31", "The API returns error 429", {"api-access": 2}),
    ("q32", "How do I check that a webhook request really comes from Nimbus?", {"webhooks": 2}),
    ("q33", "Can I search inside the text of PDF files?", {"search": 2}),
    ("q34", "Why can't I find files from my vault by their contents?",
     {"search": 2, "encryption": 1}),
    ("q35", "Is there phone support for urgent problems?", {"support-hours": 2}),
    ("q36", "What compensation do we get if the service goes down?", {"uptime-sla": 2}),
    ("q37", "Is planned maintenance counted as downtime?", {"uptime-sla": 2}),
    ("q38", "What happens to the files of an employee who leaves the company?",
     {"admin-console": 2}),
    ("q39", "How much storage do I get for free?", {"plans-pricing": 2}),
    ("q40", "Which login methods protect my account better than a password alone?",
     {"two-factor": 2, "sso": 1}),
]  # fmt: skip


def build(out: Path) -> None:
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    with (out / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for doc_id, (title, text) in ARTICLES.items():
            f.write(json.dumps({"_id": doc_id, "title": title, "text": text}) + "\n")
    with (out / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for query_id, text, _ in QUESTIONS:
            f.write(json.dumps({"_id": query_id, "text": text}) + "\n")
    with (out / "qrels" / "test.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, _, judgements in QUESTIONS:
            for doc_id, grade in judgements.items():
                if doc_id not in ARTICLES:
                    raise KeyError(f"{query_id} judges unknown document {doc_id}")
                f.write(f"{query_id}\t{doc_id}\t{grade}\n")


if __name__ == "__main__":
    build(Path(__file__).parent / "datasets" / "nimbus")
