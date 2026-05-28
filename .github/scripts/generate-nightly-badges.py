#!/usr/bin/env python3
"""Generate package-level status badges for the nightly workflow."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

PACKAGES = (
    "essreduce",
    "essimaging",
    "essnmx",
    "essdiffraction",
    "essreflectometry",
    "esssans",
    "essspectroscopy",
)

BADGE_GROUPS = (
    {
        "slug": "test",
        "job_prefix": "Nightly ",
        "label": "nightly",
        "package_label": "{package} nightly",
    },
    {
        "slug": "lower-bound",
        "job_prefix": "Lower bound ",
        "label": "lower bound",
        "package_label": "{package} lower bound",
    },
    {
        "slug": "latest-dependencies",
        "job_prefix": "Latest dependencies ",
        "label": "latest deps",
        "package_label": "{package} latest deps",
    },
)

STATUS = {
    "success": ("passing", "#2ea44f"),
    "failure": ("failing", "#d73a49"),
    "startup_failure": ("failing", "#d73a49"),
    "timed_out": ("timed out", "#d73a49"),
    "action_required": ("action required", "#bf8700"),
    "cancelled": ("cancelled", "#6e7781"),
    "skipped": ("skipped", "#6e7781"),
    "neutral": ("neutral", "#6e7781"),
    "missing": ("missing", "#6e7781"),
    "unknown": ("unknown", "#6e7781"),
}

FAILURE_CONCLUSIONS = {
    "failure",
    "startup_failure",
    "timed_out",
    "action_required",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--jobs-json", type=Path)
    return parser.parse_args()


def request_json(url: str, token: str) -> dict[str, Any]:
    if not url.startswith("https://api.github.com/"):
        raise ValueError(f"Unexpected API URL: {url}")
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "ess-nightly-badge-generator",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            return json.load(response)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API request failed: {err.code} {detail}") from err


def fetch_jobs(
    repo: str, run_id: str, run_attempt: str | None, token: str
) -> list[dict[str, Any]]:
    base_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
    if run_attempt:
        base_url = f"{base_url}/attempts/{run_attempt}"
    return request_json(f"{base_url}/jobs?per_page=100", token).get("jobs", [])


def conclusion(job: dict[str, Any] | None) -> str:
    if job is None:
        return "missing"
    return job.get("conclusion") or "unknown"


def aggregate(conclusions: list[str]) -> str:
    if any(item in FAILURE_CONCLUSIONS for item in conclusions):
        return "failure"
    for status in ("cancelled", "missing", "unknown", "skipped", "neutral"):
        if status in conclusions:
            return status
    if conclusions and all(item == "success" for item in conclusions):
        return "success"
    return "unknown"


def text_width(text: str) -> int:
    # Close enough to Shields-style badges without requiring font metrics.
    return max(28, 6 * len(text) + 10)


def badge_svg(label: str, status: str) -> str:
    message, color = STATUS.get(status, STATUS["unknown"])
    label_width = text_width(label)
    message_width = text_width(message)
    width = label_width + message_width
    label_text_x = label_width / 2
    message_text_x = label_width + message_width / 2
    title = escape(f"{label}: {message}")
    label = escape(label)
    message = escape(message)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20"
  role="img" aria-label="{title}">
  <title>{title}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{message_width}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle"
     font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_text_x:.1f}" y="15" fill="#010101"
      fill-opacity=".3">{label}</text>
    <text x="{label_text_x:.1f}" y="14">{label}</text>
    <text x="{message_text_x:.1f}" y="15" fill="#010101"
      fill-opacity=".3">{message}</text>
    <text x="{message_text_x:.1f}" y="14">{message}</text>
  </g>
</svg>
"""


def write_badge(path: Path, label: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(badge_svg(label, status), encoding="utf-8")


def jobs_by_name(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {job["name"]: job for job in jobs if "name" in job}


def main() -> int:
    args = parse_args()
    if args.jobs_json:
        jobs = json.loads(args.jobs_json.read_text(encoding="utf-8"))["jobs"]
    else:
        missing = [
            name
            for name, value in (("repo", args.repo), ("run-id", args.run_id))
            if not value
        ]
        token = os.environ.get("GITHUB_TOKEN")
        if missing or not token:
            raise SystemExit("Missing GitHub context or GITHUB_TOKEN")
        jobs = fetch_jobs(args.repo, args.run_id, args.run_attempt, token)

    by_name = jobs_by_name(jobs)
    for group in BADGE_GROUPS:
        group_dir = args.output_dir / group["slug"]
        group_conclusions: list[str] = []
        for package in PACKAGES:
            job_name = f"{group['job_prefix']}{package}"
            status = conclusion(by_name.get(job_name))
            group_conclusions.append(status)
            write_badge(
                group_dir / f"{package}.svg",
                group["package_label"].format(package=package),
                status,
            )
        write_badge(
            args.output_dir / f"{group['slug']}.svg",
            group["label"],
            aggregate(group_conclusions),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
