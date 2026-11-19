"""Bot-owned Release body regions, separate from human-authored scope."""

import json

from syntax import OperationError

NAV_START = "<!-- ucm-release-navigation:start -->"
NAV_END = "<!-- ucm-release-navigation:end -->"
OVERVIEW_START = "<!-- ucm-release-overview:start -->"
OVERVIEW_END = "<!-- ucm-release-overview:end -->"


def replace_region(body, start, end, content):
    if body.count(start) != body.count(end) or body.count(start) > 1:
        raise OperationError("Invalid Bot region markers: " + start)
    if start in body:
        a, b = body.index(start), body.index(end)
        if b < a:
            raise OperationError("Invalid Bot region order: " + start)
        body = body[:a] + body[b + len(end) :]
    return body, start + "\n" + content + "\n" + end


def release_body(body, state):
    if "### Release Information" in body:
        navigation = []
        if state.get("milestone"):
            navigation.append(
                f"**Milestone:** [{state.get('version', '')}]({state['milestone']['html_url']})"
            )
        if state.get("archive"):
            navigation.append(f"**Bug Archive:** #{state['archive']}")
        body, nav = replace_region(
            body,
            NAV_START,
            NAV_END,
            " · ".join(navigation) or "Version links will appear after initialization.",
        )
        heading = "### Release Information"
        index = body.index(heading) + len(heading)
        body = body[:index] + "\n\n" + nav + "\n\n" + body[index:].lstrip("\r\n")

    rows = []
    entries = {str(e["number"]): e for e in state["scope"].values()}
    entries.update({n: entries.get(n, {}) for n in state.get("bugs", {})})
    for number, entry in entries.items():
        observed = state.get("observed", {}).get(number, {})
        owners = observed.get(
            "assignees", ([entry["developer"]] if entry.get("developer") else [])
        )
        prs = []
        for pr, record in state.get("prs", {}).items():
            if int(number) in record.get("requirements", []) + record.get("bugs", []):
                status = (
                    "merged"
                    if record.get("merged")
                    else (
                        "closed"
                        if record.get("state") == "closed"
                        else "draft" if record.get("draft") else "open"
                    )
                )
                prs.append(f"#{pr} ({status})")
        validation = (
            state.get("verification", {}).get(number, {}).get("result", "Pending")
        )
        rows.append(
            f"| #{number} | {', '.join(prs) or '-'} | {', '.join('@' + owner for owner in owners) or '-'} | {observed.get('state', 'open')} / {validation} |"
        )
    overview = (
        "### Release Overview\n\nAutomatically maintained by the release bot.\n\n| Issue | Pull requests | Assignee | State / Validation |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
    )
    if state.get("readiness"):
        overview += (
            "\n\n<details>\n<summary>Release readiness</summary>\n\n```json\n"
            + json.dumps(state["readiness"], ensure_ascii=False)
            + "\n```\n\n</details>"
        )
    if state.get("history"):
        overview += (
            "\n\n<details>\n<summary>Scope history</summary>\n\n"
            + "\n".join(state["history"][-8:])
            + "\n\n</details>"
        )
    body, managed = replace_region(body, OVERVIEW_START, OVERVIEW_END, overview)
    return body.rstrip() + "\n\n" + managed + "\n"
