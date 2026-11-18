"""Structured Markdown syntax for the Release Checklist."""

import hashlib
import re
from datetime import date

VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?\Z")
REF_RE = re.compile(
    r"(?<![\w/])(?:https://github\.com/([\w.-]+/[\w.-]+)/issues/|"
    r"(?:([\w.-]+/[\w.-]+))?#)(\d+)\b"
)


class OperationError(ValueError):
    pass


def field(body, name):
    match = re.search(
        r"^[ \t]*" + re.escape(name) + r"[ \t]*[:：][ \t]*([^\r\n]*?)[ \t]*$",
        body,
        re.M,
    )
    return match[1] if match else ""


def issue_reference(value, repository):
    matches = list(REF_RE.finditer(value))
    if len(matches) != 1:
        raise OperationError("Expected one Issue reference: " + value)
    match = matches[0]
    target = match[1] or match[2] or repository
    if target.lower() != repository.lower():
        raise OperationError("Cross-repository Issue reference: " + target)
    return int(match[3])


def references(text, repository):
    """Return local references; unrelated cross-repository mentions stay ordinary text."""
    # A PR/document link with a numeric fragment is not an Issue reference.
    text = re.sub(
        r"\[[^\]]*\]\((https?://[^)]+)\)",
        lambda m: (
            m[0]
            if re.match(r"https://github\.com/[^/]+/[^/]+/issues/\d+", m[1])
            else " "
        ),
        text,
    )
    return {
        int(m[3])
        for m in REF_RE.finditer(text)
        if (m[1] or m[2] or repository).lower() == repository.lower()
    }


def sections(body):
    lines = body.splitlines()
    heads = [(i, line[4:]) for i, line in enumerate(lines) if line.startswith("### ")]
    found = {}
    for pos, (start, name) in enumerate(heads):
        if name in found:
            raise OperationError(f"Line {start + 1}: duplicate section {name}")
        end = heads[pos + 1][0] if pos + 1 < len(heads) else len(lines)
        found[name] = (start + 1, end)
    return lines, found


def person(value, location):
    if not value or value == "TBD":
        return None
    if not re.fullmatch(r"@[A-Za-z0-9][A-Za-z0-9-]*", value):
        raise OperationError(f"{location}: use one @username or leave blank")
    return value[1:]


def blocks(body):
    lines, found = sections(body)
    if "Requirements" not in found:
        raise OperationError(
            "Missing '### Requirements'; use the Release Checklist template"
        )
    start, end = found["Requirements"]
    starts = [i for i in range(start, end) if lines[i].startswith("- ")]
    if any(lines[i].strip() for i in range(start, starts[0] if starts else end)):
        raise OperationError(f"Line {start + 1}: requirements must start with '- '")
    result = []
    for pos, first in enumerate(starts):
        last = starts[pos + 1] if pos + 1 < len(starts) else end
        result.append((first, last))
    return lines, result


def parse_release(body, repository):
    lines, found = sections(body)
    if "Release Information" not in found:
        raise OperationError(
            "Missing '### Release Information'; use the Release Checklist template"
        )
    start, end = found["Release Information"]
    info = {}
    allowed = {
        "version",
        "branch",
        "goal",
        "release_manager",
        "test_lead",
        "release_date",
    }
    for i in range(start, end):
        if not lines[i].strip():
            continue
        match = re.fullmatch(r"([a-z_]+):[ ]*(.*)", lines[i])
        if not match or match[1] not in allowed or match[1] in info:
            raise OperationError(f"Line {i + 1}: invalid or duplicate release field")
        info[match[1]] = match[2].strip()
    if not VERSION_RE.fullmatch(info.get("version", "")):
        raise OperationError("version: use a version such as 0.12.0")
    for role in ("release_manager", "test_lead"):
        person(info.get(role, ""), role)
    if not info.get("branch"):
        raise OperationError("branch: specify the release branch")
    if info.get("release_date") not in (None, "", "TBD"):
        try:
            date.fromisoformat(info["release_date"])
        except ValueError:
            raise OperationError("release_date: use YYYY-MM-DD or TBD") from None
    entries, keys = [], set()
    _, ranges = blocks(body)
    for first, last in ranges:

        def fail(message, line=first):
            raise OperationError(
                f"Requirement at line {first + 1}, line {line + 1}: {message}"
            )

        fields = {}
        for i in range(first + 1, last):
            line = lines[i]
            if "\t" in line[: len(line) - len(line.lstrip())]:
                fail("use spaces for indentation", i)
            if not line.strip():
                continue
            if not line.startswith("  "):
                fail("expected indentation of at least two spaces", i)
            if line.startswith("  ") and not line.startswith("   "):
                match = re.fullmatch(
                    r"  (description|assignee|pull_requests):(.*)", line
                )
                if match:
                    if match[1] in fields:
                        fail("duplicate field " + match[1], i)
                    fields[match[1]] = (i, match[2].strip())
                elif fields:
                    fail("unknown field or invalid indentation", i)
        if set(fields) != {"description", "assignee", "pull_requests"}:
            fail("expected description:, assignee:, and pull_requests:")
        desc, owner, prs = (
            fields[name][0] for name in ("description", "assignee", "pull_requests")
        )
        if not first < desc < owner < prs:
            fail("expected description, assignee, pull_requests in that order")
        if fields["description"][1] or fields["pull_requests"][1]:
            fail(
                "put description and pull_requests content on indented following lines"
            )
        title_lines = [lines[first][2:]]
        for i in range(first + 1, desc):
            if lines[i].strip():
                if not lines[i].startswith("  ") or lines[i].startswith("   "):
                    fail("title continuation must use two spaces", i)
                title_lines.append(lines[i].strip())
        title = " ".join(title_lines).strip()
        match = re.fullmatch(r"(.+?)\s+\[([^\[\]]*)\]", title)
        if not match:
            fail("end the title with [] or [Issue reference]")
        title, reference = match[1].strip(), match[2].strip()
        if len(title) > 256:
            fail("title exceeds 256 characters")
        number = issue_reference(reference, repository) if reference else None
        if reference and not re.fullmatch(
            r"(?:#\d+|https://github\.com/[\w.-]+/[\w.-]+/issues/\d+)", reference
        ):
            fail("use #123 or a complete Issue URL inside brackets")
        for a, b in ((desc + 1, owner), (prs + 1, last)):
            for i in range(a, b):
                if lines[i].strip() and not lines[i].startswith("    "):
                    fail("field content must use at least four spaces", i)
        if any(lines[i].strip() for i in range(owner + 1, prs)):
            fail("assignee must be a single line", owner)
        description = "\n".join(
            line[4:] if line else "" for line in lines[desc + 1 : owner]
        ).strip("\n")
        key = (
            f"issue-{number}"
            if number
            else "new-" + hashlib.sha256(title.encode()).hexdigest()[:20]
        )
        if key in keys:
            fail("duplicate Issue reference or new requirement title")
        keys.add(key)
        entries.append(
            {
                "key": key,
                "number": number,
                "title": title,
                "title_text": title,
                "description": description,
                "developer": person(fields["assignee"][1], f"Line {owner + 1}"),
            }
        )
    change_start, change_end = found.get("Change Log", (0, 0))
    return {
        "version": info["version"],
        "metadata": info,
        "entries": entries,
        "change_note": field("\n".join(lines[change_start:change_end]), "reason"),
    }


def bind_entries(entries, objects):
    return [
        dict(
            e,
            number=e["number"] or objects.get(e["key"]),
            key=(
                f"issue-{e['number'] or objects[e['key']]}"
                if e["number"] or e["key"] in objects
                else e["key"]
            ),
        )
        for e in entries
    ]


def scope_diff(previous, entries):
    current = {e["key"]: e for e in entries}
    return {
        "added": sorted(current.keys() - previous.keys()),
        "removed": sorted(previous.keys() - current.keys()),
        "changed": sorted(
            k
            for k in current.keys() & previous.keys()
            if any(
                previous[k].get(f) != current[k].get(f)
                for f in ("title", "description", "developer")
            )
        ),
    }


def patch_references(body, mappings, repository):
    parsed = parse_release(body, repository)
    lines, ranges = blocks(body)
    for entry, (start, end) in zip(parsed["entries"], ranges):
        if entry["number"] or entry["key"] not in mappings:
            continue
        for i in range(start, end):
            if re.search(r"\[\]\s*$", lines[i]):
                lines[i] = re.sub(
                    r"\[\]\s*$",
                    f"[https://github.com/{repository}/issues/{mappings[entry['key']]}]",
                    lines[i],
                )
                break
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def patch_pull_requests(body, rows, repository):
    parsed = parse_release(body, repository)
    lines, ranges = blocks(body)
    for entry, (start, end) in reversed(list(zip(parsed["entries"], ranges))):
        first = next(i for i in range(start, end) if lines[i] == "  pull_requests:")
        suffix = []
        for pr in sorted(rows.get(entry["number"], []), key=lambda p: p["number"]):
            status = (
                "merged"
                if pr.get("merged_at")
                else (
                    "closed"
                    if pr["state"] == "closed"
                    else "draft" if pr.get("draft") else "open"
                )
            )
            suffix.append(
                f"    - [#{pr['number']}](https://github.com/{repository}/pull/{pr['number']}) | {status}"
            )
        lines[first + 1 : end] = suffix + [""]
    return "\n".join(lines).rstrip() + "\n"
