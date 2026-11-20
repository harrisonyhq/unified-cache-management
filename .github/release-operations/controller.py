"""Deterministic release operations shared by Actions and the JSON CLI."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from associations import PullRequestAssociations
from checks import ReleaseChecks
from operations import Operations, tracked
from ownership import manual_mode, milestone_number
from pinning import ReleasePinning
from presentation import release_body
from syntax import (
    OperationError,
    bind_entries,
    field,
    issue_reference,
    parse_release,
    patch_references,
    references,
    scope_diff,
)

SUMMARY_MARKER = "<!-- ucm-release-summary:v1 -->"
STATE_RE = re.compile(r"<!-- ucm-release-state:v1 ([A-Za-z0-9+/=]+) -->")
ORIGIN = "<!-- ucm-origin:{release}:{key} -->"


def bug_progress(closed: bool, merged: set[str], has_pr: bool) -> tuple[str, str]:
    if closed:
        return "status:verified", "Done"
    if merged:
        return "status:verification", "In review"
    if has_pr:
        return "status:fixing", "In progress"
    return "status:triage", "Ready"


class GitHub:
    def __init__(self, repository: str):
        self.repository = repository

    def call(self, path: str, method="GET", data=None, project=False):
        args = ["gh", "api", path, "--method", method]
        if data is not None:
            args += ["--input", "-"]
        env = dict(os.environ)
        if project:
            token = env.get("UCM_PROJECT_TOKEN")
            if not token:
                raise OperationError("配置 UCM_PROJECT_TOKEN 后启用 Project 同步")
            env["GH_TOKEN"] = token
        result = subprocess.run(
            args,
            input=None if data is None else json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
            text=True,
            capture_output=True,
            env=env,
            timeout=90,
        )
        if result.returncode:
            raise OperationError(result.stderr.strip() or result.stdout.strip())
        parsed = json.loads(result.stdout) if result.stdout.strip() else None
        if isinstance(parsed, dict) and parsed.get("errors"):
            raise OperationError(json.dumps(parsed["errors"], ensure_ascii=False))
        return parsed

    def rest(self, path: str, method="GET", data=None):
        endpoint = "repos/" + self.repository
        return self.call(endpoint + ("/" + path if path else ""), method, data)

    def pages(self, path: str) -> list:
        items = []
        for page in range(1, 101):
            part = self.rest(
                path + ("&" if "?" in path else "?") + f"per_page=100&page={page}"
            )
            if not isinstance(part, list):
                raise OperationError("预期分页数组：" + path)
            items.extend(part)
            if len(part) < 100:
                return items
        raise OperationError("分页超出处理上限")

    def graphql(self, query: str, **variables):
        return self.call(
            "graphql", "POST", {"query": query, "variables": variables}, project=True
        )["data"]


class Controller(PullRequestAssociations, ReleasePinning, Operations, ReleaseChecks):
    def __init__(self, api: GitHub, config: dict):
        self.api, self.config = api, config
        self.repository = config["repository"]
        self.run_url = (
            os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            + "/"
            + self.repository
            + "/actions/runs/"
            + os.environ.get("GITHUB_RUN_ID", "local")
        )

    def issue(self, number: int):
        issue = self.api.rest(f"issues/{number}")
        if "pull_request" in issue:
            raise OperationError(f"#{number} 是 PR，请引用 Issue")
        return issue

    def load(self, release: int):
        comments = self.api.pages(f"issues/{release}/comments")
        candidates = [
            c
            for c in comments
            if SUMMARY_MARKER in c["body"]
            and c["user"]["login"] == "github-actions[bot]"
        ]
        if len(candidates) > 1:
            raise OperationError("发现多条机器汇总，请先处理重复评论")
        comment = candidates[0] if candidates else None
        state = {
            "schema": 1,
            "release": release,
            "scope": {},
            "objects": {},
            "bugs": {},
            "prs": {},
            "revision": 0,
            "history": [],
        }
        if comment and (match := STATE_RE.search(comment["body"])):
            state = json.loads(base64.b64decode(match[1]).decode("utf-8"))
        state.pop("readiness", None)
        state.pop("verification", None)
        state.get("metadata", {}).pop("goal", None)
        # Read historical snapshots without replaying the old requirement grammar.
        state["scope"] = {
            f"issue-{e['number']}": dict(
                {
                    k: v
                    for k, v in e.items()
                    if k not in ("parent", "verifier", "acceptance")
                },
                key=f"issue-{e['number']}",
                description=e.get("description", e.get("acceptance", "")),
            )
            for e in state["scope"].values()
        }
        return comment, state

    def save(self, release: int, state: dict, result: str):
        if (
            state.get("pin_warning")
            and ("Release pin: " + state["pin_warning"]) not in result
        ):
            result += "\n\nRelease pin: " + state["pin_warning"]
        state["last_result"] = result

        def checkpoint():
            serialized = base64.b64encode(
                json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode()
            ).decode()
            body = SUMMARY_MARKER + f"\n<!-- ucm-release-state:v1 {serialized} -->"
            if len(body) > 60000:
                raise OperationError("State checkpoint reached the comment capacity")
            return body

        comment, _ = self.load(release)
        if comment:
            comment = self.api.rest(
                f"issues/comments/{comment['id']}", "PATCH", {"body": checkpoint()}
            )
        else:
            comment = self.api.rest(
                f"issues/{release}/comments", "POST", {"body": checkpoint()}
            )
        attempted = getattr(self, "_minimize_attempted", None)
        if attempted is None:
            attempted = self._minimize_attempted = set()
        if (
            not state.get("checkpoint_minimized")
            and comment.get("node_id")
            and comment["id"] not in attempted
        ):
            attempted.add(comment["id"])
            try:
                self.api.call(
                    "graphql",
                    "POST",
                    {
                        "query": "mutation($id:ID!){minimizeComment(input:{subjectId:$id,classifier:OUTDATED}){minimizedComment{isMinimized}}}",
                        "variables": {"id": comment["node_id"]},
                    },
                )
                state["checkpoint_minimized"] = True
                self.api.rest(
                    f"issues/comments/{comment['id']}", "PATCH", {"body": checkpoint()}
                )
            except OperationError as error:
                print(f"Checkpoint comment could not be collapsed: {error}")
        self.sync_release_body(release, state)

    def sync_release_body(self, release, state):
        source = self.issue(release)
        patched = release_body(source["body"], state)
        if len(patched) > 60000:
            raise OperationError("Release body reached the managed display limit")
        if patched != source["body"]:
            if self.issue(release)["body"] != source["body"]:
                raise OperationError(
                    "Release body changed during rendering; retry synchronization"
                )
            self.api.rest(f"issues/{release}", "PATCH", {"body": patched})

    def validate(self, parsed: dict, release: int, state: dict):
        if state.get("version") and state["version"] != parsed["version"]:
            raise OperationError("版本号已绑定，请创建另一版本入口")
        if not parsed["entries"] and not state.get("initialized"):
            raise OperationError("初始化至少填写一个需求")
        diff = scope_diff(state["scope"], parsed["entries"])
        if diff["removed"] and not parsed["change_note"]:
            raise OperationError(
                "Fill reason: in Change Log when removing requirements"
            )
        for role in ("release_manager", "test_lead"):
            login = parsed["metadata"].get(role, "").lstrip("@")
            if login:
                self.api.call("users/" + login)
        self.api.rest("commits/" + parsed["metadata"]["branch"])
        for entry in parsed["entries"]:
            for login in (entry["developer"],):
                if login:
                    self.api.call("users/" + login)
            existing_number = entry["number"] or state["objects"].get(entry["key"])
            if existing_number:
                existing = self.issue(existing_number)
                if existing["number"] == release or existing["number"] == state.get(
                    "archive"
                ):
                    raise OperationError("管理容器请使用独立条目")
                target = milestone_number(existing)
                if target and existing["milestone"]["title"] != parsed["version"]:
                    raise OperationError(
                        f"#{existing['number']} 已关联其他 Milestone，请维护者处理"
                    )
                desired = entry["developer"]
                assigned = {person["login"] for person in existing["assignees"]}
                previous_developer = (
                    state["scope"].get(entry["key"], {}).get("developer")
                )
                if (
                    desired
                    and assigned
                    and desired not in assigned
                    and assigned != {previous_developer}
                ):
                    raise OperationError(f"#{existing['number']} 的负责人存在冲突")
        return diff

    def create_or_recover(
        self,
        release: int,
        key: str,
        title: str,
        body: str,
        labels: list[str],
        state: dict,
    ):
        if key in state["objects"]:
            return self.issue(state["objects"][key])
        marker = ORIGIN.format(release=release, key=key)
        matches = [
            i
            for i in self.api.pages("issues?state=all")
            if marker in (i.get("body") or "") and "pull_request" not in i
        ]
        if len(matches) > 1:
            raise OperationError("来源标记对应多个对象：" + key)
        issue = (
            matches[0]
            if matches
            else self.api.rest(
                "issues",
                "POST",
                {"title": title, "body": marker + "\n\n" + body, "labels": labels},
            )
        )
        state["objects"][key] = issue["number"]
        self.save(release, state, f"执行中：已记录 `{key}` → #{issue['number']}")
        return issue

    def parent(self, issue_number: int):
        # GET /parent returns 404 for an Issue with no parent.
        try:
            return self.api.rest(f"issues/{issue_number}/parent")
        except OperationError as error:
            if "404" in str(error):
                return None
            raise

    def attach(self, child: dict, parent: int, old_parent: int | None = None):
        current = self.parent(child["number"])
        if current and current["number"] == parent:
            return
        if current and current["number"] != old_parent:
            raise OperationError(
                f"#{child['number']} 已属于父需求 #{current['number']}"
            )
        self.api.rest(
            f"issues/{parent}/sub_issues",
            "POST",
            {"sub_issue_id": child["id"], "replace_parent": bool(current)},
        )
        actual = self.parent(child["number"])
        if not actual or actual["number"] != parent:
            raise OperationError("父关系回读失败")

    @tracked("initialize")
    def initialize_or_update(self, release: int, command: str, dry_run=False):
        source = self.issue(release)
        if "release" not in {label["name"] for label in source["labels"]}:
            raise OperationError("版本入口需要 release 标签")
        _, state = self.load(release)
        try:
            parsed = parse_release(source["body"], self.repository)
            source_entries = parsed["entries"]
            parsed["entries"] = bind_entries(source_entries, state["objects"])
            if command == "update" and not state.get("initialized"):
                raise OperationError("先执行 /release_init")
            diff = self.validate(parsed, release, state)
            if state.get("initialized") and state.get("metadata") != parsed["metadata"]:
                diff["metadata"] = ["Release Information"]
            plan = json.dumps(diff, ensure_ascii=False)
            if dry_run:
                actions = [
                    f"- {'Reuse' if e['number'] else 'Create'} Issue: {e['title']}"
                    + (f" (#{e['number']})" if e["number"] else "")
                    for e in parsed["entries"]
                ]
                self.save(
                    release,
                    state,
                    f"预览 `{command}`\n\n"
                    + "\n".join(actions)
                    + f"\n\n```json\n{plan}\n```",
                )
                return
            if (
                command == "update"
                and state.get("initialized")
                and not any(diff.values())
            ):
                self.release_pin(release, state)
                self.save(release, state, "Release pin checked")
                self.reconcile(release)
                _, state = self.load(release)
                self.save(release, state, "范围一致，已完成状态对账。")
                return
            if command == "release_init" and state.get("initialized"):
                self.release_pin(release, state)
                self.save(
                    release,
                    state,
                    "初始化已完成。范围变化通过 `/update` 同步。\n\n" + plan,
                )
                return
            self.save(release, state, f"执行中：`/{command}`\n\n{plan}")
            state["version"] = parsed["version"]
            state["metadata"] = parsed["metadata"]
            state["schema"] = 2
            if not state.get("milestone"):
                matches = [
                    m
                    for m in self.api.pages("milestones?state=all")
                    if m["title"] == parsed["version"]
                ]
                state["milestone"] = (
                    matches[0]
                    if matches
                    else self.api.rest(
                        "milestones",
                        "POST",
                        {
                            "title": parsed["version"],
                            "description": f"Release tracking: #{release}",
                        },
                    )
                )
                self.save(release, state, "执行中：Milestone 已记录")
            target = state["milestone"]["number"]
            metadata = parsed["metadata"]
            milestone_fields = {
                "description": f"Release tracking: #{release}\nRelease manager: {metadata.get('release_manager', '')}\nTest lead: {metadata.get('test_lead', '')}",
                "due_on": (
                    metadata["release_date"] + "T23:59:59Z"
                    if metadata.get("release_date") not in (None, "", "TBD")
                    else None
                ),
            }
            state["milestone"] = self.api.rest(
                f"milestones/{target}", "PATCH", milestone_fields
            )
            release_fields = {"milestone": target}
            if metadata.get("release_manager"):
                release_fields["assignees"] = [metadata["release_manager"].lstrip("@")]
            self.api.rest(f"issues/{release}", "PATCH", release_fields)
            archive = self.create_or_recover(
                release,
                "__bugs__",
                f"[Bug Archive] UCM {parsed['version']}",
                f"Source version: {parsed['version']}\nRelease: #{release}\n\nCreate Bug sub-issues here. The default fix version is {parsed['version']}.",
                ["release-bug-archive"],
                state,
            )
            state["archive"] = archive["number"]
            self.attach(archive, release)
            self.api.rest(f"issues/{archive['number']}", "PATCH", {"milestone": target})
            desired = {}
            for entry in parsed["entries"]:
                key = entry["key"]
                item = (
                    self.issue(entry["number"])
                    if entry["number"]
                    else self.create_or_recover(
                        release,
                        key,
                        entry["title"],
                        f"Release: #{release}\n\n{entry['description']}",
                        ["requirement"],
                        state,
                    )
                )
                source_key = key
                key = f"issue-{item['number']}"
                entry = dict(entry, key=key, number=item["number"])
                state["objects"][source_key] = item["number"]
                state["objects"][key] = item["number"]
                desired[key] = entry
                previous = state["scope"].get(key, {})
                payload = {"milestone": target}
                if entry["developer"] or previous.get("developer"):
                    payload["assignees"] = (
                        [entry["developer"]] if entry["developer"] else []
                    )
                self.api.rest(f"issues/{item['number']}", "PATCH", payload)
                self.save(release, state, f"执行中：已关联 `{key}` → #{item['number']}")
            for key in diff["removed"]:
                item = self.issue(state["scope"][key]["number"])
                if milestone_number(item) == target:
                    self.api.rest(
                        f"issues/{item['number']}", "PATCH", {"milestone": None}
                    )
            latest = self.issue(release)
            latest_parsed = parse_release(latest["body"], self.repository)
            latest_parsed["entries"] = bind_entries(
                latest_parsed["entries"], state["objects"]
            )
            expected = dict(
                parsed, entries=bind_entries(source_entries, state["objects"])
            )
            if latest_parsed != expected:
                raise OperationError(
                    "执行期间正文已修改；已记录成功对象，请重新执行同步"
                )
            patched = patch_references(
                latest["body"], state["objects"], self.repository
            )
            if patched != latest["body"]:
                self.api.rest(f"issues/{release}", "PATCH", {"body": patched})
            # Keep the actual created title as a display value while key/number identify scope.
            state["scope"] = desired
            state["release_check"] = {"status": "not_checked"}
            state["initialized"] = True
            state["revision"] += 1
            state["history"].append(
                f"- r{state['revision']} `/{command}`：{plan}；{parsed['change_note']}"
            )
            self.release_pin(release, state)
            self.save(
                release,
                state,
                f"已完成：`/{command}`，范围修订 r{state['revision']}\n\n{plan}",
            )
            self.reconcile(release)
        except Exception as error:
            self.save(
                release, state, "执行失败，已完成对象可重试恢复：\n\n" + str(error)
            )
            raise

    def project_item(
        self,
        issue: dict,
        project_name: str,
        status=None,
    ):
        project = self.config["projects"][project_name]
        item = self.api.graphql(
            """mutation($project:ID!,$content:ID!){addProjectV2ItemById(input:{projectId:$project,contentId:$content}){item{id}}}""",
            project=project["id"],
            content=issue["node_id"],
        )["addProjectV2ItemById"]["item"]
        if status:
            spec = project["fields"]["Status"]
            options = {o["name"]: o["id"] for o in spec["options"]}
            self.api.graphql(
                """mutation($input:UpdateProjectV2ItemFieldValueInput!){updateProjectV2ItemFieldValue(input:$input){projectV2Item{id}}}""",
                input={
                    "projectId": project["id"],
                    "itemId": item["id"],
                    "fieldId": spec["id"],
                    "value": {"singleSelectOptionId": options[status]},
                },
            )

    @tracked("Release synchronization")
    def reconcile(self, release: int, event=None):
        _, state = self.load(release)
        if not state.get("initialized"):
            return
        problems = []
        target = state["milestone"]["number"]
        human = event and event.get("sender", {}).get("type") == "User"
        event_issue = (event or {}).get("issue", {})
        event_pr = (event or {}).get("pull_request", {})
        milestone_event = (event or {}).get("action") in ("milestoned", "demilestoned")
        for bug in self.api.pages("issues?state=all&labels=bug"):
            if "pull_request" in bug:
                continue
            source = field(bug.get("body") or "", "source_bug_archive") or field(
                bug.get("body") or "", "源版本主 Bug Issue"
            )
            try:
                archive_number = (
                    issue_reference(source, self.repository) if source else None
                )
            except OperationError:
                problems.append(f"Bug #{bug['number']} 的源版本待归档")
                continue
            if archive_number == state["archive"]:
                self.attach(bug, state["archive"])
        bugs = self.api.pages(f"issues/{state['archive']}/sub_issues")
        for bug in bugs:
            number = str(bug["number"])
            current = milestone_number(bug)
            record = state["bugs"].get(number)
            if record is None:
                record = {
                    "mode": "auto" if current in (None, target) else "manual",
                    "last": current,
                }
            else:
                record = manual_mode(
                    record,
                    current,
                    bool(
                        human
                        and milestone_event
                        and event_issue.get("number") == bug["number"]
                    ),
                )
            record = self.milestone_history(bug["number"], record)
            if record["mode"] == "auto" and current != target:
                self.api.rest(f"issues/{number}", "PATCH", {"milestone": target})
                record["last"] = target
                bug["milestone"] = state["milestone"]
            state["bugs"][number] = record
            record["target_title"] = (bug.get("milestone") or {}).get("title")
            state.setdefault("observed", {})[number] = {
                "state": bug["state"],
                "assignees": [p["login"] for p in bug.get("assignees", [])],
                "milestone": record["last"],
            }
            if "bug" not in {label["name"] for label in bug["labels"]}:
                self.api.rest(f"issues/{number}/labels", "POST", {"labels": ["bug"]})
        self.save(release, state, "同步中：Bug 默认版本与人工指定已记录")
        problems.extend(self.sync_pull_requests(release, state, event))
        for bug in bugs:
            number = str(bug["number"])
            related = {
                key: value
                for key, value in state["prs"].items()
                if bug["number"] in value.get("bugs", [])
            }
            merged = {key for key, value in related.items() if value.get("merged")}
            status_label, project_status = bug_progress(
                bug["state"] == "closed", merged, bool(related)
            )
            current_labels = [label["name"] for label in bug["labels"]]
            if status_label not in current_labels:
                labels = [
                    label for label in current_labels if not label.startswith("status:")
                ]
                self.api.rest(
                    f"issues/{number}",
                    "PATCH",
                    {
                        "labels": (
                            labels + ["bug", status_label]
                            if "bug" not in labels
                            else labels + [status_label]
                        )
                    },
                )
            self.project_item(bug, "Bugfix", status=project_status)
        for entry in state["scope"].values():
            issue = self.issue(entry["number"])
            state.setdefault("observed", {})[str(issue["number"])] = {
                "state": issue["state"],
                "assignees": [p["login"] for p in issue.get("assignees", [])],
                "milestone": milestone_number(issue),
            }
        state["sync_warnings"] = problems
        self.save(
            release,
            state,
            "同步完成：Bug 默认版本、人工目标与修复 PR 已对账。\n\n"
            + "\n".join(problems),
        )

    def releases(self):
        return [
            i["number"]
            for i in self.api.pages("issues?state=all&labels=release")
            if "pull_request" not in i
        ]

    def milestone_history(self, number: int, record: dict):
        changes = [
            event
            for event in self.api.pages(f"issues/{number}/events")
            if event["event"] in ("milestoned", "demilestoned")
        ]
        if "event_id" in record:
            later = [event for event in changes if event["id"] > record["event_id"]]
            if any(event.get("actor", {}).get("type") == "User" for event in later):
                record["mode"] = "manual"
        record["event_id"] = max((event["id"] for event in changes), default=0)
        return record

    def authorize(self, actor: str):
        permission = self.api.rest(f"collaborators/{actor}/permission")["permission"]
        if permission not in ("admin", "maintain", "write"):
            raise OperationError("此操作需要仓库 write、maintain 或 admin 权限")

    def drain_commands(self, release: int, ceiling: int | None = None):
        for comment in sorted(
            self.api.pages(f"issues/{release}/comments"), key=lambda item: item["id"]
        ):
            if ceiling is not None and comment["id"] >= ceiling:
                break
            if comment["user"]["type"] != "User":
                continue
            if (
                not comment["body"]
                .strip()
                .startswith(
                    (
                        "/release_init",
                        "/update",
                        "/release_check",
                    )
                )
            ):
                continue
            _, state = self.load(release)
            if comment["id"] <= state.get("command_cursor", 0) or str(
                comment["id"]
            ) in state.get("processed_comments", []):
                continue
            try:
                self.handle(
                    {
                        "issue": {"number": release},
                        "comment": comment,
                        "sender": comment["user"],
                    },
                    "issue_comment",
                    replay=True,
                )
            except OperationError as error:
                print(f"Command #{comment['id']} requires attention: {error}")
                _, state = self.load(release)
                state["command_cursor"] = max(
                    state.get("command_cursor", 0), comment["id"]
                )
                self.save(release, state, f"命令 #{comment['id']} 待人工处理：{error}")

    def handle(self, event: dict, event_name: str, replay=False):
        if event_name == "issue_comment":
            if "pull_request" in event.get("issue", {}):
                old_body = event.get("changes", {}).get("body", {}).get("from", "")
                if references(
                    event["comment"].get("body", "") + "\n" + old_body, self.repository
                ):
                    self.acknowledge_comment(event)
                self.reconcile_all(event)
                return
            command = event["comment"]["body"].strip()
            if not command.startswith(
                (
                    "/release_init",
                    "/update",
                    "/release_check",
                )
            ):
                return
            actor = event["sender"]["login"]
            self.authorize(actor)
            self.acknowledge_comment(event)
            release = event["issue"]["number"]
            if not replay:
                self.drain_commands(release, ceiling=event["comment"]["id"])
            _, before = self.load(release)
            comment_id = str(event["comment"]["id"])
            if int(comment_id) <= before.get(
                "command_cursor", 0
            ) or comment_id in before.get("processed_comments", []):
                return
            kind = {
                "/release_init": "Release initialization",
                "/update": "Release synchronization",
                "/release_check": "Release check",
            }.get(command.split()[0], "Release operation")
            if "--dry-run" in command:
                kind += " preview"
            try:
                with self.operation(release, kind):
                    if match := re.fullmatch(
                        r"/(release_init|update)( --dry-run)?", command
                    ):
                        self.initialize_or_update(release, match[1], bool(match[2]))
                    elif command == "/release_check":
                        self.release_check(release)
                    else:
                        raise OperationError("命令格式错误")
            except Exception as error:
                _, failed = self.load(release)
                self.save(release, failed, f"执行失败：`{command}`\n\n{error}")
                raise
            _, latest = self.load(release)
            latest.setdefault("processed_comments", []).append(comment_id)
            latest["processed_comments"] = latest["processed_comments"][-500:]
            latest["command_cursor"] = max(
                latest.get("command_cursor", 0), int(comment_id)
            )
            self.save(release, latest, latest.get("last_result", ""))
        else:
            self.reconcile_all(event)

    def acknowledge_comment(self, event):
        comment = event.get("comment", {})
        author = comment.get("user", event.get("sender", {}))
        if (
            event.get("action") == "deleted"
            or author.get("type") == "Bot"
            or not comment.get("id")
        ):
            return
        try:
            # GitHub reuses an existing reaction by this Bot on repeated requests.
            self.api.rest(
                f"issues/comments/{comment['id']}/reactions",
                "POST",
                {"content": "eyes"},
            )
        except OperationError as error:
            print(f"Comment #{comment['id']} acknowledgement unavailable: {error}")

    def reconcile_all(self, event=None, replay_commands=True):
        errors = []
        self._pr_snapshot = self.api.pages("pulls?state=all")
        self._discussion_snapshot = {}
        self._native_issue_snapshot = {}
        for release in self.releases():
            try:
                if replay_commands:
                    self.drain_commands(release)
                self.reconcile(release, event)
            except OperationError as error:
                _, state = self.load(release)
                self.save(
                    release, state, "Synchronization requires attention: " + str(error)
                )
                errors.append(f"Release #{release}: {error}")
        self._pr_snapshot = None
        self._discussion_snapshot = None
        self._native_issue_snapshot = None
        if errors:
            raise OperationError("\n".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--event", type=Path, default=os.environ.get("GITHUB_EVENT_PATH")
    )
    parser.add_argument(
        "--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "schedule")
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    repository = os.environ.get("GITHUB_REPOSITORY", config["repository"])
    if repository != config["repository"]:
        raise OperationError("部署仓库与 config.json 的 repository 不一致")
    event = json.loads(args.event.read_text(encoding="utf-8")) if args.event else {}
    controller = Controller(GitHub(repository), config)
    if args.finalize:
        controller.finalize_operations(os.environ.get("UCM_JOB_STATUS", "failure"))
    else:
        controller.handle(event, args.event_name)
    print(
        json.dumps(
            {"status": "completed", "event": args.event_name, "repository": repository}
        )
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
