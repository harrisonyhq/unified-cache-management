"""Deterministic release operations shared by Actions and the JSON CLI."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SUMMARY_MARKER = "<!-- ucm-release-summary:v1 -->"
STATE_RE = re.compile(r"<!-- ucm-release-state:v1 ([A-Za-z0-9+/=]+) -->")
ORIGIN = "<!-- ucm-origin:{release}:{key} -->"
KEY_RE = re.compile(r"^(#{3,4})\s+(需求|子需求)\s+([a-zA-Z0-9][a-zA-Z0-9_-]*)\s*$")
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?\Z")
REF_RE = re.compile(r"(?:https://github\.com/([\w.-]+/[\w.-]+)/issues/|(?:(\w[\w.-]*/[\w.-]+))?#)(\d+)")


class OperationError(ValueError):
    pass


def field(body: str, name: str) -> str:
    match = re.search(r"^\s*" + re.escape(name) + r"\s*[:：]\s*(.*?)\s*$", body, re.M)
    return match[1] if match else ""


def issue_reference(value: str, repository: str) -> int | None:
    if re.search(r"https?://", value) and "/issues/" not in value:
        raise OperationError("需求引用必须是 GitHub Issue 链接")
    matches = list(REF_RE.finditer(value))
    if not matches:
        if "#" in value or "/issues/" in value:
            raise OperationError("无效 Issue 引用：" + value)
        return None
    if len(matches) != 1:
        raise OperationError("每项需求填写一个 Issue 引用")
    match = matches[0]
    target = match[1] or match[2] or repository
    if target.lower() != repository.lower():
        raise OperationError("请在目标仓库管理该跨仓库需求：" + target)
    return int(match[3])


def parse_release(body: str, repository: str) -> dict:
    version = field(body, "版本号")
    if not VERSION_RE.fullmatch(version):
        raise OperationError("版本号采用 0.11.0 这样的格式")
    starts = [(i, KEY_RE.fullmatch(line.strip())) for i, line in enumerate(body.splitlines())]
    starts = [(i, match) for i, match in starts if match]
    lines = body.splitlines()
    entries, parent, keys, references = [], None, set(), set()
    for offset, (start, match) in enumerate(starts):
        end = starts[offset + 1][0] if offset + 1 < len(starts) else len(lines)
        section_lines = lines[start + 1:end]
        # A following top-level section terminates the last requirement.
        for index, line in enumerate(section_lines):
            if re.match(r"^#{1,2}\s", line):
                section_lines = section_lines[:index]
                break
        section = "\n".join(section_lines)
        key, kind = match[3], match[2]
        if key in keys:
            raise OperationError("重复条目标识：" + key)
        keys.add(key)
        if kind == "需求":
            if len(match[1]) != 3:
                raise OperationError("父需求使用 ### 需求 <稳定标识>")
            parent = key
            parent_key = None
        else:
            if len(match[1]) != 4 or parent is None:
                raise OperationError("子需求使用 #### 子需求 <稳定标识>，放在父需求下")
            parent_key = parent
        title = field(section, "需求")
        if not title:
            raise OperationError("条目缺少需求标题或引用：" + key)
        number = issue_reference(title, repository)
        if number is not None:
            if number in references:
                raise OperationError("同一 Issue 在范围中出现多次")
            references.add(number)
        people = {}
        for role in ("开发负责人", "验证负责人"):
            value = field(section, role)
            if value in ("", "待指定", "待定", "TBD"):
                people[role] = None
            elif re.fullmatch(r"@[A-Za-z0-9][A-Za-z0-9-]*", value):
                people[role] = value[1:]
            else:
                raise OperationError(role + "填写单个 @账号或待指定：" + key)
        title_text = re.sub(r"\s*[（(]#\d+[）)]\s*$", "", title)
        if number and title_text == title:
            title_text = None
        entries.append({"key": key, "parent": parent_key, "title": title, "title_text": title_text,
                        "number": number, "developer": people["开发负责人"],
                        "verifier": people["验证负责人"], "acceptance": field(section, "验收要求")})
    return {"version": version, "entries": entries, "change_note": field(body, "变更说明")}


def scope_diff(previous: dict, entries: list[dict]) -> dict:
    current = {entry["key"]: entry for entry in entries}
    changed = []
    for key in current.keys() & previous.keys():
        # Titles with inserted references normalize to the real Issue ID.
        before, after = previous[key], current[key]
        content_changed = any(before.get(name) != after.get(name)
                              for name in ("parent", "developer", "verifier", "acceptance"))
        if after.get("number") and before.get("number") != after["number"]:
            raise OperationError("条目标识已绑定其他 Issue：" + key)
        if not after.get("number") and before.get("title") != after["title"]:
            content_changed = True
        if after.get("title_text") and before.get("title_text") != after["title_text"]:
            content_changed = True
        if content_changed:
            changed.append(key)
    return {"added": sorted(current.keys() - previous.keys()),
            "removed": sorted(previous.keys() - current.keys()), "changed": sorted(changed)}


def patch_references(body: str, mappings: dict[str, int], repository: str) -> str:
    lines, key = body.splitlines(), None
    for index, line in enumerate(lines):
        match = KEY_RE.fullmatch(line.strip())
        if match:
            key = match[3]
        elif re.match(r"^#{1,2}\s", line):
            key = None
        elif key in mappings and re.match(r"^\s*需求\s*[:：]", line):
            if issue_reference(field(line, "需求"), repository) is None:
                lines[index] = line.rstrip() + f"（#{mappings[key]}）"
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def milestone_number(issue: dict) -> int | None:
    return issue["milestone"]["number"] if issue.get("milestone") else None


def manual_mode(record: dict, current: int | None, human_event: bool = False) -> dict:
    updated = dict(record)
    if human_event or ("last" in record and current != record["last"]):
        updated.update(mode="manual", last=current)
    return updated


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
        result = subprocess.run(args, input=None if data is None else json.dumps(data, ensure_ascii=False),
                                encoding="utf-8", text=True, capture_output=True, env=env, timeout=90)
        if result.returncode:
            raise OperationError(result.stderr.strip() or result.stdout.strip())
        parsed = json.loads(result.stdout) if result.stdout.strip() else None
        if isinstance(parsed, dict) and parsed.get("errors"):
            raise OperationError(json.dumps(parsed["errors"], ensure_ascii=False))
        return parsed

    def rest(self, path: str, method="GET", data=None):
        return self.call("repos/" + self.repository + "/" + path, method, data)

    def pages(self, path: str) -> list:
        items = []
        for page in range(1, 101):
            part = self.rest(path + ("&" if "?" in path else "?") + f"per_page=100&page={page}")
            if not isinstance(part, list):
                raise OperationError("预期分页数组：" + path)
            items.extend(part)
            if len(part) < 100:
                return items
        raise OperationError("分页超出处理上限")

    def graphql(self, query: str, **variables):
        return self.call("graphql", "POST", {"query": query, "variables": variables}, project=True)["data"]


class Controller:
    def __init__(self, api: GitHub, config: dict):
        self.api, self.config = api, config
        self.repository = config["repository"]
        self.run_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com") + "/" + self.repository + "/actions/runs/" + os.environ.get("GITHUB_RUN_ID", "local")

    def issue(self, number: int):
        issue = self.api.rest(f"issues/{number}")
        if "pull_request" in issue:
            raise OperationError(f"#{number} 是 PR，请引用 Issue")
        return issue

    def load(self, release: int):
        comments = self.api.pages(f"issues/{release}/comments")
        candidates = [c for c in comments if SUMMARY_MARKER in c["body"]
                      and c["user"]["login"] == "github-actions[bot]"]
        if len(candidates) > 1:
            raise OperationError("发现多条机器汇总，请先处理重复评论")
        comment = candidates[0] if candidates else None
        state = {"schema": 1, "release": release, "scope": {}, "objects": {}, "bugs": {}, "prs": {}, "revision": 0, "history": []}
        if comment and (match := STATE_RE.search(comment["body"])):
            state = json.loads(base64.b64decode(match[1]).decode("utf-8"))
        return comment, state

    def save(self, release: int, state: dict, result: str):
        rows = []
        for key, entry in state["scope"].items():
            parent = "子需求" if entry["parent"] else "需求"
            observed = state.get("observed", {}).get(str(entry["number"]), {})
            validation = state.get("verification", {}).get(str(entry["number"]), {}).get("result", "待验证")
            rows.append(f"| {parent} `{key}` | #{entry['number']} | @{entry['developer']} | @{entry['verifier']} | {observed.get('state', 'open')} / {validation} |")
        serialized = base64.b64encode(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode()).decode()
        body = SUMMARY_MARKER + "\n\n## 版本操作结果\n\n" + result + f"\n\n[Workflow]({self.run_url})\n\n"
        if state.get("milestone"):
            body += f"Milestone：[目标版本]({state['milestone']['html_url']})\n\n"
        if state.get("archive"):
            body += f"主 Bug Issue：#{state['archive']}\n\n"
        body += "| 条目 | Issue | 开发负责人 | 验证负责人 | 状态 / 验证 |\n| --- | --- | --- | --- | --- |\n" + "\n".join(rows)
        body = body.replace("@None", "待指定")
        body += "\n\n### 缺陷与修复 PR\n\n| 对象 | Milestone 编号 | 管理方式 |\n| --- | --- | --- |\n"
        for number, record in state.get("bugs", {}).items():
            body += f"| Bug #{number} | {record.get('last')} | {record['mode']} |\n"
        for number, record in state.get("prs", {}).items():
            body += f"| PR #{number} | {record.get('last')} | {record['mode']} |\n"
        body += "\n\n### 最近范围操作\n\n" + "\n".join(state.get("history", [])[-8:])
        body += f"\n\n<!-- ucm-release-state:v1 {serialized} -->"
        if len(body) > 60000:
            raise OperationError("状态记录达到评论容量，请分片存储后继续")
        comment, _ = self.load(release)
        if comment:
            self.api.rest(f"issues/comments/{comment['id']}", "PATCH", {"body": body})
        else:
            self.api.rest(f"issues/{release}/comments", "POST", {"body": body})

    def validate(self, parsed: dict, release: int, state: dict):
        if state.get("version") and state["version"] != parsed["version"]:
            raise OperationError("版本号已绑定，请创建另一版本入口")
        if not parsed["entries"] and not state.get("initialized"):
            raise OperationError("初始化至少填写一个需求")
        diff = scope_diff(state["scope"], parsed["entries"])
        if diff["removed"] and not parsed["change_note"]:
            raise OperationError("移出需求时填写‘变更说明：原因’")
        for entry in parsed["entries"]:
            for login in (entry["developer"], entry["verifier"]):
                if login:
                    self.api.call("users/" + login)
            existing_number = entry["number"] or state["objects"].get(entry["key"])
            if existing_number:
                existing = self.issue(existing_number)
                if existing["number"] == release or existing["number"] == state.get("archive"):
                    raise OperationError("管理容器请使用独立条目")
                target = milestone_number(existing)
                if target and existing["milestone"]["title"] != parsed["version"]:
                    raise OperationError(f"#{existing['number']} 已关联其他 Milestone，请维护者处理")
                desired = entry["developer"]
                assigned = {person["login"] for person in existing["assignees"]}
                previous_developer = state["scope"].get(entry["key"], {}).get("developer")
                if desired and assigned and desired not in assigned and assigned != {previous_developer}:
                    raise OperationError(f"#{existing['number']} 的负责人存在冲突")
                if entry["parent"]:
                    parent_entry = next(e for e in parsed["entries"] if e["key"] == entry["parent"])
                    desired_parent = parent_entry["number"] or state["objects"].get(parent_entry["key"])
                    previous_parent = state["scope"].get(entry["key"], {}).get("parent")
                    previous_number = state["scope"].get(previous_parent, {}).get("number")
                    current_parent = self.parent(existing["number"])
                    if current_parent and current_parent["number"] not in (desired_parent, previous_number):
                        raise OperationError(f"#{existing['number']} 已属于其他父需求 #{current_parent['number']}")
        return diff

    def create_or_recover(self, release: int, key: str, title: str, body: str, labels: list[str], state: dict):
        if key in state["objects"]:
            return self.issue(state["objects"][key])
        marker = ORIGIN.format(release=release, key=key)
        matches = [i for i in self.api.pages("issues?state=all") if marker in (i.get("body") or "") and "pull_request" not in i]
        if len(matches) > 1:
            raise OperationError("来源标记对应多个对象：" + key)
        issue = matches[0] if matches else self.api.rest("issues", "POST", {"title": title, "body": marker + "\n\n" + body, "labels": labels})
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
            raise OperationError(f"#{child['number']} 已属于父需求 #{current['number']}")
        self.api.rest(f"issues/{parent}/sub_issues", "POST", {"sub_issue_id": child["id"], "replace_parent": bool(current)})
        actual = self.parent(child["number"])
        if not actual or actual["number"] != parent:
            raise OperationError("父关系回读失败")

    def initialize_or_update(self, release: int, command: str, dry_run=False):
        source = self.issue(release)
        if "release" not in {label["name"] for label in source["labels"]}:
            raise OperationError("版本入口需要 release 标签")
        _, state = self.load(release)
        try:
            parsed = parse_release(source["body"], self.repository)
            if command == "update" and not state.get("initialized"):
                raise OperationError("先执行 /release_init")
            diff = self.validate(parsed, release, state)
            plan = json.dumps(diff, ensure_ascii=False)
            if dry_run:
                self.save(release, state, f"预览 `{command}`\n\n```json\n{plan}\n```")
                return
            if command == "release_init" and state.get("initialized"):
                self.save(release, state, "初始化已完成。范围变化通过 `/update` 同步。\n\n" + plan)
                return
            self.save(release, state, f"执行中：`/{command}`\n\n{plan}")
            state["version"] = parsed["version"]
            if not state.get("milestone"):
                matches = [m for m in self.api.pages("milestones?state=all") if m["title"] == parsed["version"]]
                state["milestone"] = matches[0] if matches else self.api.rest("milestones", "POST", {"title": parsed["version"], "description": f"Release tracking: #{release}"})
                self.save(release, state, "执行中：Milestone 已记录")
            target = state["milestone"]["number"]
            self.api.rest(f"issues/{release}", "PATCH", {"milestone": target})
            archive = self.create_or_recover(release, "__bugs__", f"[Bug Archive] UCM {parsed['version']} 测试问题",
                f"源版本：{parsed['version']}\nRelease：#{release}\n\n在此创建 Bug Sub-issues，默认目标版本为 {parsed['version']}。",
                ["release-bug-archive"], state)
            state["archive"] = archive["number"]
            self.attach(archive, release)
            self.api.rest(f"issues/{archive['number']}", "PATCH", {"milestone": target})
            desired = {}
            for entry in parsed["entries"]:
                key = entry["key"]
                item = self.issue(entry["number"]) if entry["number"] else self.create_or_recover(
                    release, key, entry["title"], f"Release：#{release}\n验收要求：{entry['acceptance']}\n验证负责人：@{entry['verifier']}".replace("@None", "待指定"),
                    ["requirement"], state)
                entry = dict(entry, number=item["number"])
                state["objects"][key] = item["number"]
                desired[key] = entry
                previous = state["scope"].get(key, {})
                if entry["parent"]:
                    old_parent = state["scope"].get(previous.get("parent"), {}).get("number")
                    self.attach(item, desired[entry["parent"]]["number"], old_parent)
                elif previous.get("parent"):
                    raise OperationError("子需求升级为顶层需求时，请先明确移除原父关系")
                payload = {"milestone": target}
                if key in diff["changed"] and entry.get("title_text"):
                    payload["title"] = entry["title_text"]
                if entry["developer"]:
                    payload["assignees"] = [entry["developer"]]
                self.api.rest(f"issues/{item['number']}", "PATCH", payload)
                if key in diff["changed"]:
                    self.api.rest(f"issues/{item['number']}/comments", "POST", {"body": f"版本范围更新，Release #{release}\n验收要求：{entry['acceptance']}\n验证负责人：{entry['verifier'] or '待指定'}"})
                self.save(release, state, f"执行中：已关联 `{key}` → #{item['number']}")
            for key in diff["removed"]:
                item = self.issue(state["scope"][key]["number"])
                if milestone_number(item) == target:
                    self.api.rest(f"issues/{item['number']}", "PATCH", {"milestone": None})
            latest = self.issue(release)
            latest_parsed = parse_release(latest["body"], self.repository)
            if latest_parsed != parsed:
                raise OperationError("执行期间正文已修改；已记录成功对象，请重新执行同步")
            patched = patch_references(latest["body"], state["objects"], self.repository)
            if patched != latest["body"]:
                self.api.rest(f"issues/{release}", "PATCH", {"body": patched})
            # Keep the actual created title as a display value while key/number identify scope.
            state["scope"] = desired
            state["initialized"] = True
            state["revision"] += 1
            state["history"].append(f"- r{state['revision']} `/{command}`：{plan}；{parsed['change_note']}")
            self.save(release, state, f"已完成：`/{command}`，范围修订 r{state['revision']}\n\n{plan}")
            self.reconcile(release)
        except Exception as error:
            self.save(release, state, "执行失败，已完成对象可重试恢复：\n\n" + str(error))
            raise

    def project_item(self, issue: dict, project_name: str, verifier=None, validation=None):
        project = self.config["projects"][project_name]
        item = self.api.graphql("""mutation($project:ID!,$content:ID!){addProjectV2ItemById(input:{projectId:$project,contentId:$content}){item{id}}}""",
                               project=project["id"], content=issue["node_id"])["addProjectV2ItemById"]["item"]
        for name, value in (("Validation owner", verifier), ("Validation status", validation)):
            if not value:
                continue
            spec = project["fields"][name]
            if name == "Validation owner":
                field_value = {"text": value}
            else:
                options = {o["name"]: o["id"] for o in spec["options"]}
                field_value = {"singleSelectOptionId": options[value]}
            self.api.graphql("""mutation($input:UpdateProjectV2ItemFieldValueInput!){updateProjectV2ItemFieldValue(input:$input){projectV2Item{id}}}""",
                             input={"projectId": project["id"], "itemId": item["id"], "fieldId": spec["id"], "value": field_value})

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
            source = field(bug.get("body") or "", "源版本主 Bug Issue")
            if source and issue_reference(source, self.repository) == state["archive"]:
                self.attach(bug, state["archive"])
        bugs = self.api.pages(f"issues/{state['archive']}/sub_issues")
        for bug in bugs:
            number = str(bug["number"])
            current = milestone_number(bug)
            record = state["bugs"].get(number)
            if record is None:
                record = {"mode": "auto" if current in (None, target) else "manual", "last": current}
            else:
                record = manual_mode(record, current, bool(human and milestone_event and event_issue.get("number") == bug["number"]))
            record = self.milestone_history(bug["number"], record)
            if record["mode"] == "auto" and current != target:
                self.api.rest(f"issues/{number}", "PATCH", {"milestone": target})
                record["last"] = target
                bug["milestone"] = state["milestone"]
            state["bugs"][number] = record
            state.setdefault("observed", {})[number] = {"state": bug["state"], "milestone": record["last"]}
            if "bug" not in {label["name"] for label in bug["labels"]}:
                self.api.rest(f"issues/{number}/labels", "POST", {"labels": ["bug"]})
            self.project_item(bug, "Bugfix")
        self.save(release, state, "同步中：Bug 默认版本与人工指定已记录")
        bug_numbers = {bug["number"] for bug in bugs}
        prs = self.api.pages("pulls?state=all")
        for pr in prs:
            relation = field(pr.get("body") or "", "关联缺陷")
            refs = []
            try:
                for match in REF_RE.finditer(relation):
                    refs.append(issue_reference(match[0], self.repository))
            except OperationError as error:
                problems.append(f"PR #{pr['number']}：{error}")
                continue
            if not (set(refs) & bug_numbers) and str(pr["number"]) not in state["prs"]:
                continue
            linked = [self.issue(n) for n in refs]
            if any("bug" not in {label["name"] for label in bug["labels"]} for bug in linked):
                raise OperationError("修复 PR 的关联缺陷需要 bug 标签")
            targets = {milestone_number(bug) for bug in linked}
            number = str(pr["number"])
            current = milestone_number(pr)
            record = state["prs"].get(number, {"mode": "auto" if current is None else "manual", "last": current})
            record = manual_mode(record, current, bool(human and milestone_event and event_pr.get("number") == pr["number"]))
            record = self.milestone_history(pr["number"], record)
            record["bugs"] = refs
            if len(targets) > 1:
                problems.append(f"PR #{number} 关联 Bug 的目标版本冲突，请指定 PR Milestone")
            elif record["mode"] == "auto":
                desired = next(iter(targets), None)
                if current != desired:
                    self.api.rest(f"issues/{number}", "PATCH", {"milestone": desired})
                record["last"] = desired
            elif targets and current not in targets:
                problems.append(f"PR #{number} 使用人工目标，与关联 Bug 的目标版本不同")
            state["prs"][number] = record
            self.project_item(pr, "Bugfix")
        for entry in state["scope"].values():
            issue = self.issue(entry["number"])
            state.setdefault("observed", {})[str(issue["number"])] = {"state": issue["state"], "milestone": milestone_number(issue)}
        self.save(release, state, "同步完成：Bug 默认版本、人工目标与修复 PR 已对账。\n\n" + "\n".join(problems))

    def releases(self):
        return [i["number"] for i in self.api.pages("issues?state=all&labels=release") if "pull_request" not in i]

    def milestone_history(self, number: int, record: dict):
        changes = [event for event in self.api.pages(f"issues/{number}/events")
                   if event["event"] in ("milestoned", "demilestoned")]
        if "event_id" in record:
            later = [event for event in changes if event["id"] > record["event_id"]]
            if any(event.get("actor", {}).get("type") == "User" for event in later):
                record["mode"] = "manual"
        record["event_id"] = max((event["id"] for event in changes), default=0)
        return record

    def verify(self, release: int, number: int, passed: bool, evidence: str, actor: str):
        if not re.fullmatch(r"https://\S+", evidence):
            raise OperationError("验证记录需要 HTTPS 证据链接")
        _, state = self.load(release)
        active = {entry["number"] for entry in state["scope"].values()} | {int(n) for n in state["bugs"]}
        if number not in active:
            raise OperationError("验证对象属于当前版本的需求或源归档 Bug")
        item = self.issue(number)
        result = "Passed" if passed else "Failed"
        body = f"<!-- ucm-verification:v1 -->\n验证对象：#{number}\n目标 Milestone：{milestone_number(item)}\n结果：{result}\n证据：{evidence}\n验证者：@{actor}"
        self.api.rest(f"issues/{number}/comments", "POST", {"body": body})
        status = "status:verified" if "bug" in {x["name"] for x in item["labels"]} else "status:accepted"
        labels = [x["name"] for x in item["labels"] if not x["name"].startswith("status:")]
        labels.append(status if passed else "status:fixing" if "bug" in labels else "status:developing")
        self.api.rest(f"issues/{number}", "PATCH", {"state": "closed" if passed else "open", "labels": labels})
        state.setdefault("verification", {})[str(number)] = {"result": result, "evidence": evidence, "actor": actor, "milestone": milestone_number(item)}
        self.save(release, state, f"已记录 #{number} 的验证：{result}")
        project_name = "Bugfix" if "bug" in labels else "Connector v2" if "Connector" in item["title"] else "模型适配"
        self.project_item(item, project_name, actor, result)

    def readiness(self, release: int):
        _, state = self.load(release)
        if not state.get("initialized"):
            raise OperationError("版本尚未初始化")
        source = self.issue(release)
        current_plan = parse_release(source["body"], self.repository)
        delta = scope_diff(state["scope"], current_plan["entries"])
        if current_plan["version"] != state["version"] or any(delta.values()):
            return state, {"status": "blocked", "reason": "正文范围待 /update 同步", "diff": delta}
        target = state["milestone"]["number"]
        items = [self.issue(e["number"]) for e in state["scope"].values()]
        items += [i for i in self.api.pages(f"issues?state=all&milestone={target}&labels=bug") if "pull_request" not in i]
        pending = []
        for item in {i["number"]: i for i in items}.values():
            record = state.get("verification", {}).get(str(item["number"]), {})
            if item["state"] != "closed" or record.get("result") != "Passed" or record.get("milestone") != target:
                pending.append(item["number"])
        if pending:
            return state, {"status": "blocked", "pending": pending}
        return state, {"status": "ready", "pending": []}

    def publish(self, release: int):
        self.reconcile(release)
        state, result = self.readiness(release)
        if result["status"] != "ready":
            self.save(release, state, "发布准入：" + json.dumps(result, ensure_ascii=False))
            raise OperationError("完成需求与目标 Bug 的验证后发布")
        repo = self.api.rest("")
        commit = state.get("published", {}).get("commit") or self.api.rest("commits/" + repo["default_branch"])["sha"]
        tag = "v" + state["version"]
        existing = [r for r in self.api.pages("releases") if r["tag_name"] == tag]
        marker = f"<!-- ucm-release:{self.repository}:{release} -->"
        notes = marker + f"\nRelease tracking: #{release}\n\n" + "\n".join(f"- #{e['number']} {e['title']}" for e in state["scope"].values())
        notes += "\n\n" + "\n".join(f"- 验证 #{n}：{v['evidence']}" for n, v in state.get("verification", {}).items())
        if existing and marker not in (existing[0].get("body") or ""):
            raise OperationError("该 Tag 已有其他发布入口，请维护者处理")
        try:
            tagged = self.api.rest("git/ref/tags/" + tag)["object"]
        except OperationError as error:
            if "404" not in str(error):
                raise
            tagged = None
        if tagged and (tagged["type"] != "commit" or tagged["sha"] != commit):
            raise OperationError("已有 Tag 的目标与本次发布代码不同")
        published = existing[0] if existing else self.api.rest("releases", "POST", {"tag_name": tag, "target_commitish": commit, "name": "UCM " + state["version"], "body": notes})
        receipt = self.api.rest(f"releases/{published['id']}")
        if receipt["draft"] or receipt["tag_name"] != tag:
            raise OperationError("Release 回读失败")
        actual_tag = self.api.rest("git/ref/tags/" + tag)["object"]
        if actual_tag["type"] != "commit" or actual_tag["sha"] != commit:
            raise OperationError("发布 Tag 代码回读失败")
        state["published"] = {"url": receipt["html_url"], "tag": tag, "commit": commit}
        self.save(release, state, "已发布：[GitHub Release](" + receipt["html_url"] + ")")
        self.api.rest(f"milestones/{state['milestone']['number']}", "PATCH", {"state": "closed"})
        self.api.rest(f"issues/{release}", "PATCH", {"state": "closed"})

    def authorize(self, actor: str):
        permission = self.api.rest(f"collaborators/{actor}/permission")["permission"]
        if permission not in ("admin", "maintain", "write"):
            raise OperationError("此操作需要仓库 write、maintain 或 admin 权限")

    def handle(self, event: dict, event_name: str):
        if event_name == "issue_comment":
            if event.get("issue", {}).get("pull_request"):
                return
            command = event["comment"]["body"].strip()
            if not command.startswith(("/release_init", "/update", "/verify", "/release_ready", "/release_publish")):
                return
            actor = event["sender"]["login"]
            self.authorize(actor)
            release = event["issue"]["number"]
            _, before = self.load(release)
            comment_id = str(event["comment"]["id"])
            if comment_id in before.get("processed_comments", []):
                return
            if match := re.fullmatch(r"/(release_init|update)( --dry-run)?", command):
                self.initialize_or_update(release, match[1], bool(match[2]))
            elif match := re.fullmatch(r"/verify #(\d+) (passed|failed) (https://\S+)", command):
                self.verify(release, int(match[1]), match[2] == "passed", match[3], actor)
            elif command == "/release_ready":
                self.reconcile(release)
                state, result = self.readiness(release)
                self.save(release, state, "发布准入：" + json.dumps(result, ensure_ascii=False))
            elif command == "/release_publish":
                self.publish(release)
            else:
                raise OperationError("命令格式错误")
            _, latest = self.load(release)
            latest.setdefault("processed_comments", []).append(comment_id)
            latest["processed_comments"] = latest["processed_comments"][-500:]
            current_comment, _ = self.load(release)
            # Preserve the actual result while checkpointing processed command IDs.
            result = current_comment["body"].split("## 版本操作结果\n\n", 1)[1].split("\n\n[Workflow]", 1)[0]
            self.save(release, latest, result)
        else:
            for release in self.releases():
                if event_name in ("schedule", "workflow_dispatch"):
                    _, state = self.load(release)
                    for comment in self.api.pages(f"issues/{release}/comments"):
                        body = comment["body"].strip()
                        if (comment["user"]["type"] == "User" and str(comment["id"]) not in state.get("processed_comments", [])
                                and body.startswith(("/release_init", "/update", "/verify", "/release_ready", "/release_publish"))):
                            try:
                                self.handle({"issue": {"number": release}, "comment": comment, "sender": comment["user"]}, "issue_comment")
                            except OperationError as error:
                                print(f"Command #{comment['id']} requires attention: {error}")
                self.reconcile(release, event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--event", type=Path, default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "schedule"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    repository = os.environ.get("GITHUB_REPOSITORY", config["repository"])
    if repository != config["repository"]:
        raise OperationError("部署仓库与 config.json 的 repository 不一致")
    event = json.loads(args.event.read_text(encoding="utf-8")) if args.event else {}
    Controller(GitHub(repository), config).handle(event, args.event_name)
    print(json.dumps({"status": "completed", "event": args.event_name, "repository": repository}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
