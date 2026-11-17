import copy
import unittest

from controller import (
    Controller,
    OperationError,
    issue_reference,
    manual_mode,
    parse_release,
    patch_references,
    scope_diff,
)

REPOSITORY = "harrisonyhq/unified-cache-management"
PLAN = """## 版本信息
版本号：0.11.0
### 需求 connector
需求：Connector v2 首批上库
开发负责人：@harrisonyhq
验证负责人：待指定
验收要求：验收整体调用路径
#### 子需求 contract
需求：Connector v2 接口契约
开发负责人：待指定
验证负责人：@harrisonyhq
验收要求：通过契约测试
## 变更记录
变更说明：初始范围
"""


class ParserContracts(unittest.TestCase):
    def test_parent_and_children_are_unambiguous(self):
        parsed = parse_release(PLAN, REPOSITORY)
        self.assertEqual(parsed["version"], "0.11.0")
        self.assertIsNone(parsed["entries"][0]["parent"])
        self.assertEqual(parsed["entries"][1]["parent"], "connector")
        self.assertEqual(parsed["entries"][1]["verifier"], "harrisonyhq")
        self.assertEqual(parsed["entries"][1]["acceptance"], "通过契约测试")

    def test_reference_insertion_round_trips_and_preserves_text(self):
        patched = patch_references(PLAN, {"connector": 12, "contract": 13}, REPOSITORY)
        self.assertIn("需求：Connector v2 首批上库（#12）", patched)
        self.assertIn("需求：Connector v2 接口契约（#13）", patched)
        self.assertIn("验收要求：通过契约测试", patched)
        self.assertEqual(
            patch_references(patched, {"connector": 12, "contract": 13}, REPOSITORY),
            patched,
        )
        self.assertEqual(
            [e["number"] for e in parse_release(patched, REPOSITORY)["entries"]],
            [12, 13],
        )

    def test_existing_issue_reference_formats(self):
        for value in [
            "#12",
            REPOSITORY + "#12",
            "https://github.com/" + REPOSITORY + "/issues/12",
            "[已有需求](https://github.com/" + REPOSITORY + "/issues/12)",
            "需求名称（#12）",
        ]:
            with self.subTest(value=value):
                self.assertEqual(issue_reference(value, REPOSITORY), 12)

    def test_invalid_references_never_become_new_titles(self):
        for value in ["#abc", "https://github.com/a/b/pull/12", "a/b#12", "#12 #13"]:
            with self.subTest(value=value), self.assertRaises(OperationError):
                issue_reference(value, REPOSITORY)

    def test_duplicate_keys_and_orphan_children_are_rejected(self):
        with self.assertRaises(OperationError):
            parse_release(
                PLAN.replace("子需求 contract", "子需求 connector"), REPOSITORY
            )
        with self.assertRaises(OperationError):
            parse_release(
                PLAN.replace("### 需求 connector", "#### 子需求 connector"), REPOSITORY
            )

    def test_scope_changes_do_not_confuse_reference_patches_with_edits(self):
        original = parse_release(PLAN, REPOSITORY)["entries"]
        previous = {e["key"]: dict(e, number=n) for e, n in zip(original, [12, 13])}
        patched = parse_release(
            patch_references(PLAN, {"connector": 12, "contract": 13}, REPOSITORY),
            REPOSITORY,
        )
        self.assertEqual(
            scope_diff(previous, patched["entries"]),
            {"added": [], "removed": [], "changed": []},
        )
        shortened = patched["entries"][:1]
        shortened[0]["acceptance"] = "新的验收条件"
        self.assertEqual(
            scope_diff(previous, shortened),
            {"added": [], "removed": ["contract"], "changed": ["connector"]},
        )

    def test_reusing_a_key_for_another_issue_is_rejected(self):
        entries = parse_release(PLAN, REPOSITORY)["entries"]
        previous = {entries[0]["key"]: dict(entries[0], number=12)}
        with self.assertRaises(OperationError):
            scope_diff(previous, [dict(entries[0], number=99)])


class MilestoneOwnership(unittest.TestCase):
    def test_manual_clear_is_persistent(self):
        record = manual_mode({"mode": "auto", "last": 1}, None)
        self.assertEqual(record, {"mode": "manual", "last": None})
        self.assertEqual(manual_mode(record, None), record)

    def test_human_event_takes_ownership_even_when_value_matches(self):
        self.assertEqual(
            manual_mode({"mode": "auto", "last": 1}, 1, True)["mode"], "manual"
        )

    def test_automatic_readback_preserves_automatic_mode(self):
        self.assertEqual(manual_mode({"mode": "auto", "last": 1}, 1)["mode"], "auto")


class MemoryGitHub:
    """Repository fixture with real Issue IDs and native parent relationships."""

    def __init__(self):
        self.items = {1: self.make_issue(1, "[Release] UCM 0.11.0", PLAN, ["release"])}
        self.comments = []
        self.milestones = []
        self.parents = {}
        self.lose_response_for = None

    @staticmethod
    def make_issue(number, title, body, labels):
        return {
            "number": number,
            "id": number + 100,
            "title": title,
            "body": body,
            "labels": [{"name": x} for x in labels],
            "milestone": None,
            "assignees": [],
            "state": "open",
            "node_id": f"I_{number}",
        }

    def call(self, path):
        return {"login": path.split("/")[-1]}

    def pages(self, path):
        if path.startswith("milestones"):
            return copy.deepcopy(self.milestones)
        if path.endswith("/comments"):
            number = int(path.split("/")[1])
            return copy.deepcopy([c for c in self.comments if c["issue"] == number])
        if path == "issues?state=all&labels=bug":
            return copy.deepcopy(
                [i for i in self.items.values() if {"name": "bug"} in i["labels"]]
            )
        if path.endswith("/sub_issues"):
            number = int(path.split("/")[1])
            return copy.deepcopy(
                [
                    self.items[n]
                    for n, parent in self.parents.items()
                    if parent == number
                ]
            )
        if path == "pulls?state=all":
            return []
        if path.startswith("issues?"):
            return copy.deepcopy(list(self.items.values()))
        raise AssertionError(path)

    def rest(self, path, method="GET", data=None):
        parts = path.split("/")
        if path == "milestones" and method == "POST":
            item = dict(
                data,
                number=len(self.milestones) + 1,
                html_url="https://example.com/milestone",
            )
            self.milestones.append(item)
            return copy.deepcopy(item)
        if path == "issues" and method == "POST":
            number = max(self.items) + 1
            item = self.make_issue(number, data["title"], data["body"], data["labels"])
            self.items[number] = item
            if data["title"] == self.lose_response_for:
                self.lose_response_for = None
                raise OperationError(
                    "transport interrupted after server accepted issue"
                )
            return copy.deepcopy(item)
        if parts[:2] == ["issues", "comments"] and method == "PATCH":
            comment = next(c for c in self.comments if c["id"] == int(parts[2]))
            comment.update(data)
            return copy.deepcopy(comment)
        if parts[0] == "issues" and len(parts) == 2:
            item = self.items[int(parts[1])]
            if method == "PATCH":
                for key, value in data.items():
                    if key == "milestone":
                        item[key] = next(
                            (m for m in self.milestones if m["number"] == value), None
                        )
                    elif key == "assignees":
                        item[key] = [{"login": x} for x in value]
                    else:
                        item[key] = value
            return copy.deepcopy(item)
        if parts[0] == "issues" and parts[-1] == "comments" and method == "POST":
            comment = dict(
                data,
                issue=int(parts[1]),
                id=len(self.comments) + 1,
                user={"login": "github-actions[bot]", "type": "Bot"},
            )
            self.comments.append(comment)
            return copy.deepcopy(comment)
        if parts[-1] == "parent":
            parent = self.parents.get(int(parts[1]))
            if parent is None:
                raise OperationError("HTTP 404")
            return copy.deepcopy(self.items[parent])
        if parts[-1] == "sub_issues" and method == "POST":
            child = next(
                i for i in self.items.values() if i["id"] == data["sub_issue_id"]
            )
            self.parents[child["number"]] = int(parts[1])
            return copy.deepcopy(child)
        raise AssertionError((path, method, data))


class RepositoryOperations(unittest.TestCase):
    def setUp(self):
        self.api = MemoryGitHub()
        self.controller = Controller(self.api, {"repository": REPOSITORY})
        self.controller.reconcile = lambda *args, **kwargs: None

    def test_preview_then_apply_reuses_the_same_result_comment(self):
        self.controller.initialize_or_update(1, "release_init", dry_run=True)
        comment_id = self.api.comments[0]["id"]
        self.assertEqual(len(self.api.items), 1)
        self.assertFalse(self.api.milestones)
        self.assertEqual(self.api.items[1]["body"], PLAN)
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(len(self.api.comments), 1)
        self.assertEqual(self.api.comments[0]["id"], comment_id)
        self.assertNotIn("预览", self.api.comments[0]["body"])
        _, state = self.controller.load(1)
        self.assertEqual(
            self.api.parents[state["scope"]["contract"]["number"]],
            state["scope"]["connector"]["number"],
        )
        self.assertEqual(self.api.parents[state["archive"]], 1)
        self.assertEqual(len(self.api.items), 4)

    def test_retry_recovers_issue_created_before_lost_response(self):
        self.api.lose_response_for = "Connector v2 接口契约"
        with self.assertRaises(OperationError):
            self.controller.initialize_or_update(1, "release_init")
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(len(self.api.items), 4)
        self.assertEqual(len(self.api.milestones), 1)
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(len(self.api.items), 4)

    def test_update_removes_delivery_scope_and_keeps_parent_and_issue(self):
        self.controller.initialize_or_update(1, "release_init")
        _, before = self.controller.load(1)
        parent = before["scope"]["connector"]["number"]
        child = before["scope"]["contract"]["number"]
        self.api.items[1]["body"] = (
            self.api.items[1]["body"].split("#### 子需求")[0]
            + "\n## 变更记录\n变更说明：子需求延期\n"
        )
        self.controller.initialize_or_update(1, "update")
        _, after = self.controller.load(1)
        self.assertEqual(set(after["scope"]), {"connector"})
        self.assertIsNone(self.api.items[child]["milestone"])
        self.assertEqual(self.api.parents[child], parent)
        self.assertEqual(self.api.items[child]["state"], "open")
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(len(self.api.items), 4)

    def test_update_retains_manually_retargeted_removed_item(self):
        self.controller.initialize_or_update(1, "release_init")
        _, before = self.controller.load(1)
        child = before["scope"]["contract"]["number"]
        self.api.items[child]["milestone"] = {"number": 2, "title": "0.12.0"}
        self.api.items[1]["body"] = (
            self.api.items[1]["body"].split("#### 子需求")[0]
            + "\n变更说明：延期至 0.12.0\n"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.items[child]["milestone"]["number"], 2)

    def test_later_command_recovers_cancelled_earlier_command_in_order(self):
        self.controller.authorize = lambda actor: None
        user = {"login": "maintainer", "type": "User"}
        preview = {
            "id": 10,
            "issue": 1,
            "user": user,
            "body": "/release_init --dry-run",
        }
        apply = {"id": 11, "issue": 1, "user": user, "body": "/release_init"}
        self.api.comments.extend([preview, apply])
        self.controller.handle(
            {"issue": {"number": 1}, "comment": apply, "sender": user}, "issue_comment"
        )
        _, state = self.controller.load(1)
        self.assertTrue(state["initialized"])
        self.assertEqual(state["processed_comments"], ["10", "11"])
        self.assertEqual(state["command_cursor"], 11)
        self.controller.drain_commands(1)
        self.assertEqual(len(self.api.items), 4)

    def test_unassigned_bug_template_does_not_block_version_reconciliation(self):
        self.controller.initialize_or_update(1, "release_init")
        self.api.items[99] = self.api.make_issue(
            99, "Unassigned Bug", "源版本主 Bug Issue：#", ["bug"]
        )
        Controller.reconcile(self.controller, 1)
        comment, state = self.controller.load(1)
        self.assertIn("待归档", comment["body"])
        self.assertTrue(state["initialized"])


if __name__ == "__main__":
    unittest.main()
