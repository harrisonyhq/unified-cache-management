import copy
import unittest
from pathlib import Path

from controller import Controller, GitHub, bug_progress, manual_mode
from syntax import (
    OperationError,
    bind_entries,
    field,
    issue_reference,
    parse_release,
    patch_pull_requests,
    patch_references,
    references,
    scope_diff,
)

REPOSITORY = "harrisonyhq/unified-cache-management"
PLAN = """### Release Information

version: 0.12.0
branch: main
goal: Deliver the connector.
release_manager: @harrisonyhq
test_lead:
release_date: TBD

### Requirements

- Connector v2
  initial integration []

  description:
    Implement the connector.

    Acceptance criteria:
    - Pass the integration test.
    - Document the interface.

  assignee: @harrisonyhq

  pull_requests:

- Store interface []

  description:
    Implement the first interface.

  assignee:

  pull_requests:

### Change Log

reason: Initial scope.
"""


class ParserContracts(unittest.TestCase):
    def test_multiline_title_description_and_empty_fields(self):
        parsed = parse_release(PLAN, REPOSITORY)
        self.assertEqual(
            parsed["entries"][0]["title"], "Connector v2 initial integration"
        )
        self.assertIn(
            "\n\nAcceptance criteria:\n- Pass", parsed["entries"][0]["description"]
        )
        self.assertIsNone(parsed["entries"][1]["developer"])
        self.assertEqual(field("assignee:\nnext: @someone", "assignee"), "")
        self.assertEqual(parsed["metadata"]["test_lead"], "")

    def test_actual_release_checklist_is_english_and_parses(self):
        root = Path(__file__).parent
        example = (root / "examples/release-checklist.md").read_text()
        template = (
            root.parent / "ISSUE_TEMPLATE/900-release-checklist.yml"
        ).read_text()
        self.assertTrue(example.isascii())
        self.assertTrue(template.isascii())
        self.assertEqual(len(parse_release(example, REPOSITORY)["entries"]), 2)
        self.assertNotIn("### Pull Requests", example)
        self.assertFalse((root.parent / "ISSUE_TEMPLATE/010-release-plan.yml").exists())

    def test_only_requirements_section_is_parsed(self):
        value = PLAN.replace(
            "### Change Log", "### Functional Tests\n\n- [ ] A test\n\n### Change Log"
        )
        self.assertEqual(len(parse_release(value, REPOSITORY)["entries"]), 2)

    def test_reference_writeback_is_idempotent_and_preserves_description(self):
        entries = parse_release(PLAN, REPOSITORY)["entries"]
        mappings = {e["key"]: n for e, n in zip(entries, [14, 13])}
        patched = patch_references(PLAN, mappings, REPOSITORY)
        self.assertIn("/issues/14]", patched)
        self.assertEqual(patch_references(patched, mappings, REPOSITORY), patched)
        actual = parse_release(patched, REPOSITORY)["entries"]
        self.assertEqual([e["key"] for e in actual], ["issue-14", "issue-13"])
        self.assertEqual(actual[0]["description"], entries[0]["description"])
        previous = {e["key"]: e for e in bind_entries(entries, mappings)}
        self.assertFalse(any(scope_diff(previous, actual).values()))

    def test_existing_references_and_issue15_regression(self):
        body = PLAN.replace(
            "Connector v2\n  initial integration []",
            "Support GLM5.5 [https://github.com/" + REPOSITORY + "/issues/14]",
        )
        body = body.replace("Store interface []", "SGLang EPD [#13]")
        self.assertEqual(
            [e["number"] for e in parse_release(body, REPOSITORY)["entries"]], [14, 13]
        )

    def test_invalid_input_reports_line_before_creation(self):
        for bad in [
            PLAN.replace("  description:", " description:", 1),
            PLAN.replace("    Implement the connector.", "  Implement the connector."),
            PLAN.replace("  assignee: @harrisonyhq", "  assignee: @a @b"),
            PLAN.replace(
                "  assignee: @harrisonyhq",
                "  assignee: @harrisonyhq\n  assignee: @other",
            ),
            PLAN.replace("integration []", "integration"),
            PLAN.replace("Store interface []", "Store interface [#abc]"),
            PLAN.replace(
                "Store interface []",
                "Store interface [https://github.com/a/b/issues/1]",
            ),
            PLAN.replace(
                "Store interface []",
                "Store interface [https://github.com/" + REPOSITORY + "/pull/1]",
            ),
            PLAN.replace("Store interface []", "Connector v2 initial integration []"),
        ]:
            with self.subTest(body=bad), self.assertRaises(OperationError):
                parse_release(bad, REPOSITORY)

    def test_mention_collection_filters_cross_repository_and_pr_links(self):
        text = (
            "#14, fixes #8; "
            + REPOSITORY
            + "#16 https://github.com/"
            + REPOSITORY
            + "/issues/17"
        )
        text += (
            " other/repo#22 [#99](https://github.com/"
            + REPOSITORY
            + "/pull/99) https://docs.test/topic#55"
        )
        self.assertEqual(references(text, REPOSITORY), {8, 14, 16, 17})

    def test_pr_display_does_not_change_scope(self):
        body = patch_references(
            PLAN, {parse_release(PLAN, REPOSITORY)["entries"][0]["key"]: 14}, REPOSITORY
        )
        before = parse_release(body, REPOSITORY)["entries"]
        updated = patch_pull_requests(
            body, {14: [{"number": 33, "state": "open", "merged_at": None}]}, REPOSITORY
        )
        self.assertIn("| open", updated)
        self.assertEqual(before, parse_release(updated, REPOSITORY)["entries"])
        self.assertEqual(
            patch_pull_requests(
                updated,
                {14: [{"number": 33, "state": "open", "merged_at": None}]},
                REPOSITORY,
            ),
            updated,
        )

    def test_repository_root_endpoint(self):
        api = GitHub(REPOSITORY)
        api.call = lambda path, method, data: path
        self.assertEqual(api.rest(""), "repos/" + REPOSITORY)

    def test_release_date_validation(self):
        self.assertEqual(
            parse_release(
                PLAN.replace("release_date: TBD", "release_date: 2026-12-01"),
                REPOSITORY,
            )["metadata"]["release_date"],
            "2026-12-01",
        )
        with self.assertRaises(OperationError):
            parse_release(
                PLAN.replace("release_date: TBD", "release_date: tomorrow"), REPOSITORY
            )

    def test_pr_template_has_no_placeholder_issue_mentions(self):
        body = (Path(__file__).parent.parent / "PULL_REQUEST_TEMPLATE.md").read_text()
        self.assertFalse(references(body, REPOSITORY))


class MemoryGitHub:
    def __init__(self):
        self.items = {1: self.make_issue(1, "[Release] UCM 0.12.0", PLAN, ["release"])}
        self.comments, self.milestones, self.parents = [], [], {}
        self.prs, self.reviews, self.inline, self.events = {}, {}, {}, {}
        self.projects = {}
        self.lose_response_for = None
        self.lose_milestone_response = False
        self.lose_project_response = False

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
            "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
        }

    def call(self, path):
        return {"login": path.split("/")[-1]}

    def pages(self, path):
        if path.startswith("milestones"):
            return copy.deepcopy(self.milestones)
        if path.endswith("/comments") and path.startswith("issues/"):
            number = int(path.split("/")[1])
            return copy.deepcopy([c for c in self.comments if c["issue"] == number])
        if path.endswith("/events"):
            return copy.deepcopy(self.events.get(int(path.split("/")[1]), []))
        if path.startswith("pulls/"):
            _, number, kind = path.split("/")
            return copy.deepcopy(
                (self.reviews if kind == "reviews" else self.inline).get(
                    int(number), []
                )
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
            return copy.deepcopy(list(self.prs.values()))
        if path.startswith("issues?"):
            values = [i for i in self.items.values() if "pull_request" not in i]
            if "labels=" in path:
                label = path.split("labels=")[1].split("&")[0]
                values = [i for i in values if {"name": label} in i["labels"]]
            if "milestone=" in path:
                target = int(path.split("milestone=")[1].split("&")[0])
                values = [
                    i for i in values if (i["milestone"] or {}).get("number") == target
                ]
            return copy.deepcopy(values)
        raise AssertionError(path)

    def rest(self, path, method="GET", data=None):
        if path == "":
            return {"default_branch": "main"}
        if path.startswith("commits/"):
            return {"sha": "a" * 40}
        if path.startswith("collaborators/"):
            return {"permission": "write"}
        parts = path.split("/")
        if parts[0] == "milestones" and len(parts) == 2 and method == "PATCH":
            item = next(m for m in self.milestones if m["number"] == int(parts[1]))
            item.update(data)
            return copy.deepcopy(item)
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
            if int(parts[1]) not in self.items:
                raise OperationError("HTTP 404")
            item = self.items[int(parts[1])]
            if method == "PATCH":
                for key, value in data.items():
                    if key == "milestone":
                        item[key] = next(
                            (m for m in self.milestones if m["number"] == value), None
                        )
                        if int(parts[1]) in self.prs:
                            self.prs[int(parts[1])][key] = item[key]
                            if self.lose_milestone_response:
                                self.lose_milestone_response = False
                                raise OperationError(
                                    "Milestone response lost after acceptance"
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
                id=max([c["id"] for c in self.comments] + [0]) + 1,
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
        if parts[-1] == "labels" and method == "POST":
            self.items[int(parts[1])]["labels"] += [
                {"name": label} for label in data["labels"]
            ]
            return []
        raise AssertionError((path, method, data))

    def graphql(self, query, **variables):
        if query.startswith("query"):
            values = self.projects.get(variables["id"], {})
            return {
                "node": {
                    "projectItems": {
                        "nodes": [
                            {"id": item, "project": {"id": p}}
                            for p, item in values.items()
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        if "addProjectV2ItemById" in query:
            values = self.projects.setdefault(variables["content"], {})
            item = values.setdefault(
                variables["project"], variables["content"] + "_" + variables["project"]
            )
            if self.lose_project_response:
                self.lose_project_response = False
                raise OperationError("Project response lost after acceptance")
            return {"addProjectV2ItemById": {"item": {"id": item}}}
        if "deleteProjectV2Item" in query:
            for values in self.projects.values():
                if values.get(variables["project"]) == variables["item"]:
                    del values[variables["project"]]
            return {"deleteProjectV2Item": {"deletedItemId": variables["item"]}}
        if "updateProjectV2ItemFieldValue" in query:
            return {}
        raise AssertionError(query)

    def add_pr(self, number, body):
        item = self.make_issue(number, "Implementation", body, [])
        item["pull_request"] = {}
        self.items[number] = item
        pr = dict(
            item,
            html_url=f"https://github.com/{REPOSITORY}/pull/{number}",
            merged_at=None,
            draft=False,
        )
        self.prs[number] = pr
        return pr

    def comment(self, number, body):
        item = {
            "issue": number,
            "id": max([c["id"] for c in self.comments] + [0]) + 1,
            "body": body,
            "user": {"login": "developer", "type": "User"},
            "html_url": "https://example.com/comment",
        }
        self.comments.append(item)
        return item


class RepositoryOperations(unittest.TestCase):
    def setUp(self):
        self.api = MemoryGitHub()
        self.controller = Controller(
            self.api,
            {
                "repository": REPOSITORY,
                "projects": {
                    "Bugfix": {
                        "id": "P_BUG",
                        "fields": {
                            "Status": {
                                "id": "STATUS",
                                "options": [
                                    {"name": n, "id": n}
                                    for n in [
                                        "Ready",
                                        "In progress",
                                        "In review",
                                        "Done",
                                    ]
                                ],
                            }
                        },
                    }
                },
            },
        )

    def initialize(self):
        self.controller.initialize_or_update(1, "release_init")
        return self.controller.load(1)[1]

    def test_preview_then_apply_replaces_result_and_creates_flat_requirements(self):
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertEqual(len(self.api.items), 1)
        self.assertFalse(self.api.milestones)
        cid = self.api.comments[0]["id"]
        state = self.initialize()
        self.assertEqual(len(self.api.comments), 1)
        self.assertEqual(self.api.comments[0]["id"], cid)
        self.assertNotIn("预览", self.api.comments[0]["body"])
        self.assertEqual(self.api.parents, {state["archive"]: 1})
        self.assertEqual(set(state["scope"]), {"issue-3", "issue-4"})
        self.assertIn("Acceptance criteria:", self.api.items[3]["body"])
        self.assertEqual(len(self.api.items), 4)
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

    def test_recovery_after_lost_create_response(self):
        self.api.lose_response_for = "Store interface"
        with self.assertRaises(OperationError):
            self.initialize()
        self.initialize()
        self.assertEqual(len(self.api.items), 4)
        self.assertEqual(len(self.api.milestones), 1)

    def test_remove_retains_issue_and_manual_milestone(self):
        self.initialize()
        self.api.items[4]["milestone"] = {"number": 22, "title": "0.13.0"}
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")]
            + "### Change Log\n\nreason: Deferred.\n"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(set(self.controller.load(1)[1]["scope"]), {"issue-3"})
        self.assertEqual(self.api.items[4]["milestone"]["number"], 22)

    def test_remove_clears_owned_milestone_and_retains_native_parent(self):
        self.initialize()
        self.api.parents[4] = 3
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")]
            + "### Change Log\n\nreason: Deferred.\n"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertIsNone(self.api.items[4]["milestone"])
        self.assertEqual(self.api.parents[4], 3)
        self.assertEqual(self.api.items[4]["state"], "open")

    def test_invalid_plan_creates_no_repository_objects(self):
        self.api.items[1]["body"] = PLAN.replace("  description:", " description:", 1)
        with self.assertRaises(OperationError):
            self.initialize()
        self.assertEqual(len(self.api.items), 1)
        self.assertFalse(self.api.milestones)

    def test_metadata_update_sets_manager_and_milestone_date(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "release_date: TBD", "release_date: 2026-12-01"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.milestones[0]["due_on"], "2026-12-01T23:59:59Z")
        self.assertEqual(self.api.items[1]["assignees"], [{"login": "harrisonyhq"}])
        self.assertEqual(self.controller.load(1)[1]["revision"], 2)

    def test_existing_issue_description_is_preserved_and_assignee_can_clear(self):
        self.initialize()
        self.api.items[3]["body"] += "\nManual implementation notes."
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "assignee: @harrisonyhq", "assignee:"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertIn("Manual implementation notes.", self.api.items[3]["body"])
        self.assertEqual(self.api.items[3]["assignees"], [])

    def test_pending_commands_are_replayed_in_order(self):
        preview = self.api.comment(1, "/release_init --dry-run")
        apply = self.api.comment(1, "/release_init")
        self.controller.handle(
            {"issue": {"number": 1}, "comment": apply, "sender": apply["user"]},
            "issue_comment",
        )
        state = self.controller.load(1)[1]
        self.assertTrue(state["initialized"])
        self.assertEqual(
            state["processed_comments"], [str(preview["id"]), str(apply["id"])]
        )

    def test_body_comment_review_and_inline_form_one_reference_set(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_CONNECTOR": "M_REQ"}
        self.api.projects["I_4"] = {"P_STORE": "M_REQ2"}
        self.api.add_pr(10, "Implements #3")
        comment = self.api.comment(10, "Also #4")
        self.api.reviews[10] = [{"body": "Related #3", "user": {"type": "User"}}]
        self.api.inline[10] = [{"body": "Implements #4", "user": {"type": "User"}}]
        self.controller.reconcile(1)
        state = self.controller.load(1)[1]
        self.assertEqual(set(state["prs"]["10"]["requirements"]), {3, 4})
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertEqual(set(self.api.projects["I_10"]), {"P_CONNECTOR", "P_STORE"})
        self.assertEqual(self.api.items[1]["body"].count("/pull/10)"), 2)
        self.api.prs[10]["body"] = ""
        self.api.comments.remove(comment)
        self.controller.reconcile(1)
        self.assertEqual(
            set(self.controller.load(1)[1]["prs"]["10"]["requirements"]), {3, 4}
        )
        self.api.reviews[10] = []
        self.api.inline[10] = []
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])
        self.assertFalse(self.api.projects["I_10"])
        self.assertNotIn("/pull/10)", self.api.items[1]["body"])
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

    def test_pr_comment_event_triggers_reconciliation_without_a_command(self):
        self.initialize()
        self.api.add_pr(10, "")
        comment = self.api.comment(10, "Implements #3")
        self.controller.handle(
            {"issue": {"number": 10, "pull_request": {}}, "comment": comment},
            "issue_comment",
        )
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)

    def test_bot_receipts_are_excluded_from_mentions(self):
        self.initialize()
        self.api.add_pr(10, "")
        comment = self.api.comment(10, "Related #3")
        comment["user"]["type"] = "Bot"
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])

    def test_pr_state_change_refreshes_display_without_scope_revision(self):
        self.initialize()
        pr = self.api.add_pr(10, "#3")
        self.controller.reconcile(1)
        pr["merged_at"] = "2026-11-18T00:00:00Z"
        pr["state"] = "closed"
        self.controller.reconcile(1)
        self.assertIn("| merged", self.api.items[1]["body"])
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

    def test_conflicting_versions_and_manual_metadata_are_preserved(self):
        self.initialize()
        self.api.items[4]["milestone"] = {"number": 2, "title": "0.13.0"}
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "#3 #4")
        self.api.projects["I_10"] = {"P_MANUAL": "MANUAL"}
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])
        self.assertIn("conflicting", self.controller.load(1)[0]["body"])
        self.assertEqual(set(self.api.projects["I_10"]), {"P_REQ", "P_MANUAL"})
        self.api.prs[10]["body"] = ""
        self.controller.reconcile(1)
        self.assertEqual(self.api.projects["I_10"], {"P_MANUAL": "MANUAL"})

    def test_manual_milestone_clear_and_project_remove_persist(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "#3")
        self.controller.reconcile(1)
        self.api.prs[10]["milestone"] = None
        self.api.projects["I_10"].clear()
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])
        self.assertFalse(self.api.projects["I_10"])
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["mode"], "manual")

    def test_lost_milestone_response_recovers_automatic_ownership(self):
        self.initialize()
        self.api.add_pr(10, "#3")
        self.api.lose_milestone_response = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.controller.reconcile(1)
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["mode"], "auto")
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)

    def test_lost_project_response_recovers_item_identity(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "#3")
        self.api.lose_project_response = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.controller.reconcile(1)
        record = self.controller.load(1)[1]["prs"]["10"]["projects"]["P_REQ"]
        self.assertEqual(record["item"], self.api.projects["I_10"]["P_REQ"])
        self.api.projects["I_10"].clear()
        self.controller.reconcile(1)
        self.assertFalse(self.api.projects["I_10"])

    def test_bugfix_mention_uses_target_version_and_bugfix_project(self):
        state = self.initialize()
        bug = self.api.make_issue(8, "Bug", "", ["bug"])
        bug["milestone"] = {"number": 2, "title": "0.13.0"}
        self.api.items[8] = bug
        self.api.parents[8] = state["archive"]
        self.api.add_pr(10, "Fixes #8")
        self.controller.reconcile(1)
        self.assertIn("P_BUG", self.api.projects["I_10"])
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["last"], 2)
        self.assertEqual(self.api.parents[8], state["archive"])

    def test_changed_code_requires_current_verification(self):
        state = self.initialize()
        for entry in state["scope"].values():
            self.api.items[entry["number"]]["state"] = "closed"
        state["verification"] = {
            str(e["number"]): {
                "result": "Passed",
                "milestone": 1,
                "tested_commit": "old",
            }
            for e in state["scope"].values()
        }
        self.controller.save(1, state, "Previous candidate")
        self.assertEqual(self.controller.readiness(1)[1]["status"], "blocked")
        for record in state["verification"].values():
            record["tested_commit"] = "a" * 40
        self.controller.save(1, state, "Current candidate")
        self.assertEqual(self.controller.readiness(1)[1]["status"], "ready")

    def test_unassigned_bug_does_not_block_other_versions(self):
        self.initialize()
        self.api.items[99] = self.api.make_issue(
            99, "Unassigned", "source_bug_archive: #", ["bug"]
        )
        self.controller.reconcile(1)
        self.assertIn("待归档", self.controller.load(1)[0]["body"])

    def test_manual_milestone_ownership_and_bug_status(self):
        self.assertEqual(
            manual_mode({"mode": "auto", "last": 1}, None)["mode"], "manual"
        )
        self.assertEqual(
            manual_mode({"mode": "auto", "last": 1}, 1, True)["mode"], "manual"
        )
        self.assertEqual(
            bug_progress(False, {}, {"9"}, True), ("status:verification", "In review")
        )
        record = {"result": "Failed", "merged_prs": ["9"]}
        self.assertEqual(
            bug_progress(False, record, {"9"}, True), ("status:fixing", "In progress")
        )


if __name__ == "__main__":
    unittest.main()
