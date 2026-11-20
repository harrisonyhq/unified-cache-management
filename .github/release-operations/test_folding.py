import unittest
from html import escape

import test_controller as fixtures
from folding import fold_requirements, patch_issue_states
from presentation import OVERVIEW_START, release_body
from syntax import OperationError, parse_release, patch_pull_requests, patch_references


class FoldingContracts(unittest.TestCase):
    def test_status_survives_summary_closing_tag_on_its_own_line(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        entries = parse_release(body, fixtures.REPOSITORY)["entries"]
        body = patch_references(body, {entries[0]["key"]: 14}, fixtures.REPOSITORY)
        body = patch_issue_states(body, {14: "open"}, fixtures.REPOSITORY)
        body = body.replace("open</summary>", "open\n</summary>", 1)
        expected = parse_release(body, fixtures.REPOSITORY)
        body = patch_issue_states(body, {14: "closed"}, fixtures.REPOSITORY)
        self.assertEqual(parse_release(body, fixtures.REPOSITORY), expected)
        self.assertEqual(body.count("<!-- ucm-issue-state:"), 1)
        self.assertIn("🟣 closed</summary>", body)

    def test_open_closed_and_reopened_status_preserves_scope(self):
        body = fold_requirements(fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY)
        entries = parse_release(body, fixtures.REPOSITORY)["entries"]
        body = patch_references(
            body, {e["key"]: n for e, n in zip(entries, [14, 13])}, fixtures.REPOSITORY
        )
        expected = parse_release(body, fixtures.REPOSITORY)
        for state in ("open", "closed", "open"):
            body = patch_issue_states(
                body, {14: state, 13: "open"}, fixtures.REPOSITORY
            )
            self.assertIn(
                ("🟢 open" if state == "open" else "🟣 closed") + "</summary>", body
            )
            self.assertEqual(parse_release(body, fixtures.REPOSITORY), expected)
            self.assertEqual(
                patch_issue_states(body, {14: state, 13: "open"}, fixtures.REPOSITORY),
                body,
            )
            self.assertEqual(body.count("<!-- ucm-issue-state:"), 2)

    def test_status_rendering_preserves_error_reporting_for_invalid_input(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
            "</summary>", "", 1
        )
        self.assertEqual(
            patch_issue_states(body, {14: "open"}, fixtures.REPOSITORY), body
        )
        rendered = release_body(
            body,
            {"scope": {}, "bugs": {}, "observed": {"14": {"state": "open"}}},
            fixtures.REPOSITORY,
        )
        self.assertIn("### Release Overview", rendered)

    def test_real_sync_updates_status_without_scope_revision(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        before = x.initialize()
        for state in ("closed", "open"):
            x.api.items[3]["state"] = state
            x.controller.reconcile(1)
            after = x.controller.load(1)[1]
            self.assertIn(f"ucm-issue-state:{state}", x.api.items[1]["body"])
            self.assertEqual(after["scope"], before["scope"])
            self.assertEqual(after["revision"], before["revision"])
            self.assertEqual(len(x.api.items), 4)
        x.controller.initialize_or_update(1, "update")
        self.assertEqual(x.controller.load(1)[1]["revision"], before["revision"])

    def test_last_requirements_section_preserves_managed_regions(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        prefix, rest = body.split("### Requirements", 1)
        requirements, change = rest.split("### Change Log", 1)
        body = prefix + "### Change Log" + change + "\n### Requirements" + requirements
        body = release_body(body, {"scope": {}, "bugs": {}})
        updated = patch_pull_requests(body, {}, fixtures.REPOSITORY)
        self.assertEqual(
            parse_release(updated, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )
        self.assertIn(OVERVIEW_START, updated)

    def test_folded_scope_addition_and_removal(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        x.initialize()
        x.api.items[1]["body"] = x.api.items[1]["body"].replace(
            "### Change Log",
            "<details>\n<summary>Additional feature</summary>\n\nassignee:\n\ndescription:\n  Additional work.\n\npull_requests:\n\n</details>\n\n### Change Log",
        )
        x.controller.initialize_or_update(1, "update")
        self.assertIn(
            "Additional feature #5 <!-- ucm-issue-state:open -->",
            x.api.items[1]["body"],
        )
        begin = x.api.items[1]["body"].index("<details>\n<summary>Additional feature")
        end = x.api.items[1]["body"].index("</details>", begin) + len("</details>")
        x.api.items[1]["body"] = (
            x.api.items[1]["body"][:begin] + x.api.items[1]["body"][end:]
        )
        x.controller.initialize_or_update(1, "update")
        self.assertIsNone(x.api.items[5]["milestone"])
        self.assertEqual(x.controller.load(1)[1]["revision"], 3)

    def test_conversion_preserves_scope_and_is_idempotent(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        self.assertEqual(
            parse_release(body, fixtures.REPOSITORY),
            parse_release(fixtures.PLAN, fixtures.REPOSITORY),
        )
        self.assertEqual(fold_requirements(body, fixtures.REPOSITORY), body)
        self.assertIn("<summary>Connector v2\ninitial integration []</summary>", body)

    def test_reference_formats_and_html_titles_remain_stable(self):
        for title in (
            "Integration & API <v2>",
            f"Integration [Issue](https://github.com/{fixtures.REPOSITORY}/issues/14)",
            f"Integration https://github.com/{fixtures.REPOSITORY}/issues/14",
            f"[GLM \\[API\\]](https://github.com/{fixtures.REPOSITORY}/issues/14)",
            "Integration #14",
        ):
            with self.subTest(title=title):
                body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
                    "<summary>Connector v2\ninitial integration []</summary>",
                    "<summary>" + escape(title, quote=False) + "</summary>",
                )
                entry = parse_release(body, fixtures.REPOSITORY)["entries"][0]
                patched = patch_references(
                    body, {entry["key"]: 14}, fixtures.REPOSITORY
                )
                self.assertIn("#14</summary>", patched)
                self.assertEqual(
                    patch_references(patched, {}, fixtures.REPOSITORY), patched
                )
                self.assertEqual(
                    parse_release(patched, fixtures.REPOSITORY)["entries"][0]["title"],
                    entry["title"],
                )

    def test_pr_refresh_keeps_description_and_card_boundaries(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        entries = parse_release(body, fixtures.REPOSITORY)["entries"]
        body = patch_references(
            body, {e["key"]: n for e, n in zip(entries, [14, 13])}, fixtures.REPOSITORY
        )
        rows = {14: [{"number": 29, "state": "closed", "merged_at": "today"}]}
        updated = patch_pull_requests(body, rows, fixtures.REPOSITORY)
        self.assertIn("  - #29 | merged", updated)
        self.assertEqual(updated.count("</details>"), 2)
        self.assertEqual(
            parse_release(updated, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )
        self.assertEqual(
            patch_pull_requests(updated, rows, fixtures.REPOSITORY), updated
        )
        removed = patch_pull_requests(updated, {}, fixtures.REPOSITORY)
        self.assertNotIn("#29", removed)
        self.assertEqual(
            parse_release(removed, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )

    def test_invalid_cards_fail_before_creation(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        variants = [
            body.replace("</details>", "", 1),
            body.replace("assignee: @harrisonyhq", "owner: @harrisonyhq", 1),
            body.replace("description:", "assignee:\n\ndescription:", 1),
            body.replace("  Implement the connector.", "Implement the connector.", 1),
            body.replace("</summary>", "", 1),
        ]
        for value in variants:
            with self.subTest(body=value), self.assertRaises(OperationError):
                parse_release(value, fixtures.REPOSITORY)

    def test_plain_title_creation_and_repeated_update(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        x.initialize()
        self.assertIn(
            "<summary>Connector v2\ninitial integration #3 <!-- ucm-issue-state:open -->",
            x.api.items[1]["body"],
        )
        x.controller.initialize_or_update(1, "update")
        self.assertEqual(len(x.api.items), 4)
        self.assertEqual(x.controller.load(1)[1]["revision"], 1)

    def test_migration_preserves_initialized_scope_and_objects(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        before = x.initialize()
        x.api.items[1]["body"] = fold_requirements(
            x.api.items[1]["body"], fixtures.REPOSITORY
        )
        x.controller.initialize_or_update(1, "update")
        after = x.controller.load(1)[1]
        self.assertEqual(after["scope"], before["scope"])
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(len(x.api.items), 4)

    def test_lost_creation_response_is_recovered(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        x.api.lose_response_for = "Store interface"
        with self.assertRaises(OperationError):
            x.initialize()
        x.initialize()
        self.assertEqual(len(x.api.items), 4)
        self.assertIn("#4 <!-- ucm-issue-state:open -->", x.api.items[1]["body"])
