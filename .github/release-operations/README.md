# Release operations

Create a version using **Release Checklist**. Its structured Markdown requirements
are the release scope. The controller manages a Milestone, flat requirement Issues,
a native Bug Archive under the Release Issue, and PR associations. The complete
body is available in [the example](examples/release-checklist.md).

## Input and commands

All template headings and field names use English and ASCII punctuation.
Requirements start with a top-level `- `. Title continuation and fields use two
spaces; description and generated PR content use at least four spaces. Titles
are joined with spaces when an Issue is created; descriptions retain paragraphs
and nested lists.

```markdown
- Connector v2
  initial integration []

  description:
    Implement the connector.

    Acceptance criteria:
    - Pass the integration test.

  assignee: @username

  pull_requests:
```

A final `[]` creates an Issue. An existing requirement ends in a native `#123`
reference. Markdown links, bare URLs, and older bracket formats remain accepted
input; writeback normalizes them to a title plus `#123` so GitHub can display
native Issue references in lists. The whole title may be pasted as a Markdown link.
Issue IDs identify synchronized scope. Creation-origin markers recover accepted requests after lost
responses. Requirement parent relationships are managed by developers in GitHub.
The controller retains historical snapshots and existing parent relationships.

| Command | Result |
| --- | --- |
| `/release_init --dry-run` | Validate input and preview object creation/reuse. |
| `/release_init` | Create/reuse Milestone, requirements, and Bug Archive; write Issue links; replace the preview. |
| `/update --dry-run` | Preview differences from the last synchronized scope. |
| `/update` | Synchronize requirement additions/removals, assignees, metadata, and PR associations. |
| `/verify #123 passed https://...` | Record candidate-bound evidence and accept the requirement/Bug. |
| `/verify #123 failed https://...` | Record failure and reopen the item. |
| `/release_ready` | Check synchronized scope, current-candidate verification, and linked PRs. |
| `/release_publish` | Publish a GitHub Release and Tag, verify the receipt, and close the Milestone/Release Issue. |

Commands require repository write, maintain, or admin permission. Each command
is a separate comment on the Release Issue. Update `reason:` in Change Log for
scope removals. Removal retains the Issue and clears its current release
Milestone when that value still belongs to this version. Existing Issue titles
and descriptions are maintained in the Issue; the release entry stores the
version's plan. Assignee edits, including clearing an assigned owner, synchronize.

Release Information contains `version`, `branch`, `goal`, `release_manager`,
`test_lead`, and `release_date`. Roles use one @username or an empty value. The
release manager is assigned to the Release Issue. The goal and roles are recorded
in the Milestone description; `release_date` accepts YYYY-MM-DD or TBD and controls
the Milestone due date. Verification and publication use the configured branch.

## Release navigation and overview

The Bot maintains two bounded regions in the Issue body. The navigation region
sits directly under Release Information and links the Milestone and Bug Archive.
The Release Overview region is always at the bottom, with one table for requirements
and actual Bugs: Issue, Pull requests, Assignee, State / Validation. Issue and PR
cells use native #number references. Lists can expand native references into title
and state cards; the table also retains explicit Issue and PR states.

Region markers start/end with `ucm-release-navigation` and `ucm-release-overview`.
Generated content is excluded from scope parsing and revision comparison. Human
fields, descriptions, custom sections, and checklist edits outside those regions
are retained. Repeated refreshes replace the same regions.

The bottom Bot region contains one latest operation line: a running arrow and
workflow link during execution, a green check on success, or a red cross with a
short reason on failure. Initialization, synchronization, verification, readiness,
and publication use the same region. Nested steps preserve their outer operation.
Preview plans remain in a collapsible panel until formal execution replaces them.

The trusted checkpoint comment contains only machine data and is automatically
collapsed. Existing notification text is removed in place, retaining object maps,
verification, and command cursors. Results, links, tables, readiness, and scope
history are displayed in the body. The workflow finalizer marks interrupted
operations from its own run as failed without overwriting another run's status.

## PR and Project synchronization

Formal initialization automatically pins the Release Issue above the repository
Issues list. The Bug Archive remains accessible from the release receipt and
native sub-issue relationship. Dry-runs leave pins untouched. Repeated initialization
or updates retry a pending pin; occupied slots are reported without removing other
Issues. Manual pins/unpins are retained. Successful publication removes only a
Bot-managed Release pin. Published versions stay unpinned on repeated initialization.

The Bot adds an `eyes` reaction to an authorized version-command comment when the
controller starts processing it. PR conversation comments with local Issue
references receive the same acknowledgement before reconciliation, including
edits that remove a previous reference. Repeated requests reuse the Bot's reaction.
The reaction means processing has started; execution results remain in the release
receipt. Deleted comments, automated Bot comments, and unrelated conversation
messages receive no acknowledgement. Reaction API failures are logged while the
operation continues. A queued workflow acknowledges when its controller starts.

Reference a local Issue in PR descriptions, conversation comments, review bodies,
or inline review comments, using `#123`, `OWNER/REPO#123`, or a complete Issue URL.
Each reconciliation reads all current sources, deduplicates references, and diffs
the resulting set. References in automated Bot comments are excluded from inputs.
Execution receipts retain source permalinks.

GitHub's native Development sidebar links are an additional association source.
The controller reads `closedByPullRequestsReferences` using the repository token,
includes manually linked and closed/merged PRs, and combines them with textual
references. Removing a sidebar link preserves any remaining textual relationship;
removing the last source cleans up automatic metadata and the table row. Native
links have the same Milestone/Project ownership and conflict rules as mentions.
Queries are paginated, restricted to this repository, and cached within a run.
A failed native query aborts association reconciliation before cleanup.

Sidebar link/unlink changes are picked up by the 15-minute schedule, `/update`,
or manual workflow dispatch. PR lifecycle events refresh native links as well.
GitHub Actions does not expose the timeline connected/disconnected activities as
direct workflow triggers, so sidebar edits use these reconciliation entry points.

- A scoped requirement supplies its Milestone and current native Projects.
- A Bug supplies its target fix Milestone and the configured Bugfix Project.
- Multiple requirements supply the union of their Projects.
- Conflicting Milestones require an explicit maintainer choice and block readiness
  while automatic ownership is unresolved.
- Maintainer Milestone changes/clears persist. Existing manual Project memberships
  remain; removing a managed membership manually suppresses its automatic re-add.
- Removing the last reference removes automatic associations and the requirement's
  generated PR row. A remaining mention in any source retains the relationship.

Choose requirement Projects in the Issue sidebar. The controller queries native
Project membership and synchronizes associated PRs. `pull_requests:` is generated
inside each release requirement, with native PR references and draft/open/closed/merged status. Changes to
this field alone refresh display without creating scope revisions.

PR creation, body edits, state changes, and ordinary comment creation/edit/deletion
trigger Release operations. Review events trigger a credential-free **Release
review observer**; its completion wakes the trusted default-branch controller.
The observer checks out no code and supplies no artifacts to the receiver.
`/release_init`, `/update`, manual dispatch, and the 15-minute schedule also
reconcile existing references and subsequent Issue Milestone/Project changes.

## Bugs, verification, and publication

Initialization creates a Bug Archive as a native sub-issue of the Release Issue,
with the source version's Milestone. Create actual Bugs under this archive, or
use **Version test bug** with `source_bug_archive: #123`. Bugs default to the
source version's Milestone; maintainer retargets/clears persist. Their parent
continues to identify the source version. Legacy Bug source fields remain readable.
Bug and fix PRs enter Bugfix. The Bug status follows implementation, merged fixes,
and verification. Filter parent Issue for source-version defects and Milestone
for target-version fixes.

Verification records bind evidence to the candidate commit. Readiness requires
accepted requirements and target Bugs on that candidate and merged linked PRs
whose merge commits are included. After a candidate change, record new evidence.
The publication adapter creates GitHub Release + Tag. Wheel/container distribution
steps remain checklist tasks for their publishing adapters.

## Deployment and interfaces

Deploy this directory and both workflows to the default branch. Configure
`config.json.repository`, Project/field IDs, and the `UCM_PROJECT_TOKEN` Actions
secret. Repository metadata uses `GITHUB_TOKEN`; Project API calls use the Project
credential. The workflow runs trusted default-branch code and reads PR content
as data. No PR code build is required by this metadata workflow.

`Controller.initialize_or_update`, `reconcile`, `verify`, `readiness`, and `publish`
are Python interfaces. The CLI accepts `--config`, `--event`, and `--event-name`.
Mutations go through `GitHub.rest`/`graphql`. The parser and PR association logic
are separate modules using only the Python standard library.

One Bot result comment stores scope, object IDs, ownership, verification, revision
history, and processed command IDs. Formal execution replaces previews. Workflow
concurrency serializes writers. Later events replay pending release commands in
order; a persistent cursor prevents completed history from replaying. Periodic
reconciliation supplements event delivery.

Run contracts with:

```shell
python -m unittest discover -s .github/release-operations -p 'test_*.py' -v
```
