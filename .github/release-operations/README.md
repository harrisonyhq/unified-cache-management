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

A final `[]` creates an Issue; `[#123]` or `[https://github.com/OWNER/REPO/issues/123]`
reuses a local Issue. New Issue links replace the empty brackets. Issue IDs identify
synchronized scope. Creation-origin markers recover accepted requests after lost
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

## PR and Project synchronization

Reference a local Issue in PR descriptions, conversation comments, review bodies,
or inline review comments, using `#123`, `OWNER/REPO#123`, or a complete Issue URL.
Each reconciliation reads all current sources, deduplicates references, and diffs
the resulting set. References in automated Bot comments are excluded from inputs.
Execution receipts retain source permalinks.

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
inside each release requirement, with draft/open/closed/merged status. Changes to
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
