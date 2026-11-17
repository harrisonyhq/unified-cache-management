# Release operations

The controller manages a version through one Release Issue, a Milestone, native
requirement sub-issues, a Bug Archive, and linked bugfix PRs. The configuration
currently names the three personal Projects used for acceptance: 模型适配,
Connector v2, and Bugfix.

## Commands

Create an Issue using **Version plan**, keep stable requirement keys, and use
these commands as separate comments on that Release Issue:

| Command | Result |
| --- | --- |
| `/release_init --dry-run` | Show the initialization plan in the Bot result comment. |
| `/release_init` | Create/reuse the Milestone, requirements, native sub-issues and Bug Archive; append Issue references and replace the preview. |
| `/update --dry-run` | Compare the edited body with the last successfully synchronized scope. |
| `/update` | Add or reuse requirements, move removed entries out of the version, update controlled fields and parent relationships, and checkpoint the scope. |
| `/verify #123 passed https://...` | Record verification evidence, close the requirement/Bug and update its validation field. |
| `/verify #123 failed https://...` | Record failure and return the item to implementation/fixing. |
| `/release_ready` | Check synchronized scope, accepted requirements and verified target-version Bugs. |
| `/release_publish` | Publish a GitHub Release and Tag from the current default-branch commit, verify the receipt, then close the Milestone and Release Issue. |

Commands require repository write, maintain or admin permission. A removed scope
entry retains its Issue, original parent relationship and development history;
the controller clears its current-version Milestone. Record the reason in
`变更说明：...`. Requirement Project membership is selected in the Issue sidebar.

Create Bugs as direct sub-issues of the generated Bug Archive, or use **Version
test bug** and fill `源版本主 Bug Issue：#123`. The controller attaches the parent,
sets a default fix Milestone and adds the Bug to Bugfix. A bugfix PR uses
`关联缺陷：#123`; its Milestone follows the Bug. Maintainer edits and clears take
ownership of that object's Milestone, including changes detected through the
Issue event history. Multiple conflicting Bug targets appear in the result.

## Deployment and interfaces

Commit this directory and `release-operations.yml` to the default branch. Set
`UCM_PROJECT_TOKEN` as an Actions secret with access to the configured personal
Projects. Repository metadata uses `GITHUB_TOKEN`; only Project API requests use
the separate Project credential. Set `config.json.repository` and Project/field
IDs for the deployment target. The default-branch controller handles metadata
events; native Issue/PR contents are parsed as data.

`Controller.initialize_or_update`, `reconcile`, `verify`, `readiness` and `publish`
are callable Python interfaces. The CLI accepts `--config`, `--event` (a GitHub
event JSON file) and `--event-name`. It prints a structured completion result.
All mutations go through `GitHub.rest`/`graphql`.

The Bot owns one result comment with an embedded machine state: stable object
IDs, source keys, scope snapshots, Milestone ownership, verification evidence and
processed command IDs. Formal execution replaces previews. Workflow concurrency
serializes writers; each metadata event and later command replays earlier pending
comments in order, with periodic reconciliation as a fallback. A persistent
command cursor keeps completed history from being replayed. Creation-origin markers recover objects when a
request succeeded before its checkpoint was saved.

The publication adapter produces a GitHub Release and a Tag. Package, container,
and external distribution adapters can call the same readiness interface and
attach their own verified delivery receipts.

Verification records include the tested default-branch commit. Readiness checks
that accepted requirements and target Bugs were verified against the current
candidate, and that linked target-version fix PRs are merged into that candidate.
After a code change, submit verification for the new candidate before publishing.

Run the contract tests with:

```shell
python -m unittest discover -s .github/release-operations -p 'test_*.py' -v
```
