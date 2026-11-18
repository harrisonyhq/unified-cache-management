### Release Information

version: 0.12.0
branch: main
goal: Complete the planned integrations and interface updates.
release_manager:
test_lead:
release_date: TBD

### Requirements

- MiniMax M3.1 support
  with vLLM-Ascend v0.32.0 [Issue]()

  description:
    Integrate MiniMax M3.1 with vLLM-Ascend v0.32.0.

    Acceptance criteria:
    - Complete model loading and inference.
    - Verify cache reuse.
    - Document the supported configuration.

  assignee:

  pull_requests:

- Connector v2 integration
  for the first release [Issue]()

  description:
    Complete the initial Connector v2 integration.

    Scope:
    - Define the connector interface.
    - Implement the initial integration.
    - Provide configuration examples.

    Acceptance criteria:
    - Pass the functional tests.
    - Document the interface and usage.

  assignee:

  pull_requests:

### Release Notes

- [ ] Summarize new features.
- [ ] Summarize bug fixes.
- [ ] Document compatibility changes.
- [ ] Document upgrade instructions.
- [ ] Document known issues.

### Functional Tests

- [ ] Verify the requirements in this release.
- [ ] Verify the bug fixes targeted for this release.
- [ ] Record test results and evidence.
- [ ] Review unresolved bugs and release blockers.

### Documentation Tests

- [ ] Verify installation instructions.
- [ ] Verify configuration and usage examples.
- [ ] Update the user guide.
- [ ] Update the developer guide.

### Artifacts

- [ ] Build and verify the wheel package.
- [ ] Build and verify the container image.
- [ ] Confirm artifact versions match the release version.

### Release Steps

- [ ] Confirm the release scope.
- [ ] Merge the required pull requests.
- [ ] Complete requirement and bug fix verification.
- [ ] Resolve release blockers.
- [ ] Finalize the release notes.
- [ ] Run /release_ready and review the result.
- [ ] Run /release_publish.
- [ ] Verify the GitHub Release and tag.
- [ ] Publish the wheel package and container image.
- [ ] Verify the published artifacts.
- [ ] Confirm the milestone and release issue are closed.

### Change Log

reason: Initial scope.
