"""Reconcile PR metadata from current Issue mentions and native Projects."""

from ownership import manual_mode, milestone_number
from syntax import OperationError, parse_release, patch_pull_requests, references


class PullRequestAssociations:
    def memberships(self, item):
        result, cursor = {}, None
        while True:
            page = self.api.graphql(
                """query($id:ID!,$after:String){node(id:$id){
                ... on Issue{projectItems(first:100,after:$after){nodes{id project{id}} pageInfo{hasNextPage endCursor}}}
                ... on PullRequest{projectItems(first:100,after:$after){nodes{id project{id}} pageInfo{hasNextPage endCursor}}}
                }}""",
                id=item["node_id"],
                after=cursor,
            )["node"]["projectItems"]
            result.update({n["project"]["id"]: n["id"] for n in page["nodes"]})
            if not page["pageInfo"]["hasNextPage"]:
                return result
            cursor = page["pageInfo"]["endCursor"]

    def discussion_references(self, pr):
        sources = [(pr.get("body") or "", pr["html_url"])]
        for path in (
            f"issues/{pr['number']}/comments",
            f"pulls/{pr['number']}/reviews",
            f"pulls/{pr['number']}/comments",
        ):
            for comment in self.api.pages(path):
                if comment.get("user", {}).get("type") == "Bot":
                    continue
                sources.append(
                    (comment.get("body") or "", comment.get("html_url", pr["html_url"]))
                )
        refs = {}
        for text, url in sources:
            for number in references(text, self.repository):
                refs.setdefault(number, []).append(url)
        return refs

    def sync_pr_projects(self, release, state, pr, record, desired):
        current = self.memberships(pr)
        managed = record.setdefault("projects", {})
        for project in desired | set(managed):
            existing = managed.get(project)
            if project not in desired:
                if existing and existing["mode"] == "auto" and project in current:
                    self.api.graphql(
                        """mutation($project:ID!,$item:ID!){deleteProjectV2Item(input:{projectId:$project,itemId:$item}){deletedItemId}}""",
                        project=project,
                        item=current[project],
                    )
                managed.pop(project, None)
                continue
            if existing is None:
                existing = managed[project] = {
                    "mode": "manual" if project in current else "auto",
                    "item": current.get(project),
                }
            if (
                existing["mode"] == "auto"
                and existing.get("item")
                and project not in current
            ):
                existing["mode"] = "manual_removed"
            if existing["mode"] == "auto" and project in current:
                existing["item"] = current[project]
            if existing["mode"] == "auto" and project not in current:
                # Save ownership before an API request whose response can be lost.
                self.save(release, state, f"Synchronizing PR #{pr['number']} Projects")
                item = self.api.graphql(
                    """mutation($project:ID!,$content:ID!){addProjectV2ItemById(input:{projectId:$project,contentId:$content}){item{id}}}""",
                    project=project,
                    content=pr["node_id"],
                )["addProjectV2ItemById"]["item"]
                existing["item"] = item["id"]

    def sync_pull_requests(self, release, state, event=None):
        event = event or {}
        target = event.get("pull_request", {}).get("number")
        if "pull_request" in event.get("issue", {}):
            target = event["issue"]["number"]
        requirements = set()
        for number in self.releases():
            _, other = self.load(number)
            requirements.update(e["number"] for e in other["scope"].values())
        local_requirements = {e["number"] for e in state["scope"].values()}
        local_bugs = {int(n) for n in state["bugs"]}
        rows, problems = {}, []
        prs = getattr(self, "_pr_snapshot", None)
        if prs is None:
            prs = self.api.pages("pulls?state=all")
        for pr in prs:
            key = str(pr["number"])
            if target and pr["number"] != target:
                for number in state["prs"].get(key, {}).get("requirements", []):
                    if number in local_requirements:
                        rows.setdefault(number, []).append(pr)
                continue
            cache = getattr(self, "_discussion_snapshot", None)
            if cache is None:
                cache = {}
            if pr["number"] not in cache:
                cache[pr["number"]] = self.discussion_references(pr)
            sources = cache[pr["number"]]
            if (
                not (set(sources) & (local_requirements | local_bugs))
                and key not in state["prs"]
            ):
                continue
            linked, reqs, bugs = [], [], []
            for number in sources:
                try:
                    item = self.api.rest(f"issues/{number}")
                except OperationError as error:
                    if "404" in str(error):
                        continue
                    raise
                if "pull_request" in item:
                    continue
                is_bug = "bug" in {label["name"] for label in item["labels"]}
                if number in requirements or is_bug:
                    linked.append(item)
                    (bugs if is_bug else reqs).append(number)
            current = milestone_number(pr)
            record = state["prs"].setdefault(
                key, {"mode": "auto" if current is None else "manual", "last": current}
            )
            if "pending_milestone" in record:
                pending = record.pop("pending_milestone")
                if current == pending:
                    record["last"] = current
            record = manual_mode(record, current)
            record = self.milestone_history(pr["number"], record)
            record.update(
                bugs=bugs,
                requirements=reqs,
                merged=bool(pr.get("merged_at")),
                sources={str(n): sources[n] for n in reqs + bugs},
            )
            state["prs"][key] = record
            targets = {milestone_number(i) for i in linked}
            record["conflict"] = len(targets) > 1 and record["mode"] == "auto"
            if len(targets) > 1:
                if record["conflict"]:
                    problems.append(
                        f"PR #{key}: linked Issues have conflicting Milestones; set the PR Milestone manually"
                    )
            elif record["mode"] == "auto":
                desired = next(iter(targets), None)
                if current != desired:
                    record["pending_milestone"] = desired
                    self.save(release, state, f"Synchronizing PR #{key} Milestone")
                    self.api.rest(f"issues/{key}", "PATCH", {"milestone": desired})
                    record.pop("pending_milestone", None)
                record["last"] = desired
            record["target_title"] = next(
                (
                    i["milestone"]["title"]
                    for i in linked
                    if i.get("milestone") and i["milestone"]["number"] == record["last"]
                ),
                (pr.get("milestone") or {}).get("title"),
            )
            projects = set()
            for item in linked:
                if item["number"] in reqs:
                    projects.update(self.memberships(item))
            if bugs:
                projects.add(self.config["projects"]["Bugfix"]["id"])
            self.sync_pr_projects(release, state, pr, record, projects)
            for number in reqs:
                if number in local_requirements:
                    rows.setdefault(number, []).append(pr)
        source = self.issue(release)
        if (
            "### Requirements\n" in source["body"]
            or "### Requirements\r\n" in source["body"]
        ):
            # Validate before writeback; requirement edits stay pending until /update.
            try:
                parse_release(source["body"], self.repository)
            except OperationError as error:
                problems.append("Release body requires attention: " + str(error))
                return problems
            patched = patch_pull_requests(source["body"], rows, self.repository)
            latest = self.issue(release)
            if latest["body"] != source["body"]:
                problems.append(
                    "Release body changed during synchronization; retry /update"
                )
            elif patched != source["body"]:
                self.api.rest(f"issues/{release}", "PATCH", {"body": patched})
        return problems
