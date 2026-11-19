"""Repository Issue pinning using the repository credential."""

from syntax import OperationError


class ReleasePinning:
    def release_pin(self, release, state, enabled=True):
        if enabled and state.get("published"):
            return
        record = state.setdefault("release_pin", {})
        owner, name = self.repository.split("/")
        try:
            snapshot = self.api.call(
                "graphql",
                "POST",
                {
                    "query": """query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){issue(number:$number){id isPinned} pinnedIssues(first:3){nodes{issue{number} pinnedBy{__typename login}}}}}""",
                    "variables": {"owner": owner, "name": name, "number": release},
                },
            )["data"]["repository"]
            issue = snapshot["issue"]
            existing = next(
                (
                    p
                    for p in snapshot["pinnedIssues"]["nodes"]
                    if p["issue"]["number"] == release
                ),
                None,
            )
            if issue["isPinned"]:
                if not record or (
                    record.get("status") == "pinned"
                    and existing
                    and existing["pinnedBy"]["__typename"] == "User"
                ):
                    record.update(mode="manual", status="pinned")
            elif record.get("status") == "pinned":
                record.update(mode="manual", status="unpinned")
            if enabled and record.get("mode") == "manual":
                return
            if not enabled and record.get("mode") != "auto":
                return
            if issue["isPinned"] == enabled:
                record["status"] = "pinned" if enabled else "unpinned"
                state.pop("pin_warning", None)
                return
            if enabled and len(snapshot["pinnedIssues"]["nodes"]) >= 3:
                raise OperationError(
                    "All three repository pin slots are occupied; Release pinning is pending"
                )
            record.update(
                mode="auto", status="pending_pin" if enabled else "pending_unpin"
            )
            self.save(release, state, "Updating Release Issue pin")
            action = "pinIssue" if enabled else "unpinIssue"
            self.api.call(
                "graphql",
                "POST",
                {
                    "query": f"mutation($id:ID!){{{action}(input:{{issueId:$id}}){{issue{{id isPinned}}}}}}",
                    "variables": {"id": issue["id"]},
                },
            )
            record["status"] = "pinned" if enabled else "unpinned"
            state.pop("pin_warning", None)
        except OperationError as error:
            state["pin_warning"] = str(error)
