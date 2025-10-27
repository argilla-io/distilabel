# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import datetime
from typing import List, Union

import pandas as pd
import requests
import mkdocs_gen_files


REPOSITORY = "argilla-io/distilabel"
DATA_PATH = "sections/community/popular_issues.md"

# public_repo and read:org scopes are required
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

EMPTY_FRAME = pd.DataFrame(
    {
        "Issue": [],
        "State": [],
        "Created at": [],
        "Milestone": [],
        "Reactions": [],
        "Comments": [],
        "URL": [],
        "Author": [],
        "Author association": [],
    }
)

def fetch_data_from_github(repository, auth_token):
    if auth_token is None:
        return EMPTY_FRAME

    headers = {"Authorization": f"token {auth_token}", "Accept": "application/vnd.github.v3+json"}
    issues_data = []

    print(f"Fetching issues from {repository}...")
    with requests.Session() as session:
        session.headers.update(headers)

        owner, repo_name = repository.split("/")
        issues_url = f"https://api.github.com/repos/{owner}/{repo_name}/issues?state=all"
        filtered_count = 0
        total_count = 0

        while issues_url:
            response = session.get(issues_url)
            issues = response.json()
            print(f"Fetched batch of issues... {len(issues)}")

            for issue in issues:
                total_count = total_count + 1
                if "pull_request" in issue:
                    filtered_count = filtered_count + 1
                    continue
                issues_data.append(
                    {
                        "Issue": f"{issue['number']} - {issue['title']}",
                        "State": issue["state"],
                        "Created at": issue["created_at"],
                        "Milestone": (issue.get("milestone") or {}).get("title"),
                        "Reactions": issue["reactions"]["total_count"],
                        "Comments": issue["comments"],
                        "URL": issue["html_url"],
                        "Author": issue["user"]["login"],
                        "Author association": issue["author_association"],
                    }
                )

            issues_url = response.links.get("next", {}).get("url", None)

    print(f"Filtered out {filtered_count} issues from {total_count}")

    if not issues_data:
        print("No issues data collected (all items were filtered out)")
        return EMPTY_FRAME
    
    return pd.DataFrame(issues_data)


with mkdocs_gen_files.open(DATA_PATH, "w") as f:
    df = fetch_data_from_github(REPOSITORY, GITHUB_TOKEN)

    if "State" in df.columns:
        open_issues = df.loc[df["State"] == "open"]
    else:
        print("WARNING: 'State' column not found in DataFrame")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns) if not df.empty else 'No columns (empty DataFrame)'}")
        print(f"DataFrame first few rows: {df.head().to_dict() if not df.empty else 'Empty DataFrame'}")
        open_issues = EMPTY_FRAME

    engagement_df = (
        open_issues[["URL", "Issue", "Reactions", "Comments"]]
        .sort_values(by=["Reactions", "Comments"], ascending=False)
        .head(10)
        .reset_index()
    )

    community_issues = df[df["Author association"] != "MEMBER"]
    community_issues_df = (
        community_issues[["URL", "Issue", "Created at", "Author", "State"]]
        .sort_values(by=["Created at"], ascending=False)
        .head(10)
        .reset_index()
    )

    df["Milestone"] = df["Milestone"].astype(str).fillna("")
    planned_issues = df[
        ((df["Milestone"].str.startswith("v1")) & (df["State"] == "open"))
        | ((df["Milestone"].str.startswith("1")) & (df["State"] == "open"))
    ]
    planned_issues_df = (
        planned_issues[["URL", "Issue", "Created at", "Milestone", "State"]]
        .sort_values(by=["Milestone"], ascending=True)
        .head(10)
        .reset_index()
    )

    f.write('=== "Most engaging open issues"\n\n')
    f.write("    | Rank | Issue | Reactions | Comments |\n")
    f.write("    |------|-------|:---------:|:--------:|\n")
    for ix, row in engagement_df.iterrows():
        f.write(f"    | {ix+1} | [{row['Issue']}]({row['URL']}) | 👍 {row['Reactions']} | 💬 {row['Comments']} |\n")

    f.write('\n=== "Latest issues open by the community"\n\n')
    f.write("    | Rank | Issue | Author |\n")
    f.write("    |------|-------|:------:|\n")
    for ix, row in community_issues_df.iterrows():
        state = "🟢" if row["State"] == "open" else "🟣"
        f.write(f"    | {ix+1} | {state} [{row['Issue']}]({row['URL']}) | by **{row['Author']}** |\n")

    f.write('\n=== "Planned issues for upcoming releases"\n\n')
    f.write("    | Rank | Issue | Milestone |\n")
    f.write("    |------|-------|:------:|\n")
    for ix, row in planned_issues_df.iterrows():
        state = "🟢" if row["State"] == "open" else "🟣"
        f.write(f"    | {ix+1} | {state} [{row['Issue']}]({row['URL']}) |  **{row['Milestone']}** |\n")

    today = datetime.today().date()
    f.write(f"\nLast update: {today}\n")
