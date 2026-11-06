"""Inspects all open work items on Gitlab and tags the ones, which haven't seen any updates in more than 90 days."""

import os
from datetime import datetime, timedelta

import requests
from tqdm import tqdm

# GitHub settings
GITLAB_URL = "https://gitlab.cern.ch/api/v4"
OWNER = "blond"
REPO = "BLonD"
TOKEN = os.environ[
    "GITLAB_ACCESS_TOKEN"
]  # Use a personal access token for authentication
# Set up headers for authentication
headers = {
    "PRIVATE-TOKEN": f"{TOKEN}",
}

DRAFT = True  # disables writing --> set false before trying to tag


# Get all open issues from the repository
def _get_issues() -> list[dict]:
    page = 1  # Gitlab numbering starts at 1
    issues = []
    while True:
        issues_url = f"{GITLAB_URL}/projects/{OWNER}%2F{REPO}/issues?state=opened&page={page}&per_page=100"
        response = requests.get(issues_url, headers=headers)

        if response.status_code != 200:  # noqa: PLR2004
            raise ValueError(f"{response.status_code=}")
        if not response.json():
            break
        issues.extend(response.json())
        page += 1
    return issues


# Get the current date minus three months
def _get_datetime_three_months_ago():
    return datetime.now() - timedelta(days=90)


# Add the "Stale" label to the issue
def _add_stale_label(issue_number):
    url = f"{GITLAB_URL}/projects/{OWNER}%2F{REPO}/issues/{issue_number}"

    if not DRAFT:
        response = requests.put(
            url,
            headers=headers,
            data={"add_labels": "Stale"},  # comma-separated string
        )

        if response.status_code == 200:  # noqa: PLR2004
            print(f"Label 'Stale' added to issue #{issue_number}")
        else:
            print(
                f"Failed to add label to issue #{issue_number} with "
                f"{response.status_code=}"
            )
    else:
        print(f"Would label 'Stale' added to issue #{issue_number}")


# Post a comment to the issue creator
def _post_comment(issue_number, creator_login):
    comment_url = (
        f"{GITLAB_URL}/projects/{OWNER}%2F{REPO}/issues/{issue_number}/notes"
    )
    comment_body = (
        f"Hi @{creator_login},\n\n"
        "This issue hasn't seen any activity in the past 3 months, so it was automatically tagged. "
        "Is this work item still relevant, or can it be closed?\n\n"
        "Thanks for taking the time to provide an update!"
    )
    if not DRAFT:
        response = requests.post(
            comment_url, headers=headers, json={"body": comment_body}
        )
        if response.status_code == 201:  # noqa: PLR2004
            print(f"Comment posted on issue #{issue_number}")
        else:
            print(f"Failed to post comment on issue #{issue_number}")
    else:
        print(f"Would post comment\n{comment_body} \non issue #{issue_number}")


# Main function to process issues
def _process_issues():
    issues = _get_issues()
    three_months_ago = _get_datetime_three_months_ago()

    num_tagged = 0

    for issue in tqdm(issues):
        # Skip issues that are closed
        if issue["state"] == "closed":
            continue

        # Skip issues that were already marked as stale
        if "Stale" in issue["labels"]:
            continue
        # Check if the issue's last update was older than 3 months
        updated_at = datetime.strptime(
            issue["updated_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        if updated_at < three_months_ago:
            num_tagged += 1
            # Add "stale" label
            _add_stale_label(issue["iid"])

            # Post a comment to the issue creator
            creator_login = issue["author"]["username"]
            _post_comment(issue["iid"], creator_login)
    tagstring = "Would've tagged" if DRAFT else "Tagged"
    print(f"{tagstring} {num_tagged} issues with stale labels")


if __name__ == "__main__":
    _process_issues()
