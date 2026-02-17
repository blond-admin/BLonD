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


# Get all open issues from the repository
def get_issues() -> list[dict]:
    page = 0
    issues = []
    while True:
        issues_url = f"{GITLAB_URL}/projects/{OWNER}%2F{REPO}/issues?state=opened&page={page}&per_page=100"
        response = requests.get(issues_url, headers=headers)

        if response.status_code != 200:
            raise ValueError(f"{response.status_code=}")
        if not response.json():
            break
        issues.extend(response.json())
        page += 1
    return issues


# Get the current date minus three months
def get_datetime_three_months_ago():
    return datetime.now() - timedelta(days=90)


# Add the "stale" label to the issue
def add_stale_label(issue_number):
    url = f"{GITLAB_URL}/projects/{OWNER}%2F{REPO}/issues/{issue_number}"

    response = requests.put(
        url,
        headers=headers,
        data={"labels": "Stale"},  # comma-separated string
    )

    if response.status_code == 200:
        print(f"Label 'stale' added to issue #{issue_number}")
    else:
        print(
            f"Failed to add label to issue #{issue_number} with "
            f"{response.status_code=}"
        )


# Post a comment to the issue creator
def post_comment(issue_number, creator_login):
    comment_url = f"{GITLAB_URL}/projects/{OWNER}%2F{REPO}/issues/{issue_number}/comments"
    comment_body = (
        f"Hello @{creator_login},\n\n"
        "This issue has not been updated in the last 3 months. Please provide an update on the status of this issue. "
        "If no update is provided within 3 months, we may have to close this issue.\n\n"
        "Thank you!"
    )
    response = requests.post(
        comment_url, headers=headers, json={"body": comment_body}
    )
    if response.status_code == 201:
        print(f"Comment posted on issue #{issue_number}")
    else:
        print(f"Failed to post comment on issue #{issue_number}")


# Main function to process issues
def process_issues():
    issues = get_issues()
    three_months_ago = get_datetime_three_months_ago()

    for issue in tqdm(issues):
        # Skip issues that are closed
        if issue["state"] == "closed":
            continue

        # Check if the issue's last update was older than 3 months
        updated_at = datetime.strptime(
            issue["updated_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        if updated_at < three_months_ago:
            # Add "stale" label
            add_stale_label(issue["iid"])

            # Post a comment to the issue creator
            creator_login = issue["author"]["username"]
            post_comment(issue["iid"], creator_login)


if __name__ == "__main__":
    process_issues()
