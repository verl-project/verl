#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 YOUR_GITHUB_TOKEN"
    echo "Please provide exactly one input argument for your github token."
    exit 1
fi

# Set your GitHub repository details
OWNER="volcengine"
REPO="verl"
TOKEN=$1

# API URL for workflow runs
API_URL="https://api.github.com/repos/$OWNER/$REPO/actions/runs?status=queued"

# Check required commands
command -v jq >/dev/null 2>&1 || { echo "jq is required but not installed. Aborting."; exit 1; }

# Get queued workflow runs
response=$(curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github.v3+json" "$API_URL")

# Run this for debugging
# echo $response

# Extract run IDs
queued_run_ids=$(echo "$response" | jq -r '.workflow_runs[] | .id')

if [ -z "$queued_run_ids" ]; then
    echo "No queued workflow runs found."
    exit 0
fi

# Cancel each queued run
for run_id in $queued_run_ids; do
    echo "Cancelling run $run_id"
    cancel_url="https://api.github.com/repos/$OWNER/$REPO/actions/runs/$run_id/cancel"
    curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github.v3+json" "$cancel_url"
done

echo "Cancelled all queued workflow runs."
