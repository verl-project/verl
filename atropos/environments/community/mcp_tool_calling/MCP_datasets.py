import random

import pandas as pd
from datasets import Dataset, load_dataset

# Load the MCP servers dataset
try:
    ds = load_dataset("DeepNLP/mcp-servers")
    train_ds = ds["train"]
    df = train_ds.to_pandas()
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Create dummy data for demonstration
    df = pd.DataFrame(
        {
            "content_name": ["AgentRPC", "Git", "Actors MCP Server"],
            "description": [
                "Toggle menu Node.js Go Python",
                "Tools to read, search, and manipulate code",
                "Use 3,000+ pre-built cloud tools",
            ],
            "subfield": ["MCP SERVER", "MCP SERVER", "MCP SERVER"],
            "field": ["AI AGENT", "AI AGENT", "AI AGENT"],
        }
    )

# Define template prompts based on server types
general_templates = [
    "I need to {action} {object}. Can you help me?",
    "How can I {action} using {tool}?",
    "I'm trying to {action}. What's the best way to do this?",
    "Can you assist me with {action}?",
    "What's the process for {action} with {tool}?",
]

# Define specific actions based on server type
server_specific_actions = {
    "AgentRPC": [
        "call a remote procedure",
        "establish a connection with a remote server",
        "execute a function on another machine",
        "implement RPC in my application",
        "set up agent communication",
    ],
    "Git": [
        "merge my branch",
        "resolve a merge conflict",
        "check the commit history",
        "revert to a previous commit",
        "create a new branch",
    ],
    "AWS KB Retrieval": [
        "find information about AWS services",
        "query the AWS knowledge base",
        "lookup AWS documentation",
        "get help with AWS configuration",
        "understand AWS pricing",
    ],
    "Anki": [
        "create flashcards for studying",
        "improve my spaced repetition system",
        "organize my study notes",
        "set up a memory training system",
        "track my learning progress",
    ],
    "ArangoDB": [
        "query a graph database",
        "store connected data",
        "implement a multi-model database",
        "perform graph traversals",
        "optimize my database queries",
    ],
}

# Default actions for any server not specifically defined
default_actions = [
    "connect to a server",
    "use an API",
    "access external data",
    "integrate with a tool",
    "automate a process",
]


def generate_prompt_for_server(server_name, description):
    """Generate a contextually appropriate prompt for a given server"""

    # Extract potential actions from description if available
    actions = []
    if description and isinstance(description, str):
        words = description.lower().split()
        verbs = [
            "use",
            "toggle",
            "enable",
            "explore",
            "search",
            "read",
            "process",
            "connect",
            "build",
        ]
        for verb in verbs:
            if verb in words:
                idx = words.index(verb)
                if idx < len(words) - 1:
                    actions.append(f"{verb} {words[idx+1]}")

    # If we couldn't extract meaningful actions, use predefined ones
    if not actions:
        if server_name in server_specific_actions:
            actions = server_specific_actions[server_name]
        else:
            actions = default_actions

    # Get a random action and template
    action = random.choice(actions)
    template = random.choice(general_templates)

    # Fill in the template
    prompt = template.format(action=action, object=server_name, tool=server_name)

    return prompt


# Generate prompts for each entry in the dataset
prompts = []
for idx, row in df.iterrows():
    server_name = row["content_name"]
    description = row.get("description", "")
    prompt = generate_prompt_for_server(server_name, description)
    prompts.append(prompt)

# Add the prompts as a new column
df["prompt"] = prompts

# Preview the results
print("\nDataset with Added Prompts:")
print(df[["content_name", "prompt"]].head())

# Save the modified dataset
modified_ds = Dataset.from_pandas(df)
modified_ds.save_to_disk("./modified_mcp_dataset")

print("\nModified dataset saved to ./modified_mcp_dataset")
print("You can load it in your RL environment with:")
print("from datasets import load_from_disk")
print("custom_dataset = load_from_disk('./modified_mcp_dataset')")

# To demonstrate how this would look in your RL environment
print("\nExample usage in RL environment:")
print("=" * 60)
for idx, row in df.head(3).iterrows():
    print(f"User Query: {row['prompt']}")
    print(f"Available Tool: {row['content_name']}")
    print(f"Tool Type: {row['subfield']}")
    print("-" * 40)
