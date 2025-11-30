#!/bin/bash
# Start Claude with CLAUDE.md pre-loaded

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_MD="$SCRIPT_DIR/notes/CLAUDE.md"

# Check if CLAUDE.md exists
if [ ! -f "$CLAUDE_MD" ]; then
    echo "Error: CLAUDE.md not found at $CLAUDE_MD"
    exit 1
fi

# Read CLAUDE.md content
CLAUDE_CONTENT=$(cat "$CLAUDE_MD")

# Start Claude with the content as initial context
if [ $# -eq 0 ]; then
    # If no arguments, start in interactive mode with system prompt
    exec claude --append-system-prompt "Please read and remember these FUES development guidelines:

$CLAUDE_CONTENT

You should follow these guidelines throughout our session."
else
    # If arguments provided, pass them along
    claude --append-system-prompt "Please read and remember these FUES development guidelines:

$CLAUDE_CONTENT

You should follow these guidelines throughout our session. Please confirm you have read and understood these guidelines before starting." "$@"
fi