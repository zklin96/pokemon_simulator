#!/bin/bash
# Start Pokemon Showdown server for VGC AI training
#
# Usage: ./scripts/start_showdown.sh
#
# Prerequisites:
# - Node.js 18+ installed
# - pokemon-showdown cloned in ../pokemon-showdown
# - npm install completed in pokemon-showdown directory

set -e

# Navigate to pokemon-showdown directory (relative to pokemon-vgc-ai)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SHOWDOWN_DIR="$(dirname "$PROJECT_DIR")/pokemon-showdown"

if [ ! -d "$SHOWDOWN_DIR" ]; then
    echo "Error: Pokemon Showdown not found at $SHOWDOWN_DIR"
    echo "Please clone it: git clone https://github.com/smogon/pokemon-showdown.git"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "$SHOWDOWN_DIR/node_modules" ]; then
    echo "Installing npm dependencies..."
    cd "$SHOWDOWN_DIR"
    npm install
fi

# Check if config.js exists
if [ ! -f "$SHOWDOWN_DIR/config/config.js" ]; then
    echo "Creating config.js from example..."
    cp "$SHOWDOWN_DIR/config/config-example.js" "$SHOWDOWN_DIR/config/config.js"
fi

cd "$SHOWDOWN_DIR"

echo "Starting Pokemon Showdown server..."
echo "Server will be available at: ws://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server with no security for local development
node pokemon-showdown start --no-security

