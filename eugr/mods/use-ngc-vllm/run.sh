#!/bin/bash
set -e

echo "Setting up cluster initialization script..."
cp run-cluster-node.sh $WORKSPACE_DIR/run-cluster-node.sh
chmod +x $WORKSPACE_DIR/run-cluster-node.sh
