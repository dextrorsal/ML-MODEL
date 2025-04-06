#!/usr/bin/env python3
"""
Configuration Migration Tool

This script helps migrate configuration files from the old src/comparison location
to the new model-evaluation directory. It copies all JSON configuration files
and updates any paths inside them to point to the new location.

Usage:
    python scripts/migrate_configs.py

The script will automatically:
1. Create the model-evaluation/config_samples directory if it doesn't exist
2. Copy all JSON files from config_samples/ and src/comparison/config_samples/
3. Update any paths in the configs to use the new location
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths
OLD_CONFIG_PATHS = [
    "config_samples",
    "src/comparison/config_samples",
]
NEW_CONFIG_PATH = "model-evaluation/config_samples"


def migrate_configs():
    """Migrate configuration files to the new location"""
    print(f"Starting migration of configuration files to {NEW_CONFIG_PATH}")

    # Create the new directory if it doesn't exist
    os.makedirs(NEW_CONFIG_PATH, exist_ok=True)
    print(f"Created directory: {NEW_CONFIG_PATH}")

    # Track the number of files migrated
    migrated_files = 0

    # Process each old config directory
    for old_path in OLD_CONFIG_PATHS:
        if not os.path.exists(old_path):
            print(f"Skipping non-existent path: {old_path}")
            continue

        print(f"Processing configs from: {old_path}")

        # Get all JSON files
        json_files = [f for f in os.listdir(old_path) if f.endswith(".json")]

        for filename in json_files:
            old_file_path = os.path.join(old_path, filename)
            new_file_path = os.path.join(NEW_CONFIG_PATH, filename)

            # Read the config file
            try:
                with open(old_file_path, "r") as f:
                    config = json.load(f)

                # Update any paths in the config
                if "output_dir" in config:
                    # Update output directory paths to use model-evaluation prefix
                    if config["output_dir"].startswith("src/comparison/"):
                        config["output_dir"] = config["output_dir"].replace(
                            "src/comparison/", "model-evaluation/"
                        )
                    elif not config["output_dir"].startswith("model-evaluation/"):
                        # If it's just a relative path like 'results'
                        config["output_dir"] = os.path.join(
                            "model-evaluation", config["output_dir"]
                        )

                # Write the updated config
                with open(new_file_path, "w") as f:
                    json.dump(config, f, indent=4)

                print(f"Migrated: {old_file_path} → {new_file_path}")
                migrated_files += 1

            except Exception as e:
                print(f"Error processing {old_file_path}: {str(e)}")

    print(
        f"\nMigration completed. {migrated_files} files migrated to {NEW_CONFIG_PATH}"
    )

    # Add instructions for updating code
    print("\nNext steps:")
    print("1. Update your scripts to use the new path: model-evaluation/")
    print(
        "2. Update any absolute imports to use 'model_evaluation' instead of 'src.comparison'"
    )


if __name__ == "__main__":
    migrate_configs()
