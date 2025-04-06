#!/usr/bin/env python3
"""
Script to start Freqtrade live trading with the Lorentzian ANN strategy
"""

import sys
import os
import subprocess
from pathlib import Path
import time
import argparse


def start_live_trading(config_file="config_live.json", dry_run=True):
    """Start Freqtrade live trading with the Lorentzian ANN strategy"""

    # Find freqtrade directory
    freqtrade_path = Path.home() / "freqtrade"
    if not freqtrade_path.exists():
        print(f"Freqtrade not found at {freqtrade_path}")
        return False

    # Check if config file exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False

    # Build command
    cmd = [
        f"{freqtrade_path}/freqtrade",
        "trade",
        "--logfile",
        "logs/freqtrade_live.log",
        "--db-url",
        "sqlite:///tradesv3_live.sqlite",
        "--config",
        str(config_path),
    ]

    if not dry_run:
        # WARNING: This will use real money!
        print("LIVE TRADING MODE ENABLED - WILL USE REAL MONEY")
        time.sleep(5)  # Give user time to cancel

        # Update config - this edits the file directly
        import json

        with open(config_path, "r") as f:
            config = json.load(f)

        config["dry_run"] = False

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Updated config file {config_path} - dry_run set to False")
    else:
        print("Starting in dry-run mode (paper trading)")

    # Create logs directory if needed
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Execute command
    print("Starting Freqtrade with command:")
    print(" ".join(cmd))

    try:
        # Start freqtrade and wait for it
        process = subprocess.Popen(cmd)

        print(f"Freqtrade started with PID {process.pid}")
        print("Press Ctrl+C to stop trading")

        # Wait for process to complete or user interrupt
        process.wait()

    except KeyboardInterrupt:
        print("Stopping Freqtrade...")
        process.terminate()
    except Exception as e:
        print(f"Error starting Freqtrade: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start Freqtrade live trading with the Lorentzian ANN strategy"
    )
    parser.add_argument(
        "--config", type=str, default="config_live.json", help="Path to config file"
    )
    parser.add_argument(
        "--live", action="store_true", help="Enable live trading (use real money)"
    )

    args = parser.parse_args()

    start_live_trading(config_file=args.config, dry_run=not args.live)
