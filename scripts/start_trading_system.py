#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse
import signal
import atexit

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Add the root directory to the path
sys.path.append(ROOT_DIR)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading system with dashboard")
    
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                       help="Confidence threshold for signals (default: 0.3)")
    
    parser.add_argument("--combined-threshold", type=float, default=0.25,
                       help="Combined confidence threshold (default: 0.25)")
    
    parser.add_argument("--model-5m", type=str, 
                       default="models/trained/extended/5m_20250402_1614/final_model_5m.pt",
                       help="Path to 5m model")
    
    parser.add_argument("--model-15m", type=str,
                       default="models/trained/extended/15m_20250402_1614/final_model_15m.pt",
                       help="Path to 15m model")
    
    return parser.parse_args()

def start_model_trader(args):
    """Start the combined model trader process"""
    print("Starting Combined Model Trader...")
    
    cmd = [
        "python", 
        os.path.join(SCRIPT_DIR, "combined_model_trader.py"),
        "--live",
        f"--confidence-threshold={args.confidence_threshold}",
        f"--combined-threshold={args.combined_threshold}",
        f"--model-5m={args.model_5m}",
        f"--model-15m={args.model_15m}"
    ]
    
    trader_process = subprocess.Popen(cmd)
    return trader_process

def start_dashboard():
    """Start the dashboard process"""
    print("Starting Trading Dashboard...")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    cmd = [
        "python", 
        os.path.join(SCRIPT_DIR, "dashboard", "trader_dashboard.py")
    ]
    
    dashboard_process = subprocess.Popen(cmd)
    return dashboard_process

def cleanup_processes(processes):
    """Clean up processes on exit"""
    print("\nShutting down trading system...")
    
    for process in processes:
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()

def print_startup_message(dashboard_url):
    """Print startup message with ASCII art"""
    print("\n" + "="*80)
    print("""
    ███████╗ ██████╗ ██╗         ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗ 
    ██╔════╝██╔═══██╗██║         ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
    ███████╗██║   ██║██║            ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
    ╚════██║██║   ██║██║            ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
    ███████║╚██████╔╝███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
    ╚══════╝ ╚═════╝ ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
                                                                                  
      ML-POWERED TRADING SYSTEM
    """)
    print("="*80)
    print(f"\n🚀 Trading model is running! The system is now monitoring SOL price in real-time.")
    print(f"\n🌐 Dashboard is available at: {dashboard_url}")
    print("\n📊 The dashboard will show:")
    print("   - Real-time price chart")
    print("   - Trading signals as they occur")
    print("   - Performance statistics")
    print("\n⚠️  Press Ctrl+C to stop the trading system")
    print("\n" + "="*80 + "\n")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data directory for dashboard if it doesn't exist
    dashboard_data_dir = os.path.join(SCRIPT_DIR, "dashboard", "data")
    os.makedirs(dashboard_data_dir, exist_ok=True)
    
    # Start processes
    processes = []
    
    try:
        # Start dashboard first
        dashboard_process = start_dashboard()
        processes.append(dashboard_process)
        
        # Wait for dashboard to start
        time.sleep(2)
        
        # Start model trader
        trader_process = start_model_trader(args)
        processes.append(trader_process)
        
        # Register cleanup function
        atexit.register(cleanup_processes, processes)
        
        # Print startup message
        dashboard_url = "http://127.0.0.1:5000"
        print_startup_message(dashboard_url)
        
        # Keep running until interrupted
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        
        # Check if any process died
        for i, process in enumerate(processes):
            if process.poll() is not None:
                print(f"Process {i} exited with code {process.returncode}")
        
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_processes(processes)

if __name__ == "__main__":
    main() 