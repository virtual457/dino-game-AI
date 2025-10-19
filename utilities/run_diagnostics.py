"""
Complete Diagnostic Workflow

Runs all debugging tools in sequence and generates a comprehensive report
"""

import os
import subprocess
import time


def run_command(command, description):
    """Run a command and show output"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    print(f"Command: {command}\n")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("COMPLETE DQN DIAGNOSTIC WORKFLOW")
    print("="*70)
    print("\nThis will run all diagnostic tools in sequence.")
    print("Each tool will generate reports and visualizations.")
    print("\nPress Ctrl+C at any time to stop.\n")
    
    input("Press Enter to start...")
    
    # Step 1: System diagnostics
    print("\n\n")
    print("📋 STEP 1: System Health Check")
    print("-" * 70)
    run_command("python diagnostics.py", "System Diagnostics")
    input("\nPress Enter to continue to Step 2...")
    
    # Step 2: Training analysis
    print("\n\n")
    print("🔍 STEP 2: Training Analysis")
    print("-" * 70)
    print("\nThis will analyze your trained model and generate visualizations.")
    print("Check the output for Q-value distributions and action preferences.")
    run_command("python analyze_training.py", "Training Analysis")
    input("\nPress Enter to continue to Step 3...")
    
    # Step 3: Quick performance test
    print("\n\n")
    print("🎮 STEP 3: Performance Test")
    print("-" * 70)
    print("\nThis will run the agent for 5 episodes and measure performance.")
    print("You'll see how many frames it survives on average.")
    
    response = input("\nRun performance test? (y/n): ")
    if response.lower() == 'y':
        run_command("python quick_test.py", "Performance Test")
    else:
        print("⏭️  Skipped")
    
    # Step 4: Live debugging (optional)
    print("\n\n")
    print("👁️ STEP 4: Live Visual Debugging (Optional)")
    print("-" * 70)
    print("\nThis opens a live dashboard showing what the network sees.")
    print("⚠️  This will open Chrome and a matplotlib window.")
    print("Press Ctrl+C in the dashboard window to stop.")
    
    response = input("\nRun live debugging? (y/n): ")
    if response.lower() == 'y':
        print("\n🚀 Starting debug dashboard...")
        print("Press Ctrl+C to stop when done watching.\n")
        run_command("python debug_dashboard.py", "Live Debug Dashboard")
    else:
        print("⏭️  Skipped")
    
    # Summary
    print("\n\n")
    print("="*70)
    print("DIAGNOSTIC WORKFLOW COMPLETE")
    print("="*70)
    print("\n📊 Generated Files:")
    
    files_to_check = [
        ("checkpoints/q_value_analysis.png", "Q-value distribution analysis"),
        ("checkpoints/training_progress.png", "Training progress plot"),
    ]
    
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            print(f"  ✅ {filepath}")
            print(f"     {description}")
        else:
            print(f"  ❌ {filepath} (not generated)")
    
    print("\n📖 Next Steps:")
    print("  1. Review the analysis outputs above")
    print("  2. Read DEBUG_GUIDE.md for solutions")
    print("  3. Apply fixes using quick_fixes.py")
    print("  4. Continue training with main.py")
    
    print("\n💡 Quick Fix Commands:")
    print("  python quick_fixes.py reset_epsilon      # More exploration")
    print("  python quick_fixes.py increase_rewards   # Value survival more")
    print("  python quick_fixes.py reset_training     # Start fresh")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Workflow stopped by user")
