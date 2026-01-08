"""
Command-line interface for King AI v2.
Allows direct interaction with the Master AI brain via the terminal.

Run with: py -3 cli.py
"""
import asyncio
import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.master_ai.brain import MasterAI
from src.database.connection import init_db
from config.settings import settings
import traceback

async def main():
    print("\n🤴 King AI v2 - Autonomous Business Empire")
    print("=" * 50)
    print("Initializing system...")
    
    # Initialize database tables
    try:
        await init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        # traceback.print_exc() # detailed logging if needed
        return

    # Initialize Master AI
    try:
        ai = MasterAI()
        print(f"✓ Master AI instantiated (Risk Profile: {settings.risk_profile})")
    except Exception as e:
        print(f"✗ Master AI initialization failed: {e}")
        traceback.print_exc()
        return
    
    print("\nCommands:")
    print("  'quit'  - Exit the program")
    print("  'auto'  - Toggle autonomous mode")
    print("  'clear' - Clear screen")
    print("  Any other text will be sent to King AI.\n")
    
    print(f"Autonomous Mode: {'ON' if ai.autonomous_mode else 'OFF'}")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            if user_input.lower() == 'auto':
                ai.autonomous_mode = not ai.autonomous_mode
                status = "ON" if ai.autonomous_mode else "OFF"
                print(f"⚙️ Autonomous mode set to: {status}")
                if ai.autonomous_mode:
                     print("   (Note: The autonomous loop runs in the background in the API, not strictly here in the CLI unless threaded. For CLI, this mostly sets the flag.)")
                continue
            
            # Show processing indicator
            print("⏳ Thinking...", end="\r", flush=True)
            
            # Process input
            result = await ai.process_input(user_input)
            
            # Clear processing indicator
            print(" " * 20, end="\r", flush=True)
            
            # Print response
            # Print response
            print(f"👑 King AI: {result.response}")
            
            # Display actions if any
            if result.actions_taken:
                print("\n[Actions Taken]:")
                for action in result.actions_taken:
                    print(f"  - {action.step_name} ({action.agent}) -> {action.success}")
            
            # Display pending approvals if any
            if result.pending_approvals:
                print("\n[⚠️ Pending Approvals]:")
                for task in result.pending_approvals:
                    # task might be a dict or object depending on how it's populated
                    # In brain.py schema pending_approvals is List[Dict]
                    name = task.get('name', 'Unknown') if isinstance(task, dict) else getattr(task, 'name', 'Unknown')
                    desc = task.get('description', 'No description') if isinstance(task, dict) else getattr(task, 'description', 'No description')
                    print(f"  - {name}: {desc}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    print("\nGoodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
