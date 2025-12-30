"""
Verify the entire system structure and imports.
"""
import sys
import asyncio

async def verification():
    print("Verifying imports...")
    try:
        import config.settings
        print("✅ Config imported")
        
        import src.utils.ollama_client
        print("✅ Utils imported")
        
        import src.database.models
        import src.database.connection
        print("✅ Database imported")
        
        import src.master_ai.brain
        import src.master_ai.prompts
        import src.master_ai.context
        print("✅ Master AI imported")
        
        import src.agents.base
        import src.agents.router
        import src.agents.research
        print("✅ Agents imported")
        
        import src.api.main
        import src.api.routes.chat
        print("✅ API imported")
        
        print("\nAll modules imported successfully!")
        return True
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(verification())
    if not success:
        sys.exit(1)
