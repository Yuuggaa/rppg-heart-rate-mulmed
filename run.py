"""
Main launcher for rPPG Heart Rate Monitoring System
Async-based modern architecture with asyncio support
"""

import sys
import asyncio
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run main application
from main import main_async

if __name__ == "__main__":
    # Run async main
    if sys.platform == "win32":
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"\nApplication error: {e}")
        sys.exit(1)
