"""
Entry point for the Subagent Manager service.
"""

import uvicorn
from .config import config
from .main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False,
    )