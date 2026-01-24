"""
Entry point for the Subagent Manager service.
"""

import uvicorn
from service.config import config
from service.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=False,
    )