"""
Entry point for the Memory Service.
"""

import uvicorn
from service.config import settings
from service.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=False,
    )