import logging
from rtc_server import RTCServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
server = RTCServer()
print(server.app)
app = server.app

def main():
    import uvicorn
    logger.info("Starting WebRTC Media Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

if __name__ == "__main__":
    main()