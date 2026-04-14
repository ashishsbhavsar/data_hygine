import os
from motor.motor_asyncio import AsyncIOMotorClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Skipping .env file load.")

# Make connection strings dynamic and configurable via environment variables
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME")

if not MONGO_URI or not DB_NAME:
    # We raise an error instead of using hardcoded defaults to ensure configuration is explicitly managed.
    raise RuntimeError("Critical API Error: MONGO_URI and DB_NAME must be set in the environment or .env file.")

# Collection constants for centralized management
MASTERLIST_COL = os.environ.get("COLLECTION_MASTERLIST", "masterlist")
EXECUTION_INFO_COL = os.environ.get("COLLECTION_EXECUTION_INFO", "ExecutionInfo")
SNAPSHOT_COL = os.environ.get("COLLECTION_SNAPSHOT", "snapshot")
PROCESSOR_DETAILS_COL = os.environ.get("COLLECTION_PROCESSOR_DETAILS", "processor details")

# Global client to maintain a persistent connection pool (FastAPI Best Practice)
_client: AsyncIOMotorClient = None

def get_db():
    """
    Returns the MongoDB database instance. 
    Initializes a new connection pool if one doesn't exist, 
    otherwise reuses the active pool for maximum performance.
    """
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
        print(f"Initialized MongoDB Connection Pool to {MONGO_URI}")
    return _client[DB_NAME]

def close_db():
    """
    Closes the global MongoDB connection pool.
    Useful for shutting down the FastAPI app gracefully.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None
        print("Closed MongoDB Connection Pool")