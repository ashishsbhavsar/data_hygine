import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as api_router
from database import get_db, _client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Skipping .env file load.")

# Initialize FastAPI app
app = FastAPI(
    title="Data Hygiene Validation API",
    description="Engine for standardizing and validating ExecutionInfo records.",
    version="1.0.0"
)

# Configure CORS for Dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.on_event("startup")
async def startup_db_client():
    # Initialize DB connection pool
    get_db()

@app.on_event("shutdown")
async def shutdown_db_client():
    # Gracefully close DB connection pool
    global _client
    if _client:
        _client.close()

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    print(f"Starting Data Hygiene API on {host}:{port}")
    # Run the app instance defined in this file
    uvicorn.run(app, host=host, port=port)
    