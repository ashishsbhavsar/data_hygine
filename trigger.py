import asyncio
import os
import logging
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError, OperationFailure
from dotenv import load_dotenv
 
from database import get_db, close_db, EXECUTION_INFO_COL
from validation import get_validator
 
# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
 
load_dotenv()
 
async def process_document(db, validator, doc):
    """
    Minimalist processing: Validates the document and updates the source collection.
    All other tasks (snapshots, suggestions, monitoring) are handled by other scripts.
    """
    try:
        exec_id = doc.get("benchmarkExecutionID", str(doc.get("_id")))
        logger.info(f"Processing new record: {exec_id}")
 
        # 1. Run Validation using validator.py
        invalid_payload, field_status = await validator.validate_doc(db, doc)
        is_val = len(invalid_payload) == 0
 
        await db[EXECUTION_INFO_COL].update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "invalidPayload": invalid_payload,
                "isValid": is_val,
                "validated": True,
                "fieldStatus": field_status,
                "validatedAt": datetime.now(timezone.utc).isoformat()
            }}
        )
        logger.info(f"Successfully processed specific record: {exec_id} | Status: {'Valid' if is_val else 'Invalid'}")
 
    except Exception as e:
        logger.error(f"Error processing document {doc.get('_id')}: {str(e)}")
 
async def run_polling(db, validator, collection_name, interval=10):
    """
    Fallback loop for Standalone MongoDB instances.
    Periodically polls for unvalidated documents.
    """
    logger.info(f"Starting Polling Mode (Interval: {interval}s) on: {collection_name}")
    collection = db[collection_name]
   
    while True:
        try:
            # Query for documents lacking BOTH fields as requested
            query = {
                "isValid": {"$exists": False},
                "validated": {"$exists": False}
            }
            cursor = collection.find(query)
           
            async for doc in cursor:
                await process_document(db, validator, doc)
           
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Polling error: {str(e)}")
            await asyncio.sleep(interval)
 
async def run_trigger():
    """
    Main loop to watch for changes. Falls back to polling if Change Streams are not supported.
    """
    db = get_db()
    collection = db[EXECUTION_INFO_COL]
 
    logger.info("Initializing Validator...")
    validator = await get_validator()
   
    # Try Change Stream first
    try:
        logger.info(f"Attempting to start Change Stream on: {EXECUTION_INFO_COL}")
        
        # Match only inserts/updates where the document hasn't been validated yet
        pipeline = [
            {"$match": {
                "operationType": {"$in": ["insert", "replace", "update"]},
                "fullDocument.validated": {"$exists": False}
            }}
        ]
        
        async with collection.watch(pipeline, full_document="updateLookup") as stream:
            async for change in stream:
                doc = change.get("fullDocument")
                if doc:
                    await process_document(db, validator, doc)
    except (OperationFailure, PyMongoError) as e:
        err_msg = str(e)
        if isinstance(e, OperationFailure) and e.code == 40573 or "not support change streams" in err_msg.lower():
            logger.warning("Change Streams not supported (Standalone MongoDB detected). Switching to Polling Fallback.")
            await run_polling(db, validator, EXECUTION_INFO_COL)
        else:
            logger.error(f"MongoDB error: {err_msg}")
            logger.info("Restarting trigger in 5 seconds...")
            await asyncio.sleep(5)
            await run_trigger()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        await asyncio.sleep(5)
        await run_trigger()
 
if __name__ == "__main__":
    try:
        asyncio.run(run_trigger())
    except KeyboardInterrupt:
        logger.info("Trigger stopped by user.")
    finally:
        close_db()
