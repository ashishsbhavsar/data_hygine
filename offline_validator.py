import asyncio
from concurrent.futures import ThreadPoolExecutor
from pymongo import UpdateOne, ReplaceOne
from database import get_db, close_db, DB_NAME, MONGO_URI, EXECUTION_INFO_COL, SNAPSHOT_COL
from validation import get_validator
import uuid
from datetime import datetime, timezone
import os

# Helper for threading
def process_record_batch(validator, docs, snapshot_cache):
    """
    Processes a batch of documents synchronously.
    Returns list of Executioninfo updates and list of Snapshot updates.
    """
    results = []
    for doc in docs:
        invalid_payload, field_status = validator.validate_doc_sync(doc)
        is_val = len(invalid_payload) == 0
        exec_id = doc.get("benchmarkExecutionID", str(doc.get("_id")))
        
        # 1. Executioninfo Update
        ei_update = UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {
                "invalidPayload": invalid_payload,
                "isValid": is_val
            }}
        )
        
        # 2. Snapshot Update
        snap_update = None
        # Use simple cache lookup instead of DB call
        prev_snap_status = snapshot_cache.get(exec_id, {}).get("status")
        prev_snap_history = snapshot_cache.get(exec_id, {}).get("history")
        prev_snap_id = snapshot_cache.get(exec_id, {}).get("snapshot_id")

        if is_val:
            if prev_snap_status:
                current_status = str(prev_snap_status).upper()
                if current_status not in ["ACCEPTED", "REJECTED", "ON HOLD"]:
                    snap_update = UpdateOne(
                        {"execution_id": exec_id},
                        {"$set": {
                            "data.0.standardization_status": "ACCEPTED",
                            "data.0.lastModifiedOn": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        }}
                    )
        else:
            snap_id = prev_snap_id or str(uuid.uuid4())
            status_val = str(prev_snap_status).upper() if prev_snap_status else "PENDING"
            
            clean_meta = []
            for p in invalid_payload:
                field = p.get("field")
                val = p.get("value")
                p_clean = {
                    "field": field,
                    "currentStatus": "invalid",
                    "value": val,
                    "validation_status": p.get("validation_status", "invalid"),
                    "mapping": p.get("mapping", "")
                }
                actual_meta_vals = {m["name"]: m.get("value", "") for m in p.get("metadata", []) if m.get("name")}
                record_suggestions = validator.get_record_level_suggestions(field, val, actual_meta_vals)
                
                primary_comparing = []
                for i, rec_sug in enumerate(record_suggestions, 1):
                    primary_comparing.append({
                        f"suggestion{i}": rec_sug["primary_value"],
                        f"score{i}": rec_sug["score"],
                        "status": "PENDING",
                        "_id": rec_sug["_id"]
                    })
                p_clean["comparingData"] = primary_comparing
                
                meta_list = []
                for m in p.get("metadata", []):
                    m_clean = dict(m)
                    m_comparing = []
                    m_name = m_clean.get("name")
                    for i, rec_sug in enumerate(record_suggestions, 1):
                        m_comparing.append({
                            f"suggestion{i}": rec_sug["metadata"].get(m_name, ""),
                            f"score{i}": rec_sug["score"],
                            "status": "PENDING",
                            "_id": rec_sug["_id"]
                        })
                    m_clean["comparingData"] = m_comparing
                    meta_list.append(m_clean)
                p_clean["metadata"] = meta_list
                clean_meta.append(p_clean)
            
            snapshot_doc = {
                "snapshot_id": snap_id,
                "execution_id": exec_id,
                "benchmark_type": doc.get("benchmarkType"),
                "benchmark_category": doc.get("benchmarkCategory"),
                "data": [{
                    "invalidFields": sorted(list(set(
                        [p.get("field") for p in clean_meta if p.get("field") and p.get("validation_status") == "invalid"] +
                        [m.get("name") for p in clean_meta for m in p.get("metadata", []) if m.get("validation_status") == "invalid"]
                    ))),
                    "invalidValues": clean_meta,
                    "standardization_status": status_val,
                    "history": prev_snap_history or {
                        "updatedOn": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "updatedBy": "xxx@amd.com",
                        "from": [], "to": [], "valueField": [], "source": []
                    }
                }]
            }
            snap_update = ReplaceOne({"execution_id": exec_id}, snapshot_doc, upsert=True)
            
        results.append((ei_update, snap_update))
    return results

async def main():
    db = get_db()
    
    print("Pre-loading validator and processor cache...", flush=True)
    validator = await get_validator()
    
    print("Pre-loading active snapshots for fast lookup...", flush=True)
    snapshot_cache = {}
    async for snap in db[SNAPSHOT_COL].find({}, {"execution_id": 1, "snapshot_id": 1, "data.standardization_status": 1, "data.history": 1}):
        if snap.get("execution_id"):
            snapshot_cache[snap["execution_id"]] = {
                "status": snap.get("data", [{}])[0].get("standardization_status"),
                "history": snap.get("data", [{}])[0].get("history"),
                "snapshot_id": snap.get("snapshot_id")
            }
    print(f"Loaded {len(snapshot_cache)} snapshots into memory.", flush=True)

    print("Fetching records from Executioninfo...", flush=True)
    cursor = db[EXECUTION_INFO_COL].find({}).sort("_id", 1)
    
    batch_size = 500
    chunk_size = 500
    doc_chunk = []
    processed = 0
    
    print(f"Starting parallel validation using {os.cpu_count() or 4} threads...", flush=True)
    start_time = datetime.now()
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        
        async for doc in cursor:
            doc_chunk.append(doc)
            if len(doc_chunk) >= chunk_size:
                # Submit chunk to pool
                tasks.append(loop.run_in_executor(executor, process_record_batch, validator, doc_chunk, snapshot_cache))
                doc_chunk = []
                
                # Print more frequently: after every N tasks completed
                if len(tasks) >= (os.cpu_count() or 1):
                    completed_batches = await asyncio.gather(*tasks)
                    await save_batches(db, completed_batches)
                    processed += sum(len(b) for b in completed_batches)
                    print(f"Processed {processed} records... ({datetime.now() - start_time})", flush=True)
                    tasks = []

        # Process final chunk
        if doc_chunk:
            tasks.append(loop.run_in_executor(executor, process_record_batch, validator, doc_chunk, snapshot_cache))
        
        if tasks:
            completed_batches = await asyncio.gather(*tasks)
            await save_batches(db, completed_batches)
            processed += (len(completed_batches) - 1) * chunk_size + len(doc_chunk)
               
    print(f"\nOffline Validation Complete! Total: {processed} in {datetime.now() - start_time}", flush=True)
    close_db()

async def save_batches(db, completed_batches):
    """Saves all updates from multiple batches using bulk_write."""
    ei_updates = []
    snap_updates = []
    
    for batch in completed_batches:
        for ei, snap in batch:
            ei_updates.append(ei)
            if snap:
                snap_updates.append(snap)
                
    if ei_updates:
        await db[EXECUTION_INFO_COL].bulk_write(ei_updates, ordered=False)
    if snap_updates:
        await db[SNAPSHOT_COL].bulk_write(snap_updates, ordered=False)

if __name__ == "__main__":
    asyncio.run(main())

    