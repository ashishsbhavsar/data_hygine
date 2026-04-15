import asyncio
from pymongo import UpdateOne, ReplaceOne, DeleteOne
from database import get_db, close_db
from validation import get_validator, build_mappings
from utils import get_nested_value
import uuid
from datetime import datetime, timezone
import collections
 
async def main():
    db = get_db()
   
    print("Fetching dynamic validation rules from masterlist_db...")
    validator = await get_validator()
    mappings = await build_mappings()
   
    print(f"Discovered {len(mappings)} validation parameters: {list(mappings.keys())}")
   
    # Optimize performance with explicit sorting
    cursor = db['Executioninfo'].find({}).sort("_id", 1)
   
    print("Starting offline validation batch processing using advanced cross-field relations...")
    print("Status preservation is now real-time to avoid race conditions.")
   
    batch_size = 2000
    updates = []
    snapshot_items = []
    processed = 0
   
    total_valid = 0
    total_invalid = 0
   
    async for doc in cursor:
        invalid_payload, field_status = await validator.validate_doc(db, doc)
        is_val = len(invalid_payload) == 0
        exec_id = doc.get("benchmarkExecutionID", str(doc.get("_id")))
        
        # 1. Update Executioninfo basic status
        updates.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {
                "invalidPayload": invalid_payload,
                "isValid": is_val
            }}
        ))
        
        if is_val:
            total_valid += 1
        else:
            total_invalid += 1
            
        # 2. Handle Snapshot Logic (Transition or Update)
        latest_snap = await db['snapshot'].find_one(
            {"execution_id": exec_id},
            {"data": {"$slice": 1}, "snapshot_id": 1}
        )
        
        if is_val:
            # Record is VALID - Only update if a snapshot ALREADY exists (History Preservation)
            if latest_snap and latest_snap.get("data"):
                prev_data = latest_snap["data"][0]
                
                # If it's already ACCEPTED, REJECTED, or ON HOLD, don't revert it
                current_status = str(prev_data.get("standardization_status", "")).upper()
                if current_status not in ["ACCEPTED", "REJECTED", "ON HOLD"]:
                    # Auto-transition to ACCEPTED since it's now perfectly valid
                    snapshot_items.append(UpdateOne(
                        {"execution_id": exec_id},
                        {"$set": {
                            "data.0.standardization_status": "ACCEPTED",
                            "data.0.lastModifiedOn": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        }}
                    ))
        else:
            # Record is INVALID - Create or Update Snapshot with fuzzy suggestions
            prev_data = {}
            snap_id = str(uuid.uuid4())
            if latest_snap:
                snap_id = latest_snap.get("snapshot_id", snap_id)
                if latest_snap.get("data"):
                    prev_data = latest_snap["data"][0]
            
            # Normalize status to uppercase, defaults to PENDING
            raw_status = prev_data.get("standardization_status", "PENDING")
            status_val = raw_status.upper() if raw_status else "PENDING"
            
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
                
                # Get record-level suggestions using the new 'Mega-String' logic
                record_suggestions = validator.get_record_level_suggestions(field, val, actual_meta_vals)
                
                # Build formatted suggestions for the dashboard (Suggestion Status = PENDING initially)
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
                    "history": prev_data.get("history", {
                        "updatedOn": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "updatedBy": "xxx@amd.com",
                        "from": [], "to": [], "valueField": [], "source": []
                    })
                }]
            }
            
            snapshot_items.append(ReplaceOne(
                {"execution_id": exec_id},
                snapshot_doc,
                upsert=True
            ))
       
        if len(updates) >= batch_size:
            await db['Executioninfo'].bulk_write(updates, ordered=False)
            if snapshot_items:
                await db['snapshot'].bulk_write(snapshot_items, ordered=False)
                snapshot_items.clear()
            processed += len(updates)
            print(f"Processed {processed} records... (Valid: {total_valid}, Invalid: {total_invalid})")
            updates.clear()
           
    if updates:
        await db['Executioninfo'].bulk_write(updates, ordered=False)
        processed += len(updates)
    if snapshot_items:
        await db['snapshot'].bulk_write(snapshot_items, ordered=False)
               
    print(f"\nOffline Validation Complete! Total: {processed}")
    close_db()
 
 
if __name__ == "__main__":
    asyncio.run(main())
    