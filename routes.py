from fastapi import APIRouter, Query, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import rapidfuzz
from rapidfuzz import process
from datetime import datetime
import uuid
from database import get_db, MASTERLIST_COL, EXECUTION_INFO_COL, SNAPSHOT_COL
from validation import build_mappings, get_validator
from utils import get_nested_value, get_metadata_schema

router = APIRouter()



class ApproveSuggestionRequest(BaseModel):
    execution_id: str
    field_name: str
    accepted_value: str
    currentStatus: str = "Accepted"
    coreCount: Optional[str] = None

class RejectRecordRequest(BaseModel):
    execution_id: str
    currentStatus: str = "L0 Data"

class DraftRecordRequest(BaseModel):
    value: str
    currentStatus: str = "ON HOLD"
    id: Optional[str] = None
    execution_id: Optional[str] = None
    family: Optional[str] = ""
    corecount: Optional[str] = ""
    cpumodel: Optional[str] = ""
    cloudprovider: Optional[str] = ""
    benchmarktype: Optional[str] = ""

async def _check_duplicate(db, type_name, value, metadata_dict=None):
    """Checks if a record with the same type, value, and precise metadata already exists in the masterlist."""
    query = {"type": type_name, "data.value": value}
    if metadata_dict:
        for k, v in metadata_dict.items():
            if v is not None and v != "":
                query[f"data.metadata.{k}"] = str(v)
    return await db[MASTERLIST_COL].find_one(query)

def _build_base_ml_doc(type_name, data_content, updated_by: str = ""):
    """Builds the common base structure for a 'In Review' masterlist document with data before history."""
    now = datetime.utcnow()
    return {
        "`_id`": str(uuid.uuid4()),
        "type": type_name,
        "status": "Draft",
        "data": data_content,
        "history": {
            "updatedOn": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "updatedBy": updated_by or "xxx@amd.com",
            "to": None,
            "valueField": None
        }
    }

async def resolve_fuzzy_benchmarks(benchmarkType: str = None, benchmarkCategory: str = None):
    """
    Fuzzy match across masterlist published records to find the standard BenchmarkType/Category.
    Returns resolved values based on high-confidence fuzzy matching scores (>70).
    """
    db = get_db()
    cursor = db[MASTERLIST_COL].find({"type": "Benchmark", "status": "Published"})
    benchmarks = await cursor.to_list(None)
    
    resolved = {}
    if benchmarkType:
        # Match against metadata.BenchmarkType or data.value (category) or the type itself
        all_types = list(set([b["data"].get("metadata", {}).get("BenchmarkType") for b in benchmarks if b["data"].get("metadata", {}).get("BenchmarkType")]))
        if all_types:
            best = rapidfuzz.process.extractOne(benchmarkType, all_types, score_cutoff=70)
            if best:
                resolved["benchmarkType"] = best[0]
                resolved["benchmarkType_is_fuzzy"] = True
                
    if benchmarkCategory:
        # Match against primary data.value (which is the grouping category for Benchmark type)
        all_categories = list(set([b["data"].get("value") for b in benchmarks if b["data"].get("value")]))
        if all_categories:
            best = rapidfuzz.process.extractOne(benchmarkCategory, all_categories, score_cutoff=70)
            if best:
                resolved["benchmarkCategory"] = best[0]
                resolved["benchmarkCategory_is_fuzzy"] = True
                
    return resolved

async def get_masterlist_mappings(field_type: str):
    """
    Retrieves the mapping structure for a given masterlist type (e.g., CPUModel, instanceType).
    Returns mapping paths for the primary value and any nested metadata fields.
    """
    db = get_db()
    doc = await db[MASTERLIST_COL].find_one({"type": field_type, "status": "Published"})
    if not doc:
        return {}
    
    data = doc.get("data", {})
    metadata = data.get("metadata", {})
    
    # Correctly parse mapping_X fields from the metadata object
    meta_mappings = {k.replace("mapping_", ""): v for k, v in metadata.items() if k.startswith("mapping_")}
    
    return {
        "mapping": data.get("mapping"),
        "metadata_mappings": meta_mappings
    }

# Lookup table for field names per record type
_DRAFT_FIELDS_MAP = {
    "cpumodel": ["value", "family", "corecount"],
    "instancetype": ["value", "cpumodel", "cloudprovider", "family", "corecount"],
    "benchmark": ["value", "benchmarktype"]
}

# Global in-memory cache for expensive dashboard queries
_report_cache = {
    "total_invalid": {"value": None, "updated_at": 0},
    "counts_metrics": {"value": None, "updated_at": 0},
    "summary_counts": {"value": None, "updated_at": 0}
}
CACHE_TTL = 300  # 5 minutes

@router.get("/invalid-records")
async def get_invalid_records(
    page: int = Query(1, ge=1), 
    size: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    Validates the ExecutionInfo collection asynchronously with pagination.
    Supports a single 'search' parameter for execution_id or fuzzy benchmark types/categories.
    """
    db = get_db()
    
    status_filter = status.upper() if status else "PENDING"
    
    if status_filter == "PENDING":
        match_query = {"isValid": False}
        res_status = "PENDING"
    elif status_filter == "ACCEPTED":
        match_query = {"isValid": True}
        res_status = "ACCEPTED"
    elif status_filter == "REJECTED":
        match_query = {"isValid": False}
        res_status = "REJECTED"
    else:
        return {"status": "success", "total_invalid_records": 0, "page": page, "size": size, "returned_records": 0, "data": []}
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        # Fuzzy resolution for both Type and Category
        resolved = await resolve_fuzzy_benchmarks(benchmarkType=search, benchmarkCategory=search)
        or_filters = [{"benchmarkExecutionID": search_regex}]
        
        if "benchmarkType" in resolved:
            if resolved.get("benchmarkType_is_fuzzy"):
                or_filters.append({"benchmarkType": resolved["benchmarkType"]})
            else:
                or_filters.append({"benchmarkType": search_regex})
        
        if "benchmarkCategory" in resolved:
            if resolved.get("benchmarkCategory_is_fuzzy"):
                or_filters.append({"benchmarkCategory": resolved["benchmarkCategory"]})
            else:
                or_filters.append({"benchmarkCategory": search_regex})
        
        # Always fallback to general regex on all three if no fuzzy matches found yet
        if len(or_filters) == 1:
            or_filters.extend([
                {"benchmarkType": search_regex},
                {"benchmarkCategory": search_regex}
            ])
            
        match_query["$or"] = or_filters

    skip_count = (page - 1) * size
    
    pipeline = [
        {"$match": match_query},
        {"$group": {
            "_id": "$benchmarkExecutionID",
            "invalidPayload": {"$first": "$invalidPayload"},
            "benchmarkType": {"$first": "$benchmarkType"},
            "benchmarkCategory": {"$first": "$benchmarkCategory"}
        }},
        {"$sort": {"_id": -1}},
        {"$skip": skip_count},
        {"$limit": size}
    ]
    
    invalid_records = []
    async for doc in db[EXECUTION_INFO_COL].aggregate(pipeline):
        invalid_payloads = doc.get("invalidPayload", [])
        
        # Traverse top level payload and metadata to collect ALL invalid fields dynamically
        invalid_fields_set = set()
        for p in invalid_payloads:
            if p.get("validation_status") == "invalid" and p.get("field"):
                invalid_fields_set.add(p["field"])
            for m in p.get("metadata", []):
                if m.get("validation_status") == "invalid" and m.get("name"):
                    invalid_fields_set.add(m["name"])
        invalid_fields = sorted(list(invalid_fields_set))
        
        record = {
            "ExecutionId": doc["_id"],
            "Status": res_status,
            "BenchmarkType": doc.get("benchmarkType"),
            "BenchmarkCategory": doc.get("benchmarkCategory")
        }
        
        if status_filter == "PENDING":
            record["InvalidFields"] = invalid_fields
            record["Details"] = invalid_payloads
            
        invalid_records.append(record)

    # Count logic (Filtered if search parameter provided or status is not PENDING)
    if search or status_filter != "PENDING":
        count_pipeline = [
            {"$match": match_query},
            {"$group": {"_id": "$benchmarkExecutionID"}},
            {"$count": "total"}
        ]
        cursor = db[EXECUTION_INFO_COL].aggregate(count_pipeline)
        result = await cursor.to_list(length=1)
        total_invalid_records = result[0]['total'] if result else 0
    else:
        global _report_cache
        if _report_cache["total_invalid"]["value"] is None or (time.time() - _report_cache["total_invalid"]["updated_at"]) > CACHE_TTL:
            count_pipeline = [
                {"$match": {"isValid": False}},
                {"$group": {"_id": "$benchmarkExecutionID"}},
                {"$count": "total"}
            ]
            cursor = db[EXECUTION_INFO_COL].aggregate(count_pipeline)
            result = await cursor.to_list(length=1)
            _report_cache["total_invalid"]["value"] = result[0]['total'] if result else 0
            _report_cache["total_invalid"]["updated_at"] = time.time()
        total_invalid_records = _report_cache["total_invalid"]["value"]
    
    return {
        "status": "success",
        "total_invalid_records": total_invalid_records,
        "page": page,
        "size": size,
        "returned_records": len(invalid_records),
        "data": invalid_records
    }

@router.get("/invalid-summary/counts")
async def get_invalid_summary_counts():
    """
    Returns the count of invalid records grouped by age (based on their snapshot timestamp).
    Uses caching for performance.
    green: < 3 days old
    yellow: 3-6 days old
    red: > 6 days old
    """
    global _report_cache
    if _report_cache["summary_counts"]["value"] is not None and (time.time() - _report_cache["summary_counts"]["updated_at"]) <= CACHE_TTL:
        return _report_cache["summary_counts"]["value"]

    db = get_db()
    counts = {"red": 0, "yellow": 0, "green": 0}
    now = datetime.utcnow()
    
    # Query all PENDING snapshots (Case-insensitive) to evaluate their age
    cursor = db[SNAPSHOT_COL].find(
        {"data.0.standardization_status": {"$regex": "^PENDING$", "$options": "i"}},
        {"data": {"$slice": 1}}
    )
    
    async for doc in cursor:
        updated_on_str = doc.get("data", [{}])[0].get("history", {}).get("updatedOn")
        if not updated_on_str:
            continue
            
        try:
            # Parse the timestamp e.g. '2026-03-27T15:18:14.632228Z'
            dt = datetime.strptime(updated_on_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            delta_days = (now - dt).total_seconds() / 86400.0
            
            if delta_days < 3:
                counts["green"] += 1
            elif delta_days <= 6:
                counts["yellow"] += 1
            else:
                counts["red"] += 1
        except Exception:
            # Fallback for unparseable dates
            counts["yellow"] += 1
    
    _report_cache["summary_counts"]["value"] = counts
    _report_cache["summary_counts"]["updated_at"] = time.time()
    return counts

@router.get("/invalid-summary")
async def get_invalid_summary(
    search: Optional[str] = Query(None, description="Search by Execution ID, Benchmark Type, or Category"),
    status: Optional[str] = Query(None, description="Filter by status: PENDING, REJECTED, ACCEPTED, 'On Hold'. Defaults to ALL if not provided."),
    page: int = Query(1, ge=1), 
    size: int = Query(50, ge=1, le=500)
):
    """
    Returns strictly the Execution_id and the names of the specific fields that are invalid.
    Optimized: Now queries the 'snapshot' collection directly for sub-second performance.
    """
    db = get_db()
    
    import re
    # 1. Primary Filter on Snapshot collection
    if status:
        status_list = [s.strip() for s in status.split(",") if s.strip()]
        regex_pattern = "^(" + "|".join(re.escape(s) for s in status_list) + ")$"
        status_filter = ",".join(status_list).upper()
        match_query = {"data.standardization_status": {"$regex": regex_pattern, "$options": "i"}}
    else:
        # Default: Include all high-level statuses
        status_filter = "ALL"
        match_query = {"data.standardization_status": {"$regex": "^(PENDING|REJECTED|ON HOLD|ACCEPTED)$", "$options": "i"}}
    
    print(f"API Executing Match Query: {match_query}")
    
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        # Fuzzy match on Type/Category
        resolved = await resolve_fuzzy_benchmarks(benchmarkType=search, benchmarkCategory=search)
        or_filters = [{"execution_id": search_regex}]
        
        if "benchmarkType" in resolved:
            if resolved.get("benchmarkType_is_fuzzy"):
                or_filters.append({"benchmark_type": resolved["benchmarkType"]})
            else:
                or_filters.append({"benchmark_type": search_regex})
        
        if "benchmarkCategory" in resolved:
            if resolved.get("benchmarkCategory_is_fuzzy"):
                or_filters.append({"benchmark_category": resolved["benchmarkCategory"]})
            else:
                or_filters.append({"benchmark_category": search_regex})
                
        if len(or_filters) == 1:
            or_filters.extend([
                {"benchmark_type": search_regex},
                {"benchmark_category": search_regex}
            ])
            
        match_query["$or"] = or_filters

    skip_count = (page - 1) * size
    
    # 2. Extract Data (Use $lookup to join with ExecutionInfo for guaranteed metadata)
    invalid_records = []
    pipeline = [
        {"$match": match_query},
        {"$sort": {"_id": -1}},
        {"$skip": skip_count},
        {"$limit": size},
        {"$lookup": {
            "from": EXECUTION_INFO_COL,
            "localField": "execution_id",
            "foreignField": "benchmarkExecutionID",
            "as": "exec_info"
        }},
        {"$unwind": {"path": "$exec_info", "preserveNullAndEmptyArrays": True}}
    ]
    
    cursor = db[SNAPSHOT_COL].aggregate(pipeline)
    
    async for doc in cursor:
        data_arr = doc.get("data", [{}])
        first_data = data_arr[0] if data_arr else {}
        exec_info = doc.get("exec_info") or {}
        
        record = {
            "ExecutionId": doc.get("execution_id"),
            "Status": first_data.get("standardization_status"),
            "BenchmarkType": (doc.get("benchmark_type") or 
                              first_data.get("benchmark_type") or 
                              exec_info.get("benchmarkType", "N/A")),
            "BenchmarkCategory": (doc.get("benchmark_category") or 
                                  first_data.get("benchmark_category") or 
                                  exec_info.get("benchmarkCategory", "N/A"))
        }
        
        # Collect all invalid fields: primary fields + metadata fields that are invalid
        invalid_fields = set()
        for val_item in first_data.get("invalidValues", []):
            if val_item.get("validation_status") == "invalid":
                invalid_fields.add(val_item.get("field"))
            for meta_item in val_item.get("metadata", []):
                if meta_item.get("validation_status") == "invalid":
                    invalid_fields.add(meta_item.get("name"))
       
        record["InvalidFields"] = sorted(list(invalid_fields))
        record["updatedOn"] = first_data.get("history", {}).get("updatedOn")
            
        invalid_records.append(record)

    # 3. Total Count Logic
    if search or status_filter != "PENDING":
        total_records = await db[SNAPSHOT_COL].count_documents(match_query)
    else:
        global _report_cache
        if _report_cache["total_invalid"]["value"] is None or (time.time() - _report_cache["total_invalid"]["updated_at"]) > CACHE_TTL:
            _report_cache["total_invalid"]["value"] = await db[SNAPSHOT_COL].count_documents({"data.standardization_status": "PENDING"})
            _report_cache["total_invalid"]["updated_at"] = time.time()
        total_records = _report_cache["total_invalid"]["value"]
    
    # 4. Aging Counts (Red/Yellow/Green)
    counts = {"red": 0, "yellow": 0, "green": 0}
    if (search and search.strip()) or status_filter != "ALL":
        # Dynamic aging counts for the current search or status filter
        counts = {"red": 0, "yellow": 0, "green": 0}
        count_agg = [
            {"$match": match_query},
            {"$addFields": {
                # Accessing the first history timestamp safely from the data array
                "updatedOn": {"$arrayElemAt": ["$data.history.updatedOn", 0]}
            }},
            # Ensure the field exists and is a valid string before attempting to parse it as a date
            {"$match": {"updatedOn": {"$type": "string"}}},
            {"$addFields": {
                "now": datetime.utcnow(),
                "dt": {"$dateFromString": {"dateString": "$updatedOn", "onError": None}}
            }},
            # Filter out any records where date parsing failed
            {"$match": {"dt": {"$ne": None}}},
            {"$addFields": {
                "diffDays": {
                    "$divide": [{"$subtract": ["$now", "$dt"]}, 86400000]
                }
            }},
            {"$group": {
                "_id": {
                    "$cond": [
                        {"$lt": ["$diffDays", 3]}, "green",
                        {"$cond": [{"$lte": ["$diffDays", 6]}, "yellow", "red"]}
                    ]
                },
                "count": {"$sum": 1}
            }}
        ]
        async for result_doc in db[SNAPSHOT_COL].aggregate(count_agg):
            if result_doc["_id"] in counts:
                counts[result_doc["_id"]] = result_doc["count"]
    else:
        # Global cached counts for the default view
        counts = await get_invalid_summary_counts()
        
    return {
        "status": "success",
        "total_invalid_records": total_records,
        "red": counts["red"],
        "yellow": counts["yellow"],
        "green": counts["green"],
        "page": page,
        "size": size,
        "returned_records": len(invalid_records),
        "data": invalid_records
    }

async def _get_masterlist_all_unique_values(db):
    """
    Helper to fetch all published unique values for all supported parameters (including metadata).
    """
    mappings = await build_mappings()
    
    if "Benchmark" in mappings and "BenchmarkCategory" not in mappings:
        mappings["BenchmarkCategory"] = mappings["Benchmark"]
    
    all_params = list(mappings.keys())
    all_types = await db[MASTERLIST_COL].distinct("type", {"status": "Published"})
    type_set = set(all_types)
    
    unique_data = {}
    for param in all_params:
        if param == "BenchmarkCategory" or param in type_set:
            query_type = "Benchmark" if param == "BenchmarkCategory" else param
            values = await db[MASTERLIST_COL].distinct("data.value", {"type": query_type, "status": "Published"})
        else:
            values = await db[MASTERLIST_COL].distinct(f"data.metadata.{param}", {"status": "Published"})
            
        # Normalize to string and strip to remove duplicates like 128 and "128" or trailing spaces
        unique_data[param] = sorted(list(set(str(v).strip() for v in values if v is not None)))
    
    return unique_data



@router.get("/metadata-values/{type_name}/{value}")
async def get_metadata_for_value(type_name: str, value: str):
    """
    Given a primary field type (e.g. 'CPUModel') and a selected value (e.g. '7543'),
    returns all metadata configurations associated with that value from the masterlist,
    along with their mapping paths.
    
    Use case: When user selects a value from the dropdown, this populates
    the dependent metadata fields (Family, coreCount, etc.) with valid options.
    """
    db = get_db()
    
    # Fetch all published masterlist records matching this type and value
    cursor = db[MASTERLIST_COL].find({
        "type": type_name,
        "data.value": value,
        "status": "Published"
    })
    
    records = await cursor.to_list(length=100)
    
    if not records:
        return {
            "status": "success",
            "type": type_name,
            "value": value,
            "metadata_records": [],
            "total_records": 0
        }
    
    metadata_records = []
    for record in records:
        record_id = record.get("`_id`") or record.get("id") or str(record.get("_id", ""))
        if isinstance(record_id, dict) and "$oid" in record_id:
            record_id = record_id["$oid"]
        
        data = record.get("data", {})
        meta = data.get("metadata", {})
        
        if not isinstance(meta, dict):
            continue
        
        meta_fields = {}
        meta_mappings = {}
        
        for mk, mv in meta.items():
            if mk.startswith("mapping_") or mk == "mapping":
                continue
            
            # Find the mapping for this metadata key
            lookup_key = f"mapping_{mk}".lower()
            mapping_path = None
            for k, v in meta.items():
                if k.lower() == lookup_key:
                    mapping_path = v
                    break
            if not mapping_path:
                mapping_path = meta.get("mapping", "")
            
            meta_fields[mk] = str(mv).strip()
            meta_mappings[mk] = mapping_path or ""
        
        if meta_fields:
            metadata_records.append({
                "_id": str(record_id),
                "metadata": meta_fields,
                "metadata_mappings": meta_mappings
            })
    
    return {
        "status": "success",
        "type": type_name,
        "value": value,
        "metadata_records": metadata_records,
        "total_records": len(metadata_records)
    }

@router.get("/validation-counts")
async def get_validation_counts():
    """
    Asynchronously returns the count of valid, invalid, and missing data for all mapped parameters.
    Dynamically discovers parameter types from the masterlist.
    Returns from cache instantly if within TTL.
    """
    global _report_cache
    if _report_cache["counts_metrics"]["value"] is not None and (time.time() - _report_cache["counts_metrics"]["updated_at"]) <= CACHE_TTL:
        return _report_cache["counts_metrics"]["value"]
        
    db = get_db()
    mappings = await build_mappings()
    
    facet_dict: Dict[str, Any] = {}
    for t in mappings.keys():
        facet_dict[t] = [
            {"$group": {
                "_id": None,
                "valid":   {"$sum": {"$cond": [{"$in": [t, {"$ifNull": ["$all_invalid_fields", []]}]}, 0, 1]}},
                "invalid": {"$sum": {"$cond": [{"$in": [t, {"$ifNull": ["$all_invalid_fields", []]}]}, 1, 0]}},
            }}
        ]
        
    facet_dict["total_docs"] = [{"$count": "total"}]
    
    pipeline = [
        {"$group": {
            "_id": "$benchmarkExecutionID",
            "invalidPayload": {"$first": "$invalidPayload"}
        }},
        {"$addFields": {
            "all_invalid_fields": {
                "$reduce": {
                    "input": {
                        "$map": {
                            "input": {"$ifNull": ["$invalidPayload", []]},
                            "as": "p",
                            "in": {
                                "$concatArrays": [
                                    {"$cond": [{"$eq": ["$$p.validation_status", "invalid"]}, ["$$p.field"], []]},
                                    {
                                        "$map": {
                                            "input": {
                                                "$filter": {
                                                    "input": {"$ifNull": ["$$p.metadata", []]},
                                                    "as": "m",
                                                    "cond": {"$eq": ["$$m.validation_status", "invalid"]}
                                                }
                                            },
                                            "as": "m_valid",
                                            "in": "$$m_valid.name"
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    "initialValue": [],
                    "in": {"$concatArrays": ["$$value", "$$this"]}
                }
            }
        }},
        {"$facet": facet_dict}
    ]
    
    cursor = db[EXECUTION_INFO_COL].aggregate(pipeline)
    result = await cursor.to_list(length=1)
    
    counts = {}
    total_docs = 0
    if result and len(result) > 0:
        res = result[0]
        total_docs = res.get("total_docs", [{"total": 0}])[0].get("total", 0) if res.get("total_docs") else 0
        
        for t in mappings.keys():
            t_data = res.get(t, [{"valid": 0, "invalid": 0}])[0]
            counts[t] = {
                "valid":   t_data.get("valid",   0),
                "invalid": t_data.get("invalid", 0),
            }
    
    response_payload = {
        "status": "success",
        "total_records_processed": total_docs,
        "counts_per_parameter": counts
    }
    
    _report_cache["counts_metrics"]["value"] = response_payload
    _report_cache["counts_metrics"]["updated_at"] = time.time()
    
    return response_payload

async def get_masterlist_mappings(field_type: str) -> dict:
    """
    Helper to fetch mapping definitions from the masterlist for a specific field type.
    Example: CPUModel -> {'mapping': '...', 'mappings': {'CPUs': '...'}}
    """
    db = get_db()
    mappings = {"mapping": "", "metadata_mappings": {}}
    
    doc = await db[MASTERLIST_COL].find_one({"type": field_type})
    if not doc or "data" not in doc:
        return mappings
    
    data = doc["data"]
    mappings["mapping"] = data.get("mapping", "")
    
    # Extract metadata mappings (different structures exist in masterlist)
    # Strategy 1: Look in 'metadata' object (like instanceType)
    if "metadata" in data and isinstance(data["metadata"], dict):
        for meta_name, meta_obj in data["metadata"].items():
            if isinstance(meta_obj, dict) and "mapping" in meta_obj:
                mappings["metadata_mappings"][meta_name] = meta_obj["mapping"]
    
    # Strategy 2: Look in representative data entries (like CPUModel/Family)
    # We find the first key that isn't 'mapping' or 'value'
    for k, v in data.items():
        if k not in ["mapping", "value", "metadata"] and isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_k.startswith("mapping_"):
                    meta_name = sub_k.replace("mapping_", "")
                    mappings["metadata_mappings"][meta_name] = sub_v
            break # Just need one representative entry
            
    return mappings

def _set_nested_key(dic, path, value):
    """
    Helper to set a value in a nested dictionary using a dot-separated path string.
    Correctly handles both nested objects and maintains original structure.
    """
    keys = path.split(".")
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    
@router.get("/snapshot-records/{Execution_id}")
async def get_snapshot_records(Execution_id: str):
    """
    Fetches a specific record from the snapshot collection by Execution_id (Path Parameter).
    Flattens invalid metadata into a simple Data array with mappings.
    """
    db = get_db()
    
    doc = await db[SNAPSHOT_COL].find_one({"execution_id": Execution_id})
    
    if not doc or not doc.get("data"):
        return {
            "status": "error",
            "message": f"No snapshot record found for Execution_id: {Execution_id}"
        }

    # Fetch metadata from ExecutionInfo
    exec_meta = await db[EXECUTION_INFO_COL].find_one({"benchmarkExecutionID": Execution_id})
    if not exec_meta:
        # Fallback to empty if not found, though it should exist
        exec_meta = {}

    item = doc["data"][0]
    validator = await get_validator()
    
    if exec_meta:
        # Live-evaluate the execution info against today's correct masterlist rules
        invalid_payload, field_status = await validator.validate_doc(db, exec_meta)
        if len(invalid_payload) == 0:
            # The record legally passes all current validation rules.
            # Flag it as valid natively!
            await db[EXECUTION_INFO_COL].update_one(
                {"benchmarkExecutionID": Execution_id},
                {
                    "$set": {"isValid": True, "invalidPayload": []},
                    "$unset": {"fieldStatus": ""}
                }
            )
            
            # Make sure we preserve the snapshot for the history/Accepted tab
            current_status = str(item.get("standardization_status", "")).upper()
            if current_status not in ["ACCEPTED", "REJECTED", "ON HOLD"]:
                await db[SNAPSHOT_COL].update_one(
                    {"execution_id": Execution_id},
                    {"$set": {"data.0.standardization_status": "ACCEPTED"}}
                )
                item["standardization_status"] = "ACCEPTED"
            
            # We no longer delete the snapshot nor return early. We let the function continue
            # so the UI can fetch and display the full history of this Accepted record!
            
    # Fetch mapping and validation details dynamically for EACH individual field
    type_mappings = {}
    data_list = []
    
    # Process each invalid primary field from the snapshot record
    for meta in item.get("invalidValues", []):
        field_name = meta.get("field")
        if not field_name:
            continue
            
        if field_name not in type_mappings:
            type_mappings[field_name] = await get_masterlist_mappings(field_name)
            
        val = meta.get("value")
        
        # 1. Build existing_data for this field (primary field + metadata)
        field_existing_data = []
        field_existing_data.append({
            "field": field_name,
            "value": val,
            "validation_status": meta.get("validation_status")
        })
        
        for support in meta.get("metadata", []):
            field_existing_data.append({
                "field": support.get("name"),
                "value": support.get("value"),
                "validation_status": support.get("validation_status")
            })
            
        # 2. Build suggestions for this field (grouped by masterlist record using ANN)
        actual_meta_vals = {s.get("name"): s.get("value", "") for s in meta.get("metadata", []) if s.get("name")}
        record_suggestions = await validator.get_record_level_suggestions_ann(field_name, val, actual_meta_vals)
        
        saved_comparing = meta.get("comparingData", [])
        
        field_suggestions = []
        for rec_sug in record_suggestions:
            sug_val = rec_sug["primary_value"]
            saved_status = "PENDING"
            
            # Find the saved status for this suggestion from the DB
            for saved_sug in saved_comparing:
                match_val = next((v for k, v in saved_sug.items() if k.startswith("suggestion")), None)
                if match_val == sug_val:
                    saved_status = saved_sug.get("status", "PENDING")
                    break

            sug_entry = {
                field_name.lower(): sug_val,
                "score": rec_sug.get("score", 0),
                "status": saved_status
            }
            for m_name, m_val in rec_sug["metadata"].items():
                sug_entry[m_name.lower()] = m_val
            field_suggestions.append(sug_entry)
            
        data_list.append({
            "invalid_field": field_name,
            "currentStatus": meta.get("currentStatus", "invalid"),
            "existing_data": field_existing_data,
            "suggestions": field_suggestions
        })
    
    # Restructure history into changes array
    raw_history = item.get("history", {})
    hist_from = raw_history.get("from") or []
    hist_to = raw_history.get("to") or []
    hist_fields = raw_history.get("valueField") or []
    
    if not isinstance(hist_from, list): hist_from = [hist_from]
    if not isinstance(hist_to, list): hist_to = [hist_to]
    if not isinstance(hist_fields, list): hist_fields = [hist_fields]
    
    changes_by_field = {}
    for idx in range(len(hist_fields)):
        f = hist_fields[idx] if idx < len(hist_fields) else ""
        frm = hist_from[idx] if idx < len(hist_from) else ""
        to = hist_to[idx] if idx < len(hist_to) else ""
        if f not in changes_by_field:
            changes_by_field[f] = {"field": f, "from": [], "to": []}
        changes_by_field[f]["from"].append(frm)
        changes_by_field[f]["to"].append(to)
        
    # Get execution info for detailed response
    sut_type = exec_meta.get("sutInstanceMetadata.sutType") if "sutInstanceMetadata.sutType" in exec_meta else exec_meta.get("sutInstanceMetadata", {}).get("sutType")
    
    return {
        "snapshot_id": doc.get("snapshot_id"),
        "execution_details": {
            "execution_id":      doc.get("execution_id"),
            "benchmarkType":     exec_meta.get("benchmarkType"),
            "benchmarkCategory": exec_meta.get("benchmarkCategory"),
            "sutType":           sut_type,
            "runCategory":       exec_meta.get("runCategory"),
            "createdOn":         exec_meta.get("createdOn"),
            "tester":            exec_meta.get("tester"),
            "resultType":        exec_meta.get("resultType"),
        },
        "data": data_list,
        "standardization_status": item.get("standardization_status", "PENDING"),
        "reason": item.get("reason"),
        "history": {
            "updatedOn": raw_history.get("updatedOn"),
            "updatedBy": raw_history.get("updatedBy"),
            "changes": list(changes_by_field.values())
        }
    }

@router.get("/unique-values")
async def get_unique_values(parameterName: Optional[str] = Query(None)):
    """
    Fetches unique values for all validated parameters from the masterlist collection.
    Dynamically discovers all available types including metadata fields.
    """
    db = get_db()
    
    # Dynamically discover all mapping parameters (including metadata)
    mappings = await build_mappings()
    
    # Support 'BenchmarkCategory' as an alias for the primary 'Benchmark' type
    if "Benchmark" in mappings and "BenchmarkCategory" not in mappings:
        mappings["BenchmarkCategory"] = mappings["Benchmark"]
    
    # Build discovery map
    all_params = list(mappings.keys())
    param_map = {p.lower(): p for p in all_params}
    
    # Get all top-level types to distinguish between primary values and metadata
    all_types = await db[MASTERLIST_COL].distinct("type", {"status": "Published"})
    type_set = set(all_types)

    if parameterName:
        param_norm = parameterName.lower()
        
        # Explicit mapping for common parameter names
        # Some are 'types' in masterlist, others are 'metadata' fields
        mapping_logic = {
            "cpumodel":          {"type": "CPUModel", "path": "data.value"},
            "instancetype":      {"type": "instanceType", "path": "data.value"},
            "benchmarktype":     {"type": "Benchmark", "path": "data.metadata.BenchmarkType"},
            "benchmarkcategory": {"type": "Benchmark", "path": "data.value"},
            "family":            {"type": None, "path": "data.metadata.Family"},
            "corecount":         {"type": None, "path": "data.metadata.coreCount"},
            "cloudprovider":     {"type": None, "path": "data.metadata.cloudProvider"}
        }
        
        if param_norm in mapping_logic:
            rule = mapping_logic[param_norm]
            query = {"status": "Published"}
            if rule["type"]:
                query["type"] = rule["type"]
            
            values = await db[MASTERLIST_COL].distinct(rule["path"], query)
            actual_param = parameterName # Keep user's casing for the response
        else:
            # Fallback for other potential parameters discovered via mappings
            if param_norm not in param_map:
                return {
                    "status": "error",
                    "message": f"Invalid parameterName. Supported values: {list(mapping_logic.keys())} or {all_params}"
                }
            
            actual_param = param_map[param_norm]
            if actual_param in type_set:
                values = await db[MASTERLIST_COL].distinct("data.value", {"type": actual_param, "status": "Published"})
            else:
                values = await db[MASTERLIST_COL].distinct(f"data.metadata.{actual_param}", {"status": "Published"})
            
        return {
            "status": "success",
            "unique_values": {
                actual_param: sorted(list(set(str(v).strip() for v in values if v is not None and v != "")))
            }
        }

    # Default: Return all unique data lists dynamically
    unique_data = await _get_masterlist_all_unique_values(db)

    return {
        "status": "success",
        "unique_values": unique_data
    }

async def get_masterlist_values(field_type: str) -> List[str]:
    """Helper to fetch unique published values for a masterlist type."""
    db = get_db()
    values = await db[MASTERLIST_COL].distinct("data.value", {"type": field_type, "status": "Published"})
    return [str(v) for v in values if v]

@router.get("/search-snapshots")
async def search_snapshots(
    status: str = Query("PENDING", description="Filter by status: PENDING, REJECTED, ACCEPTED, 'On Hold'"),
    benchmarkType: Optional[str] = Query(None),
    benchmarkCategory: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500)
):
    db = get_db()
    
    # Fuzzy resolution
    resolved = await resolve_fuzzy_benchmarks(benchmarkType, benchmarkCategory)
    
    # 1. First, search ExecutionInfo to get the IDs of benchmarks that match
    exec_query = {}
    if "benchmarkType" in resolved:
        if resolved.get("benchmarkType_is_fuzzy"):
            exec_query["benchmarkType"] = resolved["benchmarkType"]
        else:
            exec_query["benchmarkType"] = {"$regex": resolved["benchmarkType"], "$options": "i"}
            
    if "benchmarkCategory" in resolved:
        if resolved.get("benchmarkCategory_is_fuzzy"):
            exec_query["benchmarkCategory"] = resolved["benchmarkCategory"]
        else:
            exec_query["benchmarkCategory"] = {"$regex": resolved["benchmarkCategory"], "$options": "i"}

    if search:
        search_regex = {"$regex": search, "$options": "i"}
        # Fuzzy resolution for both Type and Category
        resolved = await resolve_fuzzy_benchmarks(benchmarkType=search, benchmarkCategory=search)
        or_filters = [{"benchmarkExecutionID": search_regex}]
        
        if "benchmarkType" in resolved:
            if resolved.get("benchmarkType_is_fuzzy"):
                or_filters.append({"benchmarkType": resolved["benchmarkType"]})
            else:
                or_filters.append({"benchmarkType": search_regex})
        
        if "benchmarkCategory" in resolved:
            if resolved.get("benchmarkCategory_is_fuzzy"):
                or_filters.append({"benchmarkCategory": resolved["benchmarkCategory"]})
            else:
                or_filters.append({"benchmarkCategory": search_regex})
        
        # Always fallback to general regex on all three if no fuzzy matches found yet
        if len(or_filters) == 1:
            or_filters.extend([
                {"benchmarkType": search_regex},
                {"benchmarkCategory": search_regex}
            ])
            
        exec_query["$or"] = or_filters

    if not exec_query:
        return {"status": "success", "data": [], "message": "No search parameters provided."}

    # Get matching IDs (limited by a reasonable amount to prevent enormous $in clauses)
    # Actually, we can just use the query to find in ExecutionInfo and THEN snapshot
    matching_ids = await db[EXECUTION_INFO_COL].distinct("benchmarkExecutionID", exec_query)
    
    if not matching_ids:
        return {"status": "success", "data": [], "count": 0}

    skip_count = (page - 1) * size
    
    # 2. Query snapshots for these IDs
    cursor = db[SNAPSHOT_COL].find({"execution_id": {"$in": matching_ids}}).sort([("_id", -1)]).skip(skip_count).limit(size)
    
    results = []
    async for doc in cursor:
        exec_id = doc.get("execution_id")
        # Fetch metadata for this specific record (cached or fast lookup)
        exec_meta = await db[EXECUTION_INFO_COL].find_one({"benchmarkExecutionID": exec_id})
        
        item = doc["data"][0] if doc.get("data") else {}
        results.append({
            "snapshot_id": doc.get("snapshot_id"),
            "execution_id": exec_id,
            "benchmarkType": exec_meta.get("benchmarkType") if exec_meta else None,
            "benchmarkCategory": exec_meta.get("benchmarkCategory") if exec_meta else None,
            "runCategory": exec_meta.get("runCategory") if exec_meta else None,
            "createdOn": exec_meta.get("createdOn") if exec_meta else None,
            "tester": exec_meta.get("tester") if exec_meta else None,
            "standardization_status": item.get("standardization_status"),
            "history": item.get("history", {})
        })

    return {
        "status": "success",
        "count": len(results),
        "page": page,
        "size": size,
        "data": results
    }

async def resolve_fuzzy_benchmarks(benchmarkType: Optional[str] = None, benchmarkCategory: Optional[str] = None) -> Dict[str, Any]:
    """Helper to resolve fuzzy benchmark terms into exact masterlist values."""
    resolved = {}
    if benchmarkType:
        valid_types = await get_masterlist_values("BenchmarkType")
        valid_map = {v.lower(): v for v in valid_types}
        match_res = process.extractOne(benchmarkType.lower(), valid_map.keys(), score_cutoff=60)
        if match_res:
             match_str = match_res[0]
             resolved["benchmarkType"] = valid_map[match_str]
             resolved["benchmarkType_is_fuzzy"] = True
        else:
             resolved["benchmarkType"] = benchmarkType
             resolved["benchmarkType_is_fuzzy"] = False

    if benchmarkCategory:
        valid_cats = await get_masterlist_values("BenchmarkCategory")
        valid_map = {v.lower(): v for v in valid_cats}
        match_res = process.extractOne(benchmarkCategory.lower(), valid_map.keys(), score_cutoff=60)
        if match_res:
             match_str = match_res[0]
             resolved["benchmarkCategory"] = valid_map[match_str]
             resolved["benchmarkCategory_is_fuzzy"] = True
        else:
             resolved["benchmarkCategory"] = benchmarkCategory
             resolved["benchmarkCategory_is_fuzzy"] = False
    
    return resolved


@router.put("/approve-suggestion")
async def approve_suggestion(req: ApproveSuggestionRequest):
    """
    Standardizes an invalid record by approving a specific suggestion or a custom dropdown value.
    - If the accepted_value matches a suggestion, that suggestion is 'Accepted' and others are 'Rejected'.
    - If the accepted_value doesn't match any suggestion (dropdown selection), ALL suggestions are 'Rejected'.
    Updates the snapshot status and propagates the accepted value to the original Executioninfo record.
    """
    db = get_db()
    
    # 1. Fetch Snapshot
    snap = await db[SNAPSHOT_COL].find_one({"execution_id": req.execution_id})
    if not snap or not snap.get("data"):
        return {"status": "error", "message": f"Snapshot not found for Execution ID: {req.execution_id}"}
    
    snap_data = snap["data"][0]
    invalid_values = snap_data.get("invalidValues", [])
    
    # 2. Identify the selected field and suggestion (supporting nested metadata)
    target_item = None
    target_mapping = None
    original_value = None
    suggestion_found = False
    
    # Check top-level invalid fields first
    for item in invalid_values:
        if item.get("field") == req.field_name:
            target_item = item
            original_value = item.get("value")
            # Fetch primary mapping
            m_info = await get_masterlist_mappings(req.field_name)
            target_mapping = m_info.get("mapping")
            break
        
        # Check nested metadata fields
        for meta_item in item.get("metadata", []):
            if meta_item.get("name") == req.field_name:
                target_item = meta_item
                original_value = meta_item.get("value")
                # Fetch metadata-specific mapping
                m_info = await get_masterlist_mappings(item.get("field"))
                target_mapping = m_info.get("metadata_mappings", {}).get(req.field_name)
                break
        if target_item: break

    if not target_item:
        return {"status": "error", "message": f"Field '{req.field_name}' not found in snapshot."}
        
    # 3. Update suggestion statuses
    comparing_data = target_item.get("comparingData", [])
    accepted_sug_num = None
    for sug in comparing_data:
        match_key = next((k for k in sug.keys() if k.startswith("suggestion")), None)
        if match_key and sug[match_key] == req.accepted_value:
            sug["status"] = "Accepted"
            suggestion_found = True
            accepted_sug_num = match_key.replace("suggestion", "")
        else:
            sug["status"] = "Rejected"
    
    # Determine the source of the accepted value
    value_source = "suggestion" if suggestion_found else "dropdown"
            
    # Always set to valid since we are finalizing a correction
    target_item["validation_status"] = "valid"
    target_item["currentStatus"] = req.currentStatus
        
    # 4. Propagate the change to Executioninfo if a mapping exists
    if target_mapping:
        await db[EXECUTION_INFO_COL].update_one(
            {"benchmarkExecutionID": req.execution_id},
            {"$set": {target_mapping: req.accepted_value}}
        )
    
    # 4b. CASCADE: If this is a primary field, apply the same suggestion status to ALL metadata
    is_primary_field = False
    parent_item = None
    cascaded_changes = []  # Track (field_name, from_value, to_value) for history
    
    for item in invalid_values:
        if item.get("field") == req.field_name:
            is_primary_field = True
            parent_item = item
            break
    
    if is_primary_field and parent_item:
        # Fetch metadata mappings for this primary field type
        m_info = await get_masterlist_mappings(req.field_name)
        meta_mappings = m_info.get("metadata_mappings", {})
        
        # If Dropdown, dynamically fetch the Masterlist Record for the exact value they chose!
        dropdown_metadata = {}
        if value_source == "dropdown":
            dropdown_ml_record = await db[MASTERLIST_COL].find_one({
                "type": {"$regex": f"^{req.field_name}$", "$options": "i"},
                "data.value": req.accepted_value,
                "status": "Published"
            })
            if dropdown_ml_record:
                dropdown_metadata = dropdown_ml_record.get("data", {}).get("metadata", {})
        
        for meta_item in parent_item.get("metadata", []):
            meta_name = meta_item.get("name")
            meta_original_value = meta_item.get("value")
            meta_accepted_value = None
            meta_comparing = meta_item.get("comparingData", [])
            
            if accepted_sug_num:
                # Suggestion case: accept the same suggestion number, reject others
                for sug in meta_comparing:
                    sug_key = next((k for k in sug if k.startswith("suggestion")), None)
                    if sug_key and sug_key == f"suggestion{accepted_sug_num}":
                        sug["status"] = "Accepted"
                        meta_accepted_value = sug[sug_key]
                    else:
                        sug["status"] = "Rejected"
            else:
                # Dropdown case: reject all metadata suggestions
                for sug in meta_comparing:
                    sug["status"] = "Rejected"
                    
                # Dynamically fetch the correct metadata value from the database record!
                for k, v in dropdown_metadata.items():
                    if k.lower() == str(meta_name).lower():
                        meta_accepted_value = v
                        break
                        
            # --- CUSTOM OVERRIDE FOR CORECOUNT WHEN EDITABLE ---
            if meta_name == "coreCount" and getattr(req, "coreCount", None) is not None:
                meta_accepted_value = req.coreCount
            # ---------------------------------------------------
            
            # Mark metadata as valid (do NOT update the value field in the snapshot)
            meta_item["validation_status"] = "valid"
            
            # Propagate metadata value to ExecutionInfo if we have a suggestion value and mapping
            if meta_accepted_value and meta_name:
                meta_mapping = meta_mappings.get(meta_name)
                if meta_mapping:
                    await db[EXECUTION_INFO_COL].update_one(
                        {"benchmarkExecutionID": req.execution_id},
                        {"$set": {meta_mapping: meta_accepted_value}}
                    )
            
            # Track for history
            cascaded_changes.append({
                "field": meta_name,
                "from": meta_original_value,
                "to": meta_accepted_value or meta_original_value,
                "source": "manual" if (meta_name == "coreCount" and getattr(req, "coreCount", None) is not None) else value_source
            })
    
    # 5. Check for Overall Validity (Standardization Status)
    is_fully_resolved = True
    for item in invalid_values:
        if item.get("validation_status") != "valid":
            is_fully_resolved = False
            break
        for meta in item.get("metadata", []):
            if meta.get("validation_status") != "valid":
                is_fully_resolved = False
                break
        if not is_fully_resolved: break

    # 6. Update History (BUILD THIS BEFORE FINAL SYNC)
    history = snap_data.get("history", {})
    orig_from = history.get("from")
    orig_to = history.get("to")
    orig_field = history.get("valueField")
    orig_source = history.get("source")
    
    new_from = orig_from if isinstance(orig_from, list) else ([orig_from] if orig_from else [])
    new_to = orig_to if isinstance(orig_to, list) else ([orig_to] if orig_to else [])
    new_field = orig_field if isinstance(orig_field, list) else ([orig_field] if orig_field else [])
    new_source = orig_source if isinstance(orig_source, list) else ([orig_source] if orig_source else [])
    
    # Add primary field history
    new_from.append(original_value)
    new_to.append(req.accepted_value)
    new_field.append(req.field_name)
    new_source.append(value_source)
    
    # Add cascaded metadata history
    for change in cascaded_changes:
        new_from.append(change["from"])
        new_to.append(change["to"])
        new_field.append(change["field"])
        new_source.append(change.get("source", value_source))

    snap_data["history"] = {
        "updatedOn": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "updatedBy": "xxx@amd.com",
        "from": new_from,
        "to": new_to,
        "valueField": new_field,
        "source": new_source
    }

    # 7. Final Acceptance Transition & Consistency Sync
    if is_fully_resolved:
        snap_data["standardization_status"] = "ACCEPTED"
        
        # --- FINAL CONSISTENCY SYNC (Double-Sync driven by History) ---
        ei_doc = await db[EXECUTION_INFO_COL].find_one({"benchmarkExecutionID": req.execution_id})
        
        if ei_doc:
            # Build a lookup for mappings from the current snapshot structure
            mapping_lookup = {}
            for item in invalid_values:
                field_name = item.get("field")
                if field_name:
                    mapping_lookup[field_name] = item.get("mapping")
                for meta in item.get("metadata", []):
                    meta_name = meta.get("name")
                    if meta_name:
                        mapping_lookup[meta_name] = meta.get("mapping")
            
            # Sync ALL fields currently present in the final history to Executioninfo
            final_updates_applied = False
            for i, field_name in enumerate(new_field):
                m_path = mapping_lookup.get(field_name)
                m_val = new_to[i] if i < len(new_to) else None
                
                if m_path and m_val is not None:
                    final_updates_applied = True
                    # A. Update Flattened literal key
                    if m_path in ei_doc:
                        ei_doc[m_path] = m_val
                    # B. Update Nested Path
                    _set_nested_key(ei_doc, m_path, m_val)
            
            if final_updates_applied:
                print(f"Applying Final History-Driven Double-Sync for {req.execution_id}")
                await db[EXECUTION_INFO_COL].replace_one(
                    {"_id": ei_doc["_id"]},
                    ei_doc
                )
    else:
        # Keep as PENDING if not all fields are resolved
        snap_data["standardization_status"] = "PENDING"

    # 7. Mark Executioninfo as valid ONLY if fully resolved
    if is_fully_resolved:
        await db[EXECUTION_INFO_COL].update_one(
            {"benchmarkExecutionID": req.execution_id},
            {"$set": {"isValid": True}}
        )
    
    # 8. Save entire Snapshot back
    await db[SNAPSHOT_COL].replace_one({"execution_id": req.execution_id}, snap)
    
    return {
        "status": "success",
        "message": f"Successfully {'accepted suggestion' if value_source == 'suggestion' else 'applied custom value'} for '{req.field_name}' and updated Executioninfo.",
        "execution_id": req.execution_id,
        "updated_field": req.field_name,
        "accepted_value": req.accepted_value,
        "mapping_path": target_mapping,
        "value_source": value_source
    }

@router.put("/reject-record")
async def reject_record(req: RejectRecordRequest):
    """
    Manually rejects an entire record that cannot be standardized.
    Marks all suggestions as 'Rejected' and sets the standardization status to 'REJECTED'.
    The record remains 'isValid: False' in the source collection.
    """
    db = get_db()
    execution_id = req.execution_id
    
    # 1. Fetch Snapshot
    snap = await db[SNAPSHOT_COL].find_one({"execution_id": execution_id})
    if not snap or not snap.get("data"):
        return {"status": "error", "message": f"Snapshot not found for Execution ID: {execution_id}"}
    
    snap_data = snap["data"][0]
    invalid_values = snap_data.get("invalidValues", [])
    
    # 2. Mark all suggestions for all fields and metadata as "Rejected"
    new_from = []
    new_field = []
    new_source = []
    
    for item in invalid_values:
        field_name = item.get("field")
        val = item.get("value")
        item["currentStatus"] = req.currentStatus
        
        # Track for history
        new_from.append(val)
        new_field.append(field_name)
        new_source.append("reject")
        
        # Reject primary suggestions
        for sug in item.get("comparingData", []):
            sug["status"] = "Rejected"
            
        # Reject metadata suggestions
        for meta in item.get("metadata", []):
            new_from.append(meta.get("value"))
            new_field.append(meta.get("name"))
            new_source.append("reject")
            
            for sug in meta.get("comparingData", []):
                sug["status"] = "Rejected"
    
    # 3. Transition Standardization Status
    snap_data["standardization_status"] = "REJECTED"
    snap_data["reason"] = "L0 Junk Data."
    
    # 4. Update History
    history = snap_data.get("history", {})
    snap_data["history"] = {
        "updatedOn": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "updatedBy": "xxx@amd.com",
        "from": new_from,
        "to": new_from, # Use same values as 'to' as no fix was applied
        "valueField": new_field,
        "source": new_source
    }
    
    # 5. Update Executioninfo to include the entitliment_level
    await db[EXECUTION_INFO_COL].update_one(
        {"benchmarkExecutionID": execution_id},
        {"$set": {"entitliment_level": "L0 Junk Data."}}
    )

    # 6. Save Snapshot
    await db[SNAPSHOT_COL].replace_one({"execution_id": execution_id}, snap)
    
    return {
        "status": "success",
        "message": f"Execution ID: {execution_id} has been manually REJECTED. All fuzzy suggestions have been cleared.",
        "execution_id": execution_id,
        "standardization_status": "REJECTED"
    }


@router.get("/draft-executions")
async def get_draft_executions():
    """
    Queries the collection where status=Draft and returns only the IDs.
    """
    db = get_db()
    
    # 1. Fetch all Draft masterlist records
    cursor = db[MASTERLIST_COL].find({"status": "Draft"})
    draft_docs = await cursor.to_list(None)
    
    exec_ids = []
    for d in draft_docs:
        # Extract the UUID. It might be stored as `_id` due to a typo in the DB creation script,
        # or it might be in execution_id. Fall back to the MongoDB _id.
        if "`_id`" in d:
            exec_ids.append(d["`_id`"])
        elif "execution_id" in d:
            exec_ids.append(d["execution_id"])
        else:
            exec_ids.append(str(d["_id"]))
            
    return exec_ids

@router.post("/draft-records/{type_name}")
async def create_masterlist_draft(type_name: str, draft: DraftRecordRequest):
    """
    Unified endpoint to add a new masterlist record to "In Review" status.
    Handles CPUModel, instanceType, and Benchmark types dynamically.
    """
    db = get_db()
    type_norm = type_name.lower()
    
    # 1. Validation & Duplicate Prevention
    supported_types = list(_DRAFT_FIELDS_MAP.keys())
    if type_norm not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported record type: '{type_name}'. Supported types are: {supported_types}"
        )
        
    value = draft.value
    if not value:
        raise HTTPException(status_code=400, detail="The 'value' field is required in the request body.")
        
    actual_type = "CPUModel" if type_norm == "cpumodel" else ("instanceType" if type_norm == "instancetype" else type_name)
    
    # Extract metadata to check for exact uniqueness
    metadata_dict = {}
    if type_norm == "cpumodel":
        if draft.family: metadata_dict["Family"] = draft.family
        if draft.corecount: metadata_dict["coreCount"] = draft.corecount
    elif type_norm == "instancetype":
        if draft.cpumodel: metadata_dict["CPUModel"] = draft.cpumodel
        if draft.cloudprovider: metadata_dict["cloudProvider"] = draft.cloudprovider
        if draft.family: metadata_dict["Family"] = draft.family
        if draft.corecount: metadata_dict["coreCount"] = draft.corecount
    elif type_norm == "benchmark":
        if draft.benchmarktype: metadata_dict["BenchmarkType"] = draft.benchmarktype
        
    existing = await _check_duplicate(db, actual_type, value, metadata_dict)
    if existing:
        meta_str = ", ".join([f"{k}: '{v}'" for k, v in metadata_dict.items()])
        msg = f"A record for {actual_type} '{value}'"
        if meta_str:
            msg += f" with metadata ({meta_str})"
        msg += f" already exists in the masterlist (Status: {existing['status']})."
        
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": msg
            }
        )
        
    # 2. Build type-specific data content
    data_content = {}
    
    if type_norm == "cpumodel":
        data_content = {
            "sutType": "server",
            "mapping_sutType": "sutInstanceMetadata.sutType",
            "value": value,
            "mapping": "platformProfile.sut.Summary.Server.CPUModel",
            "metadata": {
                "Family": draft.family,
                "mapping_Family": "processor_details.family",
                "coreCount": draft.corecount,
                "mapping_coreCount": "platformProfile.sut.Summary.CPU.CPU(s)"
            }
        }
    elif type_norm == "instancetype":
        data_content = {
            "sutType": "cloud",
            "mapping_sutType": "sutInstanceMetadata.sutType",
            "value": value,
            "mapping": "sutInstanceMetadata.instanceType",
            "metadata": {
                "CPUModel": draft.cpumodel,
                "mapping_CPUModel": "platformProfile.sut.Summary.Server.CPUModel",
                "cloudProvider": draft.cloudprovider,
                "mapping_cloudProvider": "sutInstanceMetadata.cloudProvider",
                "Family": draft.family,
                "mapping_Family": "processor_details.family",
                "coreCount": draft.corecount,
                "mapping_coreCount": "platformProfile.sut.Summary.CPU.CPU(s)"
            }
        }
    elif type_norm == "benchmark":
        data_content = {
            "value": value,
            "mapping": "benchmarkCategory",
            "metadata": {
                "BenchmarkType": draft.benchmarktype,
                "mapping_benchmarkType": "benchmarkType"
            }
        }
        
    # 3. Final Document Construction (Type before Status, Data before History)
    ml_doc = _build_base_ml_doc(actual_type, data_content, "")
    
    await db[MASTERLIST_COL].insert_one(ml_doc)
    
    # 4. Place snapshots containing this drafted value "On Hold"
    # If a specific execution_id was provided, prioritize it
    update_filter = {"execution_id": draft.execution_id} if draft.execution_id else {"data.invalidValues.value": value}
    
    cursor = db[SNAPSHOT_COL].find(update_filter)
    async for snap in cursor:
        updated = False
        if "data" in snap and isinstance(snap["data"], list) and len(snap["data"]) > 0:
            data_dict = snap["data"][0]
            data_dict["standardization_status"] = "ON HOLD"
            data_dict["reason"] = "New Masterlist Draft Record."
            
            # Update Timestamp in history
            if "history" not in data_dict or not isinstance(data_dict["history"], dict):
                data_dict["history"] = {}
            data_dict["history"]["updatedOn"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            data_dict["history"]["updatedBy"] = "xxx@amd.com"
            
            for item in data_dict.get("invalidValues", []):
                # Update statuses for the drafted field
                if str(item.get("field")).lower() == actual_type.lower():
                    item["currentStatus"] = "ON HOLD"
                    
                    # Reject all primary fuzzy suggestions
                    for cmp in item.get("comparingData", []):
                        cmp["status"] = "REJECTED"
                        
                    # Reject all metadata fuzzy suggestions
                    for meta in item.get("metadata", []):
                        for cmp in meta.get("comparingData", []):
                            cmp["status"] = "REJECTED"
                            
                    updated = True
        
        if updated:
            await db[SNAPSHOT_COL].replace_one({"_id": snap["_id"]}, snap)
    
    return {
        "status": "success",
        "message": f"Successfully drafted {actual_type} record to masterlist with 'In Review' status. Affected snapshots placed 'On Hold'.",
        "id": ml_doc["`_id`"],
        "record_id": ml_doc["`_id`"],
        "execution_id": draft.execution_id or "Multiple"
    }

@router.get("/draft-records/fields")
async def get_draft_record_fields(type: str = Query(..., description="Record type: cpumodel, instancetype, benchmark")):
    """
    Returns the list of required field names for a given record type.
    Pass ?type=cpumodel, ?type=instancetype, or ?type=benchmark.
    """
    type_lower = type.lower()
    if type_lower not in _DRAFT_FIELDS_MAP:
        return {
            "status": "error",
            "message": f"Unsupported type: '{type}'. Supported values: {list(_DRAFT_FIELDS_MAP.keys())}"
        }
    return {
        "status": "success",
        "type": type_lower,
        "fields": _DRAFT_FIELDS_MAP[type_lower]
    }
