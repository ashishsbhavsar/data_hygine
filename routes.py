from fastapi import APIRouter, Query, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import rapidfuzz
from rapidfuzz import process, fuzz
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
    currentStatus: str

class RejectRecordRequest(BaseModel):
    execution_id: str
    currentStatus: str = "L0 Data"

class DraftRecordRequest(BaseModel):
    value: str
    id: Optional[str] = None
    execution_id: Optional[str] = None
    # Primary fields
    family: Optional[str] = ""
    corecount: Optional[str] = ""
    cpumodel: Optional[str] = ""
    cloudprovider: Optional[str] = ""
    benchmarktype: Optional[str] = ""
    # Technical Metadata
    cpus: Optional[str] = ""
    architecture: Optional[str] = ""
    sockets: Optional[str] = ""
    threadspercore: Optional[str] = ""
    corespersocket: Optional[str] = ""
    cpumaxmhz: Optional[str] = ""
    l3cache: Optional[str] = ""
    microcode: Optional[str] = ""
    l3cacheinstances: Optional[str] = ""
    model: Optional[str] = ""
    num_l3: Optional[str] = ""
    ccdcount: Optional[str] = ""
    coreperccd: Optional[str] = ""
    microarchitecture: Optional[str] = ""
    technology: Optional[str] = ""
    peakperformance: Optional[str] = ""

# Internal Helpers for Draft Workflows
async def _check_duplicate(db, type_name, value):
    """Checks if a record with the same type and value already exists in the masterlist."""
    return await db[MASTERLIST_COL].find_one({"type": type_name, "data.value": value})

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

def _populate_metadata_content(draft: DraftRecordRequest, type_norm: str):
    """Helper to populate the metadata object with all provided tech specs and their mappings."""
    metadata = {}
    
    # Static mappings for metadata discovered from the masterlist
    # We use lowercase keys for the request fields and proper casing for the masterlist
    field_config = {
        "family": {"name": "Family", "path": "processor_details.family"},
        "corecount": {"name": "coreCount", "path": "platformProfile.sut.Summary.CPU.CPU(s)"},
        "cpus": {"name": "CPU(s)", "path": "platformProfile.sut.Summary.CPU.CPU(s)"},
        "architecture": {"name": "Architecture", "path": "platformProfile.sut.Summary.CPU.Architecture"},
        "sockets": {"name": "Socket(s)", "path": "platformProfile.sut.Summary.CPU.Socket(s)"},
        "threadspercore": {"name": "Thread(s)PerCore", "path": "platformProfile.sut.Summary.CPU.Thread(s)PerCore"},
        "corespersocket": {"name": "Core(s)PerSocket", "path": "platformProfile.sut.Summary.CPU.Core(s)PerSocket"},
        "cpumaxmhz": {"name": "CPUMaxMHz", "path": "platformProfile.sut.Summary.CPU.CPUMaxMHz"},
        "l3cache": {"name": "L3Cache", "path": "platformProfile.sut.Summary.CPU.L3Cache"},
        "microcode": {"name": "Microcode", "path": "platformProfile.sut.Summary.CPU.Microcode"},
        "l3cacheinstances": {"name": "L3CacheInstances", "path": "platformProfile.sut.Summary.CPU.L3CacheInstances"},
        "model": {"name": "Model", "path": "platformProfile.sut.Summary.CPU.Model"},
        "num_l3": {"name": "num_L3", "path": "platformProfile.sut.Summary.CPU.num_L3"},
        "ccdcount": {"name": "CCDCount", "path": "platformProfile.sut.Summary.CPU.CCDCount"},
        "coreperccd": {"name": "CorePerCCD", "path": "platformProfile.sut.Summary.CPU.CorePerCCD"},
        "microarchitecture": {"name": "Microarchitecture", "path": "platformProfile.sut.Summary.CPU.Microarchitecture"},
        "technology": {"name": "Technology", "path": "platformProfile.sut.Summary.CPU.Technology"},
        "peakperformance": {"name": "PeakPerformance", "path": "platformProfile.sut.Summary.CPU.PeakPerformance"}
    }

    # Custom logic for instancetype which has CPUModel as metadata
    if type_norm == "instancetype" and draft.cpumodel:
        metadata["CPUModel"] = draft.cpumodel
        metadata["mapping_CPUModel"] = "platformProfile.sut.Summary.Server.CPUModel"
    
    if type_norm == "instancetype" and draft.cloudprovider:
        metadata["cloudProvider"] = draft.cloudprovider
        metadata["mapping_cloudProvider"] = "sutInstanceMetadata.cloudProvider"

    if type_norm == "benchmark" and draft.benchmarktype:
        metadata["BenchmarkType"] = draft.benchmarktype
        metadata["mapping_benchmarkType"] = "benchmarkType"

    # Populate technical specs if provided
    for field_key, config in field_config.items():
        val = getattr(draft, field_key, None)
        if val:
            metadata[config["name"]] = val
            metadata[f"mapping_{config['name']}"] = config["path"]
            
    return metadata

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
            best = rapidfuzz.process.extractOne(benchmarkType, all_types, scorer=fuzz.partial_ratio, score_cutoff=80)
            if best:
                resolved["benchmarkType"] = best[0]
                resolved["benchmarkType_is_fuzzy"] = True
                
    if benchmarkCategory:
        # Match against primary data.value (which is the grouping category for Benchmark type)
        all_categories = list(set([b["data"].get("value") for b in benchmarks if b["data"].get("value")]))
        if all_categories:
            best = rapidfuzz.process.extractOne(benchmarkCategory, all_categories, scorer=fuzz.partial_ratio, score_cutoff=80)
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
    "cpumodel": [
        "value", "family", "corecount", "cpus", "architecture", "sockets", 
        "threadspercore", "corespersocket", "cpumaxmhz", "l3cache", "microcode", 
        "l3cacheinstances", "model", "num_l3", "ccdcount", "coreperccd", 
        "microarchitecture", "technology", "peakperformance"
    ],
    "instancetype": [
        "value", "cpumodel", "cloudprovider", "family", "corecount", "cpus", 
        "architecture", "sockets", "threadspercore", "corespersocket", 
        "cpumaxmhz", "l3cache", "microcode", "l3cacheinstances", "model", 
        "num_l3", "ccdcount", "coreperccd", "microarchitecture", "technology", 
        "peakperformance"
    ],
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
    print(f"\n[DEBUG] === GET /invalid-records ===")
    print(f"[DEBUG] page={page}, size={size}, search={search!r}, status={status!r}")
    
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
        invalid_fields = [p.get("field") for p in invalid_payloads if p.get("field")]
        
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
    print(f"\n[DEBUG] === GET /invalid-summary/counts ===")
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
            delta_days = (now - dt).days
            
            if delta_days < 3:
                counts["green"] += 1
            elif 3 <= delta_days <= 6:
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
    print(f"\n[DEBUG] === GET /invalid-summary ===")
    print(f"[DEBUG] search={search!r}, status={status!r}, page={page}, size={size}")
    
    # Normalize parameters for programmatic calls
    p_num = page if isinstance(page, int) else 1
    s_size = size if isinstance(size, int) else 50
    status_str = status if isinstance(status, str) else None
    search_str = search if isinstance(search, str) else None
    
    # 1. Primary Filter on Snapshot collection
    if status_str:
        status_filter = status_str.upper()
        match_query = {"data.standardization_status": {"$regex": f"^{status_filter}$", "$options": "i"}}
    else:
        # Default: Include all high-level statuses
        status_filter = "ALL"
        match_query = {"data.standardization_status": {"$in": ["PENDING", "REJECTED", "On Hold", "ACCEPTED"]}}
    
    print(f"API Executing Match Query: {match_query}")
    
    if search_str:
        search_regex = {"$regex": search_str, "$options": "i"}
        # Fuzzy match on Type/Category
        resolved = await resolve_fuzzy_benchmarks(benchmarkType=search_str, benchmarkCategory=search_str)
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

    skip_count = (p_num - 1) * s_size
    
    # 2. Extract Data (Use $lookup to join with ExecutionInfo for guaranteed metadata)
    invalid_records = []
    pipeline = [
        {"$match": match_query},
        {"$sort": {"_id": -1}},
        {"$skip": skip_count},
        {"$limit": s_size},
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
        
        # Aggregating all invalid fields (Primary + Metadata)
        all_invalid_fields = []
        for val_entry in first_data.get("invalidValues", []):
            field_name = val_entry.get("field")
            if field_name and val_entry.get("validation_status") != "valid":
                all_invalid_fields.append(field_name)
            
            # Check nested metadata
            for m in val_entry.get("metadata", []):
                m_name = m.get("name")
                if m_name and m.get("validation_status") != "valid":
                    all_invalid_fields.append(m_name)
        
        record["InvalidFields"] = sorted(list(set(all_invalid_fields)))
        record["updatedOn"] = first_data.get("history", {}).get("updatedOn")
            
        invalid_records.append(record)

    # 3. Total Count Logic
    if search_str or status_filter != "PENDING":
        total_records = await db[SNAPSHOT_COL].count_documents(match_query)
    else:
        global _report_cache
        if _report_cache["total_invalid"]["value"] is None or (time.time() - _report_cache["total_invalid"]["updated_at"]) > CACHE_TTL:
            _report_cache["total_invalid"]["value"] = await db[SNAPSHOT_COL].count_documents({"data.standardization_status": "PENDING"})
            _report_cache["total_invalid"]["updated_at"] = time.time()
        total_records = _report_cache["total_invalid"]["value"]
    
    # 4. Aging Counts (Red/Yellow/Green)
    counts = {"red": 0, "yellow": 0, "green": 0}
    if (search_str and search_str.strip()) or status_filter != "ALL":
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
                    "$floor": {
                        "$divide": [{"$subtract": ["$now", "$dt"]}, 86400000]
                    }
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
        "page": p_num,
        "size": s_size,
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
    
    # Ensure standard benchmark keys are present for the dashboard
    if "BenchmarkType" not in all_params and "benchmarkType" not in all_params:
        all_params.append("BenchmarkType")
    if "BenchmarkCategory" not in all_params and "benchmarkCategory" not in all_params:
        all_params.append("BenchmarkCategory")
        
    all_types = await db[MASTERLIST_COL].distinct("type", {"status": "Published"})
    type_set = set(all_types)
    
    unique_data = {}
    for param in all_params:
        param_norm = param.lower()
        
        # 1. Determine the query field and type
        if param_norm == "benchmarkcategory":
            # For category, we query the metadata of 'benchmarkType' records
            values = await db[MASTERLIST_COL].distinct("data.metadata.benchmarkCategory", {"type": "benchmarkType", "status": "Published"})
        elif param_norm == "benchmarktype" or param == "benchmarkType":
            # For type, we query the primary value of 'benchmarkType' records
            values = await db[MASTERLIST_COL].distinct("data.value", {"type": "benchmarkType", "status": "Published"})
        elif param in type_set:
            values = await db[MASTERLIST_COL].distinct("data.value", {"type": param, "status": "Published"})
        else:
            # Metadata fields (standardize casing for common ones)
            meta_path = param
            if param_norm == "cloudprovider": meta_path = "cloudProvider"
            
            values = await db[MASTERLIST_COL].distinct(f"data.metadata.{meta_path}", {"status": "Published"})
            
        # 2. Normalize to string and strip to remove duplicates
        clean_values = sorted(list(set(str(v).strip() for v in values if v is not None and v != "")))
        
        # 3. Use standardized keys for the response
        res_key = param
        if param_norm == "benchmarktype": res_key = "BenchmarkType"
        if param_norm == "benchmarkcategory": res_key = "BenchmarkCategory"
        
        unique_data[res_key] = clean_values
    
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
    print(f"\n[DEBUG] === GET /metadata-values/{type_name}/{value} ===")
    print(f"[DEBUG] type_name={type_name!r}, value={value!r}")
    
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
    print(f"\n[DEBUG] === GET /validation-counts ===")
    if _report_cache["counts_metrics"]["value"] is not None and (time.time() - _report_cache["counts_metrics"]["updated_at"]) <= CACHE_TTL:
        return _report_cache["counts_metrics"]["value"]
        
    db = get_db()
    mappings = await build_mappings()
    
    facet_dict: Dict[str, Any] = {}
    for t in mappings.keys():
        facet_dict[t] = [
            {"$group": {
                "_id": None,
                "valid":   {"$sum": {"$cond": [{"$in": [t, "$invalidPayload.field"]}, 0, 1]}},
                "invalid": {"$sum": {"$cond": [{"$in": [t, "$invalidPayload.field"]}, 1, 0]}},
            }}
        ]
        
    facet_dict["total_docs"] = [{"$count": "total"}]
    
    pipeline = [
        {"$group": {
            "_id": "$benchmarkExecutionID",
            "invalidPayload": {"$first": "$invalidPayload"}
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
    print(f"\n[DEBUG] === Get Snapshot Records Started ===")
    print(f"[DEBUG] Execution ID: {Execution_id}")

    db = get_db()
    
    # 1. Fetch Snapshot
    doc = await db[SNAPSHOT_COL].find_one({"execution_id": Execution_id})
    if not doc or not doc.get("data"):
        print(f"[DEBUG] ERROR: No snapshot record found for Execution_id: {Execution_id}")
        return {
            "status": "error",
            "message": f"No snapshot record found for Execution_id: {Execution_id}"
        }
    print(f"[DEBUG] Snapshot record retrieved from {SNAPSHOT_COL}")

    # 2. Fetch metadata from ExecutionInfo
    exec_meta = await db[EXECUTION_INFO_COL].find_one({"benchmarkExecutionID": Execution_id})
    if not exec_meta:
        print(f"[DEBUG] WARNING: No metadata found in {EXECUTION_INFO_COL} for {Execution_id}. Using empty fallback.")
        exec_meta = {}
    else:
        print(f"[DEBUG] Execution metadata retrieved from {EXECUTION_INFO_COL}")

    item = doc["data"][0]
    validator = await get_validator()
    
    # 3. Live-evaluate current validity
    if exec_meta:
        print(f"[DEBUG] Performing live validation check against current Masterlist rules...")
        invalid_payload, field_status = await validator.validate_doc(db, exec_meta)
        if len(invalid_payload) == 0:
            print(f"[DEBUG] Record is now VALID. Syncing status for {Execution_id}")
            # Ensure ExecutionInfo is marked as valid natively
            await db[EXECUTION_INFO_COL].update_one(
                {"benchmarkExecutionID": Execution_id},
                {
                    "$set": {"isValid": True, "invalidPayload": []},
                    "$unset": {"fieldStatus": ""}
                }
            )
            # Update snapshot status to ACCEPTED if it wasn't already
            if item.get("standardization_status") not in ("ACCEPTED",):
                print(f"[DEBUG] Updating snapshot status to ACCEPTED as live validation passed.")
                doc["data"][0]["standardization_status"] = "ACCEPTED"
                await db[SNAPSHOT_COL].replace_one({"execution_id": Execution_id}, doc)
                item = doc["data"][0] # Refresh local item
        else:
            print(f"[DEBUG] Record remains invalid with {len(invalid_payload)} errors.")
            
    # 4. Fetch mapping and validation details dynamically for EACH individual field
    type_mappings = {}
    data_list = []
    
    invalid_fields = item.get("invalidValues", [])
    print(f"[DEBUG] Processing {len(invalid_fields)} invalid primary fields for UI display...")

    for meta in invalid_fields:
        field_name = meta.get("field")
        if not field_name:
            continue
            
        if field_name not in type_mappings:
            type_mappings[field_name] = await get_masterlist_mappings(field_name)
            
        val = meta.get("value")
        print(f"[DEBUG]   - Processing field: {field_name} (Value: {val})")
        
        # Build existing_data for this field (primary field + metadata)
        field_existing_data = []
        field_existing_data.append({
            "field": field_name,
            "value": val,
            "validation_status": meta.get("validation_status"),
            "history": meta.get("history", [])
        })
        
        for support in meta.get("metadata", []):
            field_existing_data.append({
                "field": support.get("name"),
                "value": support.get("value"),
                "validation_status": support.get("validation_status"),
                "history": support.get("history", [])
            })
            
        # Build suggestions for this field (grouped by masterlist record)
        actual_meta_vals = {s.get("name"): s.get("value", "") for s in meta.get("metadata", []) if s.get("name")}
        
        # If overwritten to L0 or On Hold, we must supply the original string to the fuzzy matcher
        sug_val = val
        if val in ("L0", "On Hold"):
            hist = meta.get("history", [])
            if hist:
                sug_val = hist[0].get("from", val)
            
            for s in meta.get("metadata", []):
                m_name = s.get("name")
                if m_name and s.get("value") in ("L0", "On Hold"):
                    m_hist = s.get("history", [])
                    if m_hist:
                        actual_meta_vals[m_name] = m_hist[0].get("from", val)
                        
        record_suggestions = validator.get_record_level_suggestions(field_name, sug_val, actual_meta_vals)
        
        field_suggestions = []
        for rec_sug in record_suggestions:
            sug_status = "PENDING"
            if meta.get("currentStatus") in ("L0 Data", "On Hold") or item.get("standardization_status") == "REJECTED":
                sug_status = "Rejected"
            elif meta.get("validation_status") == "Accepted" and str(rec_sug["primary_value"]).strip() == str(val).strip():
                sug_status = "Accepted"
                
            sug_entry = {
                field_name.lower(): rec_sug["primary_value"],
                "score": rec_sug.get("score", 0),
                "status": sug_status
            }
            for m_name, m_val in rec_sug["metadata"].items():
                sug_entry[m_name.lower()] = m_val
            field_suggestions.append(sug_entry)
            
        print(f"[DEBUG] Field '{field_name}': {len(field_suggestions)} suggestions found (val={sug_val!r})")
        
        entry = {
            "invalid_field": field_name,
            "currentStatus": meta.get("currentStatus") or "invalid",
            "existing_data": field_existing_data,
        }
        if field_suggestions:
            entry["suggestions"] = field_suggestions
            
        data_list.append(entry)
    

        
    # 5. Reconstruct top-level history from field-level history arrays
    changes_by_field = {}
    last_updated_on = None
    last_updated_by = None

    for inv_item in invalid_fields:
        field = inv_item.get("field", "")
        for h in inv_item.get("history", []):
            if field not in changes_by_field:
                changes_by_field[field] = {"field": field, "from": [], "to": []}
            changes_by_field[field]["from"].append(h.get("from", ""))
            changes_by_field[field]["to"].append(h.get("to", ""))
            last_updated_on = h.get("updatedOn", last_updated_on)
            last_updated_by = h.get("updatedBy", last_updated_by)

        for meta in inv_item.get("metadata", []):
            m_name = meta.get("name", "")
            for h in meta.get("history", []):
                if m_name not in changes_by_field:
                    changes_by_field[m_name] = {"field": m_name, "from": [], "to": []}
                changes_by_field[m_name]["from"].append(h.get("from", ""))
                changes_by_field[m_name]["to"].append(h.get("to", ""))
                last_updated_on = h.get("updatedOn", last_updated_on)
                last_updated_by = h.get("updatedBy", last_updated_by)

    # Get execution info for detailed response
    sut_type = exec_meta.get("sutInstanceMetadata.sutType") if "sutInstanceMetadata.sutType" in exec_meta else exec_meta.get("sutInstanceMetadata", {}).get("sutType")
    
    print(f"[DEBUG] === Get Snapshot Records Completed Successfully ===\n")
    return {
        "status": "success",
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
            "updatedOn": last_updated_on,
            "updatedBy": last_updated_by,
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
    print(f"\n[DEBUG] === GET /unique-values ===")
    print(f"[DEBUG] parameterName={parameterName!r}")
    
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
            "benchmarktype":     {"type": "benchmarkType", "path": "data.value"},
            "benchmarkcategory": {"type": "benchmarkType", "path": "data.metadata.benchmarkCategory"},
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
    print(f"\n[DEBUG] === GET /search-snapshots ===")
    print(f"[DEBUG] status={status!r}, benchmarkType={benchmarkType!r}, benchmarkCategory={benchmarkCategory!r}, search={search!r}, page={page}, size={size}")
    
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
    print(f"\n[DEBUG] === Approve Suggestion Started ===")
    print(f"[DEBUG] Received Approve Suggestion for Execution {req.execution_id}, Field: {req.field_name}, Value: {req.accepted_value}, Status: {req.currentStatus}")
    print(f"[DEBUG] Accepted Value: {req.accepted_value}")

    db = get_db()
    
    # 1. Fetch Snapshot
    snap = await db[SNAPSHOT_COL].find_one({"execution_id": req.execution_id})
    if not snap or not snap.get("data"):
        print(f"[DEBUG] ERROR: Snapshot not found for {req.execution_id}")
        return {"status": "error", "message": f"Snapshot not found for Execution ID: {req.execution_id}"}
    
    snap_data = snap["data"][0]
    invalid_values = snap_data.get("invalidValues", [])
    print(f"[DEBUG] Snapshot found. Contains {len(invalid_values)} top-level invalid fields.")
    
    # 2. Identify the selected field and suggestion (supporting nested metadata)
    target_item = None
    target_mapping = None
    original_value = None
    suggestion_found = False
    
    req_field_norm = req.field_name.lower()
    print(f"[DEBUG] Searching for field '{req_field_norm}' (normalized) in invalid values...")
    
    # Check top-level invalid fields first
    for item in invalid_values:
        curr_field = (item.get("field") or "").lower()
        if curr_field == req_field_norm:
            target_item = item
            original_value = item.get("value")
            # Prefer mapping from snapshot, fallback to Masterlist discovery
            target_mapping = item.get("mapping")
            if not target_mapping:
                m_info = await get_masterlist_mappings(item.get("field")) # Use original case for lookup
                target_mapping = m_info.get("mapping")
            print(f"[DEBUG] Found primary field match: {item.get('field')} (Path: {target_mapping})")
            break
        
        # Check nested metadata fields
        for meta_item in item.get("metadata", []):
            curr_meta = (meta_item.get("name") or "").lower()
            if curr_meta == req_field_norm:
                target_item = meta_item
                original_value = meta_item.get("value")
                # Prefer mapping from snapshot, fallback to Masterlist discovery
                target_mapping = meta_item.get("mapping")
                if not target_mapping:
                    m_info = await get_masterlist_mappings(item.get("field"))
                    target_mapping = m_info.get("metadata_mappings", {}).get(meta_item.get("name"))
                print(f"[DEBUG] Found nested metadata match: {meta_item.get('name')} (Path: {target_mapping})")
                break
        if target_item: break

    if not target_item:
        print(f"[DEBUG] ERROR: Field '{req.field_name}' not found in current snapshot data.")
        return {"status": "error", "message": f"Field '{req.field_name}' not found in snapshot."}
        
    # 3. Update suggestion statuses
    comparing_data = target_item.get("comparingData", [])
    accepted_sug_num = None
    print(f"[DEBUG] Checking {len(comparing_data)} suggestions for a match...")
    for sug in comparing_data:
        match_key = next((k for k in sug.keys() if k.startswith("suggestion")), None)
        if match_key and str(sug[match_key]).strip() == str(req.accepted_value).strip():
            sug["status"] = "Accepted"
            suggestion_found = True
            accepted_sug_num = str(match_key).replace("suggestion", "")
            print(f"[DEBUG] Suggestion matched! ID: {match_key}")
        else:
            sug["status"] = "Rejected"
    
    # Determine the source of the accepted value
    value_source = "suggestion" if suggestion_found else "dropdown"
    if not suggestion_found:
        print(f"[DEBUG] No suggestion matched. Treating as manual dropdown entry.")
            
    # Always set to Accepted since we are finalizing a correction
    target_item["validation_status"] = "Accepted"
    target_item["currentStatus"] = req.currentStatus
    target_item["value"] = req.accepted_value
    
    if "history" not in target_item:
        target_item["history"] = []
    target_item["history"].append({
        "from": original_value,
        "to": req.accepted_value,
        "source": value_source,
        "updatedOn": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "updatedBy": "xxx@amd.com"
    })
    
    print(f"[DEBUG] Target field status set to 'Accepted' and currentStatus to '{req.currentStatus}'")
        
    # 4. Propagate the change to Executioninfo if a mapping exists
    if target_mapping:
        print(f"[DEBUG] Propagating value '{req.accepted_value}' to DB path: {target_mapping}")
        await db[EXECUTION_INFO_COL].update_one(
            {"benchmarkExecutionID": req.execution_id},
            {"$set": {target_mapping: req.accepted_value}}
        )
    else:
        print(f"[DEBUG] WARNING: No DB mapping path found for field '{req.field_name}'. Propagation skipped.")
    
    # 4b. CASCADE: If this is a primary field, apply the same suggestion status to ALL metadata
    is_primary_field = False
    parent_item = None
    cascaded_changes = []  # Track (field_name, from_value, to_value) for history
    
    for item in invalid_values:
        if (item.get("field") or "").lower() == req_field_norm:
            is_primary_field = True
            parent_item = item
            break
    
    if is_primary_field and parent_item:
        print(f"[DEBUG] Field is primary. Starting cascade to metadata fields...")
        # Fetch metadata mappings for this primary field type if not in snapshot
        m_info = None
        
        for meta_item in parent_item.get("metadata", []):
            meta_name = meta_item.get("name")
            meta_original_value = meta_item.get("value")
            meta_accepted_value = None
            meta_comparing = meta_item.get("comparingData", [])
            
            if accepted_sug_num:
                # Suggestion case: accept the same suggestion number, reject others
                for sug in meta_comparing:
                    sug_key = f"suggestion{accepted_sug_num}"
                    if sug_key in sug:
                        sug["status"] = "Accepted"
                        meta_accepted_value = sug.get(sug_key)
                    else:
                        sug["status"] = "Rejected"
            else:
                # Dropdown case: reject all metadata suggestions
                for sug in meta_comparing:
                    sug["status"] = "Rejected"
            
            # Mark metadata as Accepted and update the value field in the snapshot
            meta_item["validation_status"] = "Accepted"
            meta_item["currentStatus"] = req.currentStatus
            if meta_accepted_value is not None:
                meta_item["value"] = meta_accepted_value
                
                if "history" not in meta_item:
                    meta_item["history"] = []
                meta_item["history"].append({
                    "from": meta_original_value,
                    "to": meta_accepted_value,
                    "source": value_source,
                    "updatedOn": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "updatedBy": "xxx@amd.com"
                })
            
            # Propagate metadata value to ExecutionInfo if we have a mapping
            # Try snapshot mapping first, then Masterlist
            meta_mapping = meta_item.get("mapping")
            if not meta_mapping:
                if not m_info: m_info = await get_masterlist_mappings(parent_item.get("field"))
                meta_mapping = m_info.get("metadata_mappings", {}).get(meta_name)
                
            if meta_mapping and meta_accepted_value is not None:
                print(f"[DEBUG] Cascading to metadata '{meta_name}': Setting to '{meta_accepted_value}' at {meta_mapping}")
                await db[EXECUTION_INFO_COL].update_one(
                    {"benchmarkExecutionID": req.execution_id},
                    {"$set": {meta_mapping: meta_accepted_value}}
                )
            
            # Track for history
            cascaded_changes.append({
                "field": meta_name,
                "from": meta_original_value,
                "to": meta_accepted_value or meta_original_value
            })
        print(f"[DEBUG] Cascade complete. {len(cascaded_changes)} metadata fields updated.")
    
    # 5. Check for Overall Validity (Standardization Status)
    is_fully_resolved = True
    for item in invalid_values:
        if item.get("validation_status") not in ("valid", "Accepted"):
            is_fully_resolved = False
            print(f"[DEBUG] UNRESOLVED: Field '{item.get('field')}' is still '{item.get('validation_status')}'")
            break
        for meta in item.get("metadata", []):
            if meta.get("validation_status") not in ("valid", "Accepted"):
                is_fully_resolved = False
                print(f"[DEBUG] UNRESOLVED METADATA: Field '{meta.get('name')}' (Under {item.get('field')}) is still '{meta.get('validation_status')}'")
                break
        if not is_fully_resolved: break

    print(f"[DEBUG] Final resolution check: {'SUCCESS' if is_fully_resolved else 'STILL PENDING'}")

    # 6. Update History (Top level history removed)
    if "history" in snap_data:
        del snap_data["history"]

    # 7. Final Acceptance Transition & Consistency Sync
    if is_fully_resolved:
        print(f"[DEBUG] Transitioning record status to ACCEPTED.")
        snap_data["standardization_status"] = "ACCEPTED"
        
        # --- FINAL CONSISTENCY SYNC (Double-Sync driven by History) ---
        ei_doc = await db[EXECUTION_INFO_COL].find_one({"benchmarkExecutionID": req.execution_id})
        
        if ei_doc:
            final_updates_applied = False
            
            # Sync ALL fields currently present in the final history to Executioninfo
            for item in invalid_values:
                hist = item.get("history", [])
                if hist:
                    latest = hist[-1]
                    m_path = item.get("mapping")
                    m_val = latest.get("to")
                    
                    if m_path and m_val is not None:
                        final_updates_applied = True
                        if m_path in ei_doc: ei_doc[m_path] = m_val
                        _set_nested_key(ei_doc, m_path, m_val)
                        
                for meta in item.get("metadata", []):
                    m_hist = meta.get("history", [])
                    if m_hist:
                        latest = m_hist[-1]
                        m_path = meta.get("mapping")
                        m_val = latest.get("to")
                        
                        if m_path and m_val is not None:
                            final_updates_applied = True
                            if m_path in ei_doc: ei_doc[m_path] = m_val
                            _set_nested_key(ei_doc, m_path, m_val)
            
            if final_updates_applied:
                print(f"[DEBUG] Applying Final History-Driven Sync for {req.execution_id}")
                await db[EXECUTION_INFO_COL].replace_one(
                    {"_id": ei_doc["_id"]},
                    ei_doc
                )
        
        # Mark Executioninfo as valid ONLY if fully resolved
        await db[EXECUTION_INFO_COL].update_one(
            {"benchmarkExecutionID": req.execution_id},
            {"$set": {"isValid": True}}
        )
    else:
        # Keep as PENDING if not all fields are resolved
        snap_data["standardization_status"] = "PENDING"

    # 8. Save entire Snapshot back
    await db[SNAPSHOT_COL].replace_one({"execution_id": req.execution_id}, snap)
    
    # 9. Fetch the fully updated snapshot to return to the frontend
    updated_snapshot_data = await get_snapshot_records(req.execution_id)
    
    print(f"[DEBUG] === Approve Suggestion Completed Successfully ===\n")
    return {
        "status": "success",
        "message": f"Successfully {'accepted suggestion' if value_source == 'suggestion' else 'applied custom value'} for '{req.field_name}' and updated Executioninfo.",
        "execution_id": req.execution_id,
        "updated_field": req.field_name,
        "accepted_value": req.accepted_value,
        "mapping_path": target_mapping,
        "value_source": value_source,
        "field_status": "Accepted",
        "standardization_status": snap_data.get("standardization_status", "PENDING"),
        "snapshot_data": updated_snapshot_data
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
    print(f"\n[DEBUG] === PUT /reject-record ===")
    print(f"[DEBUG] execution_id={execution_id!r}, currentStatus={req.currentStatus!r}")
    action_value = "L0"
    action_status = "L0 Data"
    
    # 1. Fetch Snapshot
    snap = await db[SNAPSHOT_COL].find_one({"execution_id": execution_id})
    if not snap or not snap.get("data"):
        return {"status": "error", "message": f"Snapshot not found for Execution ID: {execution_id}"}
    
    snap_data = snap["data"][0]
    invalid_values = snap_data.get("invalidValues", [])
    
    # 2. Mark all suggestions for all fields and metadata as "Rejected" and record history
    for item in invalid_values:
        val = item.get("value", "")
        item["value"] = action_value
        item["currentStatus"] = action_status
        
        # Track for history
        if "history" not in item:
            item["history"] = []
        item["history"].append({
            "from": val,
            "to": action_value,
            "source": "reject",
            "updatedOn": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "updatedBy": "xxx@amd.com"
        })
        
        # Reject primary suggestions
        for sug in item.get("comparingData", []):
            sug["status"] = "Rejected"
            
        for meta in item.get("metadata", []):
            meta_val = meta.get("value", "")
            meta["value"] = action_value
            meta["currentStatus"] = action_status
            if "history" not in meta:
                meta["history"] = []
            meta["history"].append({
                "from": meta_val,
                "to": action_value,
                "source": "reject",
                "updatedOn": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "updatedBy": "xxx@amd.com"
            })
            
            for sug in meta.get("comparingData", []):
                sug["status"] = "Rejected"
    
    # 3. Transition Standardization Status
    snap_data["standardization_status"] = "REJECTED"
    snap_data["reason"] = "L0 Junk Data."
    
    # 4. Remove Top-Level History (if exists)
    if "history" in snap_data:
        del snap_data["history"]
    
    # 5. Update Executioninfo to include the entitliment_level and applied updates
    ei_doc = await db[EXECUTION_INFO_COL].find_one({"benchmarkExecutionID": execution_id})
    if ei_doc:
        final_updates_applied = False
        ei_doc["entitliment_level"] = action_status
        
        for item in invalid_values:
            hist = item.get("history", [])
            if hist:
                latest = hist[-1]
                m_path = item.get("mapping")
                m_val = latest.get("to")
                
                if m_path and m_val is not None:
                    final_updates_applied = True
                    if m_path in ei_doc: ei_doc[m_path] = m_val
                    _set_nested_key(ei_doc, m_path, m_val)
                    
            for meta in item.get("metadata", []):
                m_hist = meta.get("history", [])
                if m_hist:
                    latest = m_hist[-1]
                    m_path = meta.get("mapping")
                    m_val = latest.get("to")
                    
                    if m_path and m_val is not None:
                        final_updates_applied = True
                        if m_path in ei_doc: ei_doc[m_path] = m_val
                        _set_nested_key(ei_doc, m_path, m_val)
        
        await db[EXECUTION_INFO_COL].replace_one(
            {"_id": ei_doc["_id"]},
            ei_doc
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
    print(f"\n[DEBUG] === GET /draft-executions ===")
    
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
    print(f"\n[DEBUG] === POST /draft-records/{type_name} ===")
    print(f"[DEBUG] type_name={type_name!r}, execution_id={draft.execution_id!r}, value={draft.value!r}")
    print(f"[DEBUG] Full draft payload: {draft.dict()}")
    
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
    existing = await _check_duplicate(db, actual_type, value)
    if existing:
        return {
            "status": "error",
            "message": f"A record for {actual_type} '{value}' already exists in the masterlist (Status: {existing['status']})."
        }
        
    # 2. Build type-specific data content
    data_content = {}
    
    if type_norm == "cpumodel":
        data_content = {
            "sutType": "server",
            "mapping_sutType": "sutInstanceMetadata.sutType",
            "value": value,
            "mapping": "platformProfile.sut.Summary.Server.CPUModel",
            "metadata": _populate_metadata_content(draft, type_norm)
        }
    elif type_norm == "instancetype":
        data_content = {
            "sutType": "cloud",
            "mapping_sutType": "sutInstanceMetadata.sutType",
            "value": value,
            "mapping": "sutInstanceMetadata.instanceType",
            "metadata": _populate_metadata_content(draft, type_norm)
        }
    elif type_norm == "benchmark":
        data_content = {
            "value": value,
            "mapping": "benchmarkCategory",
            "metadata": _populate_metadata_content(draft, type_norm)
        }
        
    # 3. Final Document Construction (Type before Status, Data before History)
    ml_doc = _build_base_ml_doc(actual_type, data_content, "")
    
    await db[MASTERLIST_COL].insert_one(ml_doc)
    
    # 4. Place snapshots containing this drafted value "On Hold" — only update currentStatus
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    snap_filter = {"execution_id": draft.execution_id} if draft.execution_id else {"data.0.invalidValues.value": value}
    
    affected_snaps = await db[SNAPSHOT_COL].find(snap_filter).to_list(None)
    
    for snap in affected_snaps:
        snap_data = snap.get("data", [{}])[0]
        invalid_values = snap_data.get("invalidValues", [])
        
        for item in invalid_values:
            # Only target the item matching the drafted value
            if str(item.get("value", "")).strip().lower() != str(value).strip().lower():
                continue
            
            # Only update currentStatus, do not change value or metadata values
            prev_status = item.get("currentStatus", "invalid")
            item["currentStatus"] = "On Hold"
            
            if "history" not in item:
                item["history"] = []
            item["history"].append({
                "from": prev_status,
                "to": "On Hold",
                "source": "draft",
                "updatedOn": now_str,
                "updatedBy": "xxx@amd.com"
            })
            
            # Cascade currentStatus only to metadata fields
            for meta in item.get("metadata", []):
                prev_meta_status = meta.get("currentStatus", "invalid")
                meta["currentStatus"] = "On Hold"
                
                if "history" not in meta:
                    meta["history"] = []
                meta["history"].append({
                    "from": prev_meta_status,
                    "to": "On Hold",
                    "source": "draft",
                    "updatedOn": now_str,
                    "updatedBy": "xxx@amd.com"
                })
        
        snap_data["standardization_status"] = "On Hold"
        snap_data["reason"] = "New Masterlist Draft Record."
        
        # Save updated snapshot only
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
