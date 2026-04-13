from typing import Dict, Set, List, Tuple, Any, Optional
from database import MASTERLIST_COL

def get_nested_value(doc: Dict[str, Any], keys: str) -> Any:
    """Retrieve a nested value from dictionary using dot-separated keys, or fallback to exact match."""
    if keys in doc:
        return doc[keys]
        
    k_list = keys.split('.')
    val = doc
    for k in k_list:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return None
    return val

async def fetch_valid_values_and_meta(db: Any, mappings: Dict[str, str]) -> Any:
    valid_values: Dict[str, Any] = {}
    meta_mappings: Dict[str, Any] = {}
    
    for t in mappings.keys():
        docs = await db[MASTERLIST_COL].find({"type": t, "status": "Published"}).to_list(length=None)
        values = set()
        m_maps = {}
        for doc in docs:
            if 'data' in doc:
                data = doc['data']
                if 'value' in data:
                    val = data['value']
                    if isinstance(val, (list, dict)):
                        val = str(val)
                    values.add(val)
                
                # Dynamic metadata extraction based on masterlist payload schema
                for k, v in data.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if str(sub_k).startswith("mapping_"):
                                meta_key = str(sub_k).replace("mapping_", "")
                                m_maps[meta_key] = sub_v
                        
                        if k == "metadata" and "mapping" in v:
                            for mk in v.keys():
                                if mk != "mapping":
                                    m_maps[mk] = v["mapping"]
        valid_values[t] = values
        meta_mappings[t] = m_maps
    
    return valid_values, meta_mappings

async def get_metadata_schema(db: Any, type_name: str) -> List[str]:
    """
    Dynamically identifies the metadata fields required for a specific masterlist type.
    It looks at a representative Published document for that type.
    """
    doc = await db[MASTERLIST_COL].find_one({"type": type_name, "status": "Published"})
    if not doc or 'data' not in doc:
        # Try case-insensitive fallback if exact match fails
        doc = await db[MASTERLIST_COL].find_one({"type": {"$regex": f"^{type_name}$", "$options": "i"}, "status": "Published"})
        if not doc or 'data' not in doc:
            return []
            
    data = doc['data']
    metadata_keys = set()
    
    # Strategy 1: Check the data.metadata dictionary
    meta_obj = data.get("metadata", {})
    if isinstance(meta_obj, dict):
        for k in meta_obj.keys():
            if not k.startswith("mapping") and k != "value":
                metadata_keys.add(k)
                
    # Strategy 2: Check for top-level data keys that have a mapping_ counterpart
    for k in data.keys():
        if f"mapping_{k}" in data:
            metadata_keys.add(k)
            
    return sorted(list(metadata_keys))
