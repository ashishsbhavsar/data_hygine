import re
import rapidfuzz
import asyncio
import concurrent.futures
import numpy as np
from rapidfuzz import process, fuzz
from typing import Dict, Any, Tuple, List, Set, Optional
from utils import get_nested_value
from database import get_db, MASTERLIST_COL

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
 
 
# Conditional validation rules: fields that only apply when sutType matches a condition
# Format: {masterlist_type: {"condition": "equals"|"not_equals", "value": "cloud"}}
# If a type is NOT listed here, it is validated unconditionally (for all records).
CONDITIONAL_RULES = {
    "CPUModel": {"field": "sutInstanceMetadata.sutType", "condition": "not_equals", "value": "cloud"},
    "instanceType": {"field": "sutInstanceMetadata.sutType", "condition": "equals", "value": "cloud"}
}

async def build_mappings() -> Dict[str, str]:
    """
    Dynamically builds the MAPPINGS dict from the masterlist collection.
    Returns: {"CPUModel": "platformProfile.sut.Summary.Server.CPUModel", ...}
    """
    db = get_db()
    # 1. Discover top-level types and primary mappings
    pipeline = [
        {"$match": {"status": "Published"}},
        {"$group": {
            "_id": "$type",
            "mapping": {"$first": "$data.mapping"}
        }}
    ]
    mappings = {}
    async for doc in db[MASTERLIST_COL].aggregate(pipeline):
        ml_type = doc["_id"]
        mapping_path = doc.get("mapping")
       
        if ml_type and mapping_path:
            mappings[ml_type] = mapping_path
 
    # 2. Discover metadata-level mappings (e.g., BenchmarkType mapping inside Benchmark)
    # This allows the API to see them as distinct parameters even if they're embedded.
    cursor = db[MASTERLIST_COL].find({"status": "Published", "data.metadata": {"$exists": True}})
    async for doc in cursor:
        meta = doc.get("data", {}).get("metadata", {})
        if not isinstance(meta, dict):
            continue
           
        for k, v in meta.items():
            param_name = k
            if k.startswith("mapping_"):
                param_name = k.replace("mapping_", "")
           
            # Standardization of dash labels
            if param_name.lower() == "benchmarktype":
                param_name = "BenchmarkType"
            if param_name.lower() == "cloudprovider":
                param_name = "cloudProvider"
               
            if param_name not in mappings:
                # Store the value as the path if it started with mapping_
                mappings[param_name] = str(v) if k.startswith("mapping_") else ""
   
    return mappings
 
 
class Validator:
    def __init__(self, ml_records, mappings: Dict[str, str]):
        self.mappings = mappings
       
        self.valid_values: Dict[str, Set[str]] = {t: set() for t in mappings}
        self.value_ids: Dict[str, Dict[str, str]] = {t: {} for t in mappings}
        self.record_signatures: Dict[str, List[Dict]] = {t: [] for t in mappings}
        self.val_metadata_reqs: Dict[str, Dict[str, List[Dict]]] = {t: {} for t in mappings}
        self.all_metadata_values: Dict[str, Dict[str, str]] = {}
        self.type_metadata_paths: Dict[str, Dict[str, str]] = {t: {} for t in mappings}
       
        # Track which types are primary (explicitly defined in masterlist as type)
        self.primary_types: Set[str] = set()
       
        for record in ml_records:
            t = record.get("type")
            data = record.get("data", {})
            val = str(data.get("value", "")).strip()
           
            if t == "InstanceType":
                t = "instanceType"
           
            if not t:
                continue
               
            self.primary_types.add(t)
           
            if t not in self.valid_values:
                continue
           
            masterlist_id = str(record.get("`_id`") or record.get("id") or str(record.get("_id", "")))
            if isinstance(masterlist_id, dict) and "$oid" in masterlist_id:
                masterlist_id = masterlist_id["$oid"]

            if val:
                self.valid_values[t].add(val)
                self.value_ids[t][val] = masterlist_id
                if val not in self.val_metadata_reqs[t]:
                    self.val_metadata_reqs[t][val] = []
           
            meta_record = {}
            meta = data.get("metadata", {})
            signature_parts = [val] if val else []

            if isinstance(meta, dict):
                # We sort metadata keys alphabetically to ensure consistent signature generation
                for mk in sorted(meta.keys()):
                    if mk.startswith("mapping_") or mk == "mapping":
                        continue
                    
                    mv = meta[mk]
                    lookup_key = f"mapping_{mk}".lower()
                    meta_mapping_path = None
                   
                    for k, v in meta.items():
                        if k.lower() == lookup_key:
                            meta_mapping_path = v
                            break
                   
                    if not meta_mapping_path:
                        meta_mapping_path = meta.get("mapping", "")
                   
                    mv_str = str(mv).strip()
                    if mv_str:
                        meta_record[mk] = {"mapping": meta_mapping_path, "required_val": mv_str}
                        signature_parts.append(mv_str)
                       
                        if mk not in self.all_metadata_values:
                            self.all_metadata_values[mk] = {}
                        self.all_metadata_values[mk][mv_str] = masterlist_id
                       
                        if meta_mapping_path:
                            self.type_metadata_paths[t][mk] = meta_mapping_path
           
            # Create a Mega-String Signature for this specific configuration
            full_signature = " ".join(signature_parts).lower()
            
            if val:
                self.val_metadata_reqs[t][val].append((masterlist_id, meta_record))
                self.record_signatures[t].append({
                    "signature": full_signature,
                    "record_id": masterlist_id,
                    "primary_value": val,
                    "metadata": {mk: mv["required_val"] for mk, mv in meta_record.items()}
                })
        
        # Build ANN Indexes for high-performance suggestions
        self._ann_indexes: Dict[str, Any] = {}
        if HAS_SKLEARN:
            self._build_ann_indexes()

    def _build_ann_indexes(self):
        """Builds TF-IDF vectorizers and NearestNeighbors indexes in parallel."""
        if not HAS_SKLEARN:
            return

        def _fit_index(key, items, is_signature=False, configs=None):
            if not items:
                return None
            try:
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
                tfidf_matrix = vectorizer.fit_transform(items)
                nn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
                nn.fit(tfidf_matrix)
                
                result = {
                    "vectorizer": vectorizer,
                    "nn": nn,
                }
                if is_signature:
                    result["configs"] = configs
                else:
                    result["possibilities"] = items
                return key, result
            except Exception:
                return None

        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 1. Field-level indexes
            for t in self.mappings:
                possibilities = list(self.valid_values.get(t, set()))
                if not possibilities:
                    possibilities = list(self.all_metadata_values.get(t, {}).keys())
                tasks.append(executor.submit(_fit_index, t, possibilities))

            # 2. Record-level signature indexes
            for t, configs in self.record_signatures.items():
                signatures = [c["signature"] for c in configs]
                tasks.append(executor.submit(_fit_index, f"{t}_signatures", signatures, True, configs))

            # Collect results
            for future in concurrent.futures.as_completed(tasks):
                res = future.result()
                if res:
                    key, data = res
                    self._ann_indexes[key] = data
 
    def get_suggestions(self, field_type: str, value: str, n: int = 3) -> List[Dict[str, Any]]:
        is_metadata = field_type not in self.primary_types
        possibilities = list(self.valid_values.get(field_type, set()))
        if not possibilities and is_metadata:
            possibilities = list(self.all_metadata_values.get(field_type, {}).keys())
           
        if not possibilities or not value:
            return []
       
        matches = process.extract(value, possibilities, limit=n, scorer=fuzz.partial_ratio, score_cutoff=10)
        results = []
        for i, match_info in enumerate(matches, 1):
            match, score, _ = match_info
           
            if is_metadata:
                match_id = self.all_metadata_values.get(field_type, {}).get(match, "")
            else:
                match_id = self.value_ids.get(field_type, {}).get(match, "")
               
            results.append({
                f"suggestion{i}": match,
                f"score{i}": round(score / 100.0, 4),
                "status": "PENDING",
                "_id": match_id
            })
        return results
 
    async def get_suggestions_ann(self, field_type: str, value: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Asynchronous-ready Approximate Nearest Neighbor suggestion engine.
        Uses TF-IDF + Cosine Similarity to simulate partial_ratio behavior at scale.
        """
        if not HAS_SKLEARN or field_type not in self._ann_indexes or not value:
            return self.get_suggestions(field_type, value, n)

        index_data = self._ann_indexes[field_type]
        vectorizer = index_data["vectorizer"]
        nn = index_data["nn"]
        possibilities = index_data["possibilities"]

        def _search():
            query_vec = vectorizer.transform([value])
            distances, indices = nn.kneighbors(query_vec, n_neighbors=min(n, len(possibilities)))
            return distances[0], indices[0]

        # Run CPU-bound search in a thread to keep FastAPI responsive
        distances, indices = await asyncio.to_thread(_search)
        
        is_metadata = field_type not in self.primary_types
        results = []
        for i, (dist, idx) in enumerate(zip(distances, indices), 1):
            match = possibilities[idx]
            # Convert cosine distance to a 0-1 score (1 - dist)
            score = 1.0 - dist
            
            if is_metadata:
                match_id = self.all_metadata_values.get(field_type, {}).get(match, "")
            else:
                match_id = self.value_ids.get(field_type, {}).get(match, "")
                
            results.append({
                f"suggestion{i}": match,
                f"score{i}": round(float(score), 4),
                "status": "PENDING",
                "_id": match_id
            })
        return results
 
    def get_record_level_suggestions(self, field_type: str, value: str, actual_metadata: Dict[str, str] = None, n: int = 3) -> List[Dict[str, Any]]:
        """
        Returns top N record-level suggestions using 'Mega-String' concatenated matching.
        Combines primary value and metadata into a single string for high-accuracy fuzzy matching.
        """
        if actual_metadata is None:
            actual_metadata = {}
        
        type_configs = self.record_signatures.get(field_type, [])
        if not type_configs or (not value and not actual_metadata):
            return []
            
        # 1. Build the 'Mega-String' signature for our ACTUAL record
        # We include metadata values that exist to strengthen the search
        actual_signature_parts = [str(value).strip()]
        for m_name in sorted(actual_metadata.keys()):
            m_val = str(actual_metadata.get(m_name, "")).strip()
            if m_val and m_val.lower() != "nan":
                actual_signature_parts.append(m_val)
        
        actual_signature = " ".join(actual_signature_parts).lower()
        
        # 2. Extract signatures for matching
        signature_strings = [c["signature"] for c in type_configs]
        
        # 3. Perform Fuzzy Search using partial_ratio on Mega-Strings
        # This handles noisy names effectively by prioritizing metadata overlaps
        matches = process.extract(actual_signature, signature_strings, limit=n, scorer=fuzz.partial_ratio, score_cutoff=75)
        
        results = []
        for match_str, score, index in matches:
            config = type_configs[index]
            results.append({
                "_id": config["record_id"],
                "primary_value": config["primary_value"],
                "metadata": config["metadata"],
                "score": round(score / 100.0, 4)
            })
            
        return results

    async def get_record_level_suggestions_ann(self, field_type: str, value: str, actual_metadata: Dict[str, str] = None, n: int = 3) -> List[Dict[str, Any]]:
        """
        Record-level ANN suggestions using 'Mega-String' concatenated matching.
        Runs asynchronously in a thread pool.
        """
        if not HAS_SKLEARN or f"{field_type}_signatures" not in self._ann_indexes:
            return self.get_record_level_suggestions(field_type, value, actual_metadata, n)

        if actual_metadata is None:
            actual_metadata = {}
            
        # Build the 'Mega-String' signature
        actual_signature_parts = [str(value).strip()]
        for m_name in sorted(actual_metadata.keys()):
            m_val = str(actual_metadata.get(m_name, "")).strip()
            if m_val and m_val.lower() != "nan":
                actual_signature_parts.append(m_val)
        actual_signature = " ".join(actual_signature_parts).lower()

        index_data = self._ann_indexes[f"{field_type}_signatures"]
        vectorizer = index_data["vectorizer"]
        nn = index_data["nn"]
        configs = index_data["configs"]

        def _search():
            query_vec = vectorizer.transform([actual_signature])
            distances, indices = nn.kneighbors(query_vec, n_neighbors=min(n, len(configs)))
            return distances[0], indices[0]

        distances, indices = await asyncio.to_thread(_search)
        
        results = []
        for dist, idx in zip(distances, indices):
            config = configs[idx]
            score = 1.0 - dist
            results.append({
                "_id": config["record_id"],
                "primary_value": config["primary_value"],
                "metadata": config["metadata"],
                "score": round(float(score), 4)
            })
            
        return results
 
    async def validate_doc(self, db, doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        # 1. Extract values for each mapped field
        field_values: Dict[str, str] = {}
        for t, path in self.mappings.items():
            field_values[t] = str(get_nested_value(doc, path) or '').strip()
       
        # 2. Determine which fields should be validated based on conditional rules
        should_validate: Dict[str, bool] = {}
        for t in self.mappings:
            if t in CONDITIONAL_RULES:
                rule = CONDITIONAL_RULES[t]
                condition_val = str(get_nested_value(doc, rule["field"]) or '').strip().lower()
                if rule["condition"] == "equals":
                    should_validate[t] = (condition_val == rule["value"])
                elif rule["condition"] == "not_equals":
                    should_validate[t] = (condition_val != rule["value"])
                else:
                    should_validate[t] = True
            else:
                should_validate[t] = True
       
        # 3. Basic validation: valid or invalid only
        field_status: Dict[str, str] = {}
        param_flags: Dict[str, bool] = {}
       
        for t in self.mappings:
            if t not in self.primary_types:
                continue # Skip metadata fields from top-level validation
               
            val = field_values[t]
            is_empty = (val == "" or val == "nan")
           
            if not should_validate[t]:
                field_status[t] = "valid"
                param_flags[t] = False
                continue
           
            if is_empty or val not in self.valid_values.get(t, set()):
                field_status[t] = "invalid"
                param_flags[t] = True
            else:
                field_status[t] = "valid"
                param_flags[t] = False
       
        # 5. Construct invalid payload
        invalid_payload = []
        for t in self.mappings:
            if t not in self.primary_types:
                continue # Skip metadata fields from top-level loop
               
            val = field_values[t]
            is_empty = (val == "" or val == "nan")
           
            t_metadata = []
            has_metadata_mismatch = False
           
            overall_field_status = field_status.get(t, "valid")
            primary_validation_status = "valid" if not param_flags.get(t, False) else "invalid"
           
            if not param_flags.get(t, False) and should_validate.get(t, True) and not is_empty:
                possible_configs = self.val_metadata_reqs.get(t, {}).get(val, [])
               
                if possible_configs:
                    best_config_metadata = []
                    min_errors = 999
                    perfect_match_found = False
                   
                    for record_id, config in possible_configs:
                        current_config_metadata = []
                        current_errors = 0
                       
                        for m_name, m_info in config.items():
                            m_path = m_info.get("mapping", "")
                            m_required_val = m_info.get("required_val", "")
                           
                            m_val = ""
                            if m_path.startswith("processor_details."):
                                target_field = m_path.split(".", 1)[1]
                                cpu_model_val = field_values.get("CPUModel", "")
                                if cpu_model_val:
                                    proc_doc = await db["processor_details"].find_one({"cpuModelNo": cpu_model_val})
                                    if proc_doc:
                                        m_val = str(proc_doc.get(target_field, "")).strip()
                            else:
                                m_val = str(get_nested_value(doc, m_path) or '').strip() if m_path else ""
                           
                            m_is_empty = (m_val == "" or m_val == "nan")
                            if m_is_empty:
                                m_status = "invalid"
                                current_errors += 1
                            elif m_val != m_required_val:
                                m_status = "invalid"
                                current_errors += 1
                            else:
                                m_status = "valid"
                           
                            current_config_metadata.append({
                                "name": m_name,
                                "value": m_val,
                                "validation_status": m_status,
                                "mapping": m_path
                            })
                       
                        if current_errors == 0:
                            t_metadata = current_config_metadata
                            perfect_match_found = True
                            break
                       
                        if current_errors < min_errors:
                            min_errors = current_errors
                            best_config_metadata = current_config_metadata
                   
                    if not perfect_match_found:
                        t_metadata = best_config_metadata
                        has_metadata_mismatch = True
                        overall_field_status = "invalid"
           
            elif param_flags.get(t, False) and should_validate.get(t, True) and not is_empty:
                schema_reqs = self.type_metadata_paths.get(t, {})
                for m_name, m_path in schema_reqs.items():
                    m_val = ""
                    if m_path.startswith("processor_details."):
                        target_field = m_path.split(".", 1)[1]
                        cpu_model_val = field_values.get("CPUModel", "")
                       
                        if cpu_model_val:
                            proc_doc = await db["processor_details"].find_one({"cpuModelNo": cpu_model_val})
                            if proc_doc:
                                m_val = str(proc_doc.get(target_field, "")).strip()
                    else:
                        m_val = str(get_nested_value(doc, m_path) or '').strip() if m_path else ""
                       
                    m_is_empty = (m_val == "" or m_val == "nan")
                    m_status = "invalid"
                   
                    t_metadata.append({
                        "name": m_name,
                        "value": m_val,
                        "validation_status": m_status,
                        "mapping": m_path
                    })
           
            if overall_field_status == "invalid":
                invalid_payload.append({
                    "field": t,
                    "value": val,
                    "validation_status": primary_validation_status,
                    "mapping": self.mappings.get(t, ""),
                    "metadata": t_metadata
                })
       
        return invalid_payload, field_status
 
async def get_validator() -> Validator:
    db = get_db()
    mappings = await build_mappings()
    ml_records = await db[MASTERLIST_COL].find({"status": "Published"}).to_list(length=None)
    return Validator(ml_records, mappings)