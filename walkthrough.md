# Walkthrough: Asynchronous Approximate Nearest Neighbor Suggestions

I have implemented an Approximate Nearest Neighbor (ANN) suggestion system in the `Validator` class. This provides a high-performance alternative to the standard `rapidfuzz` matching, designed to handle large datasets without blocking the application's event loop.

## Changes Made

### 1. Dependencies [requirements.txt](file:///c:/Users/asbhavsa/Documents/GitHub/data_hygine/requirements.txt)
- Added `scikit-learn` for TF-IDF vectorization and Nearest Neighbor search.
- Note: `annoy` was omitted due to C++ compilation requirements on Windows, but `scikit-learn`'s `NearestNeighbors` provides excellent performance for this scale.

### 2. Validation Logic [validation.py](file:///c:/Users/asbhavsa/Documents/GitHub/data_hygine/validation.py)
- **Automatic Indexing**: The `Validator` now builds TF-IDF indexes for all field types and record signatures upon initialization.
- **New Methods**:
    - `get_suggestions_ann`: Asynchronous method for field-level suggestions.
    - `get_record_level_suggestions_ann`: Asynchronous method for concatenated "Mega-String" record suggestions.
- **Performance**: CPU-intensive vectorization and search tasks are wrapped in `asyncio.to_thread` to ensure the FastAPI server remains responsive.
- **Robustness**: Implemented a fallback mechanism (`HAS_SKLEARN`) to gracefully revert to standard `rapidfuzz` if dependencies are missing.

## Verification Results

I verified the implementation using a test script ([test_ann.py](file:///c:/Users/asbhavsa/Documents/GitHub/data_hygine/scratch/test_ann.py)).

### Output Summary:
- **Field Suggestions**: ANN successfully identified relevant matches for keywords like "Xeon".
- **Record Suggestions**: ANN successfully processed complex signatures and returned scored matches.
- **Async Execution**: Verified that the methods are awaitable and run correctly.

> [!TIP]
> To use the new functionality, you can await `get_suggestions_ann` or `get_record_level_suggestions_ann` instead of calling their synchronous `rapidfuzz` counterparts.
