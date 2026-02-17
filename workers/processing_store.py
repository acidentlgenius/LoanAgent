"""Thread-safe store for async document processing results.

Background threads write completed extractions here.
Journey nodes read/pop completed docs to inject into verification_queue.
"""

import threading
from typing import Any


_lock = threading.Lock()
_results: dict[str, dict[str, Any]] = {}   # doc_type → extracted_data
_processing: set[str] = set()               # doc_types currently being processed


def mark_processing(doc_type: str) -> None:
    """Mark a document as being processed."""
    with _lock:
        _processing.add(doc_type)


def store_result(doc_type: str, data: dict[str, Any]) -> None:
    """Store completed extraction and remove from processing set."""
    with _lock:
        _results[doc_type] = data
        _processing.discard(doc_type)


def pop_completed() -> list[tuple[str, dict[str, Any]]]:
    """Pop all completed documents. Returns [(doc_type, extracted_data), ...]"""
    with _lock:
        items = list(_results.items())
        _results.clear()
    return items


def is_any_processing() -> bool:
    """True if any documents are still being processed."""
    with _lock:
        return bool(_processing)


def get_processing_docs() -> list[str]:
    """Get list of doc_types currently being processed."""
    with _lock:
        return list(_processing)


def reset() -> None:
    """Clear all state. Used for testing and session resets."""
    with _lock:
        _results.clear()
        _processing.clear()
