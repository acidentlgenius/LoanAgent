# Loan Agent Project

This project implements a Loan Application Agent using **LangGraph**. It features a deterministic state machine, asynchronous document processing, and a structured 14-step user journey.

## Architecture

The core of the application is a **LangGraph** workflow that manages the user's journey through the loan application process.

### 1. State Management (`graph/state.py`)
The `LoanState` dictates the entire lifecycle of the application. It is a strictly typed dictionary containing:
- **Identity**: `user_id`
- **Journey Tracking**: `current_step` (1-14), `max_steps_guard`, `finished` boolean.
- **Data**: `journey_data` (collected user inputs), `extracted_data` (from documents).
- **Document Status**: Tracks uploads, processing status, and verification queue.

### 2. Graph Construction (`graph/builder.py`)
The graph is built dynamically based on a deterministic map of steps.
- **Nodes**:
  - `universal_step_node`: Handles most steps (Name, Income, etc.) using LLM usage for prompting and extraction.
  - `document_upload_node` (Step 5): Initiates async document processing.
  - `review_node` (Step 13): Waits for all background tasks to complete and presents a summary.
  - `verification_node`: Handles validation of extracted document data.
  - `finish_node`: Marks the journey as complete.
- **Edges**:
  - All steps route through a central **Router**.

### 3. Routing Logic (`graph/router.py`)
The router is **deterministic** and purely rule-based (no LLM). It prioritizes traversal in this order:
1.  **Safety Guard**: Terminates if step count exceeds limit (Infinite loop protection).
2.  **Completion**: Checks if `finished` flag is set.
3.  **Verification**: If `verification_queue` is non-empty, diverts to `verification_node` immediately (interrupting the standard flow).
4.  **Sequential Step**: Moves to the next step defined in `STEP_NODE_MAP`.

## Document Handling

Document processing is designed to be **non-blocking** and **asynchronous**, allowing the user to continue answering questions while their documents are analyzed in the background.

### Flow
1.  **Upload (Step 5)**: The `document_upload_node` initiates processing for required documents (Bank Statement, Payslip, etc.) using `workers.document_processor`.
2.  **Async Processing**:
    - Each document is processed in a separate background thread (simulated by `threading` and `time.sleep`).
    - The main graph execution **does not block**. The user proceeds to Step 6 immediately.
3.  **Result Injection**:
    - At the end of *every* subsequent step (nodes 6-12), the `universal_step_node` checks the shared `processing_store`.
    - If a document is finished, its data is popped and injected into the graph state (`extracted_data`).
    - The document type is added to the `verification_queue`.
4.  **Verification Interruption**:
    - The `router` sees the updated `verification_queue`.
    - Before moving to the next logical step (e.g., Step 7), the graph automatically detours to `verification_node` to validate the newly extracted data.
    - Once verified, the router returns the user to their next logical step.
5.  **Synchronization (Step 13)**:
    - The `review_node` acts as a synchronization barrier.
    - It explicitly waits (loops) until *all* document processing threads are complete before generating the final review.

This architecture ensures a smooth user experience where long-running OCR/extraction tasks do not stall the conversation.

## Future Roadmap

While the current implementation is robust for a single-instance deployment, several architectural and performance enhancements are planned for scaling to production.

### Architectural Improvements
1.  **Distributed State & Queues**:
    -   *Current*: In-memory `processing_store` and Python `threading`.
    -   *Proposed*: Migrate to **Redis** or a database for sharing document status across multiple app instances (horizontal scaling). Replace `threading` with a durable task queue like **Celery**, **Arq**, or **Temporal** to handle retries and dead-letter queues reliably.
2.  **Structured State Objects**:
    -   *Current*: `TypedDict` for `LoanState`.
    -   *Proposed*: Migrate to **Pydantic v2 Models** for the graph state. This enables runtime validation, easier serialization, and automatic schema generation for checking/debugging state transitions.
3.  **Dynamic Routing Configuration**:
    -   *Current*: Hardcoded `router.py` logic.
    -   *Proposed*: Externalize the routing logic (step order, conditional branches) to a YAML/JSON configuration or a database table. This allows non-developers to modify the loan journey (e.g., A/B testing different questions) without code changes.
4.  **Dedicated Verification Subgraph**:
    -   *Current*: A single `verification_node`.
    -   *Proposed*: Break verification into a specialized **Subgraph** with its own state. This would allow complex, multi-turn conversations about specific document discrepancies (e.g., "The name on your PAN card doesn't match. Is this you?") to happen isolated from the main journey.

### Performance Optimizations
1.  **Optimized Checkpointing**:
    -   Use an async-optimized checkpointer (e.g., `AsyncPostgresSaver`) instead of the basic memory saver to reduce I/O blocking during high concurrency.
2.  **Streaming & Latency**:
    -   Ensure the `universal_step_node` yields token streams directly to the client (via `astream_events`) rather than buffering the full response. This minimizes "Time to First Token" (TTFT) for the user.

