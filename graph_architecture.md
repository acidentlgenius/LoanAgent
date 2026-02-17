# Loan Agent Graph Architecture

A comprehensive guide to the LangGraph-based state machine that powers the loan application workflow.

## Overview

The Loan Agent is built on a **deterministic state machine** architecture using LangGraph. The system orchestrates a 14-step loan application journey while managing asynchronous document processing and intelligent routing logic.

## The Graph Structure

The core of the system is a directed graph where each node represents a specific stage in the loan application process. The Router (implemented as conditional edges) determines the flow between nodes based on the current state.

### Visual Architecture

```mermaid
graph TD
    %% Nodes
    START((START))
    END((END))
    
    %% Router Logic (Conceptual)
    ROUTER{Router<br/>(Conditional Edge)}

    %% Journey Mapping
    N1[step_1_name<br/>Name Collection]
    N2[step_2_dob<br/>Date of Birth]
    N3[step_3_contact<br/>Contact Information]
    N4[step_4_income<br/>Income Details]
    N5[step_5_document_upload<br/>Document Upload]
    N6[step_6_employment<br/>Employment Details]
    N7[step_7_address<br/>Address Information]
    N8[step_8_loan_amount<br/>Loan Amount]
    N9[step_9_loan_tenure<br/>Loan Tenure]
    N10[step_10_purpose<br/>Loan Purpose]
    N11[step_11_references<br/>References]
    N12[step_12_consent<br/>Consent]
    N13[step_13_review<br/>Application Review]
    N14[step_14_summary<br/>Final Summary]
    
    VERIFY[verification_node<br/>Document Verification]
    FINISH[finish_node<br/>Completion]

    %% Edges
    START --> N1
    
    %% All nodes route through the Router
    N1 --> ROUTER
    N2 --> ROUTER
    N3 --> ROUTER
    N4 --> ROUTER
    N5 --> ROUTER
    N6 --> ROUTER
    N7 --> ROUTER
    N8 --> ROUTER
    N9 --> ROUTER
    N10 --> ROUTER
    N11 --> ROUTER
    N12 --> ROUTER
    N13 --> ROUTER
    N14 --> ROUTER
    VERIFY --> ROUTER

    %% Router Decision Logic
    ROUTER --> |"Termination Condition"| FINISH
    ROUTER --> |"Verification Pending"| VERIFY
    ROUTER --> |"Next Step"| N1
    ROUTER --> |"Next Step"| N2
    ROUTER --> |"Next Step"| N3
    ROUTER --> |"Next Step"| N4
    ROUTER --> |"Next Step"| N5
    ROUTER --> |"Next Step"| N6
    ROUTER --> |"Next Step"| N7
    ROUTER --> |"Next Step"| N8
    ROUTER --> |"Next Step"| N9
    ROUTER --> |"Next Step"| N10
    ROUTER --> |"Next Step"| N11
    ROUTER --> |"Next Step"| N12
    ROUTER --> |"Next Step"| N13
    ROUTER --> |"Next Step"| N14
    
    FINISH --> END

    classDef custom fill:#f96,stroke:#333,stroke-width:2px;
    classDef universal fill:#9cf,stroke:#333,stroke-width:2px;
    classDef logic fill:#ff9,stroke:#333,stroke-width:2px;
    classDef term fill:#f66,stroke:#333,stroke-width:2px;

    class N5,N14 custom;
    class N1,N2,N3,N4,N6,N7,N8,N9,N10,N11,N12,N13 universal;
    class ROUTER logic;
    class FINISH,END term;
    class VERIFY custom;
```

## Asynchronous Document Processing

One of the key architectural features is the **non-blocking document processing** system that allows the application flow to continue while documents are being analyzed in the background.

### Processing Flow

1. **Document Upload (Step 5)**: When users upload documents (bank statements, payslips, etc.), the system initiates background processing threads without blocking the main conversation flow.

2. **Concurrent Execution**: Background workers process documents using simulated OCR/extraction while the user continues through subsequent steps (employment, address, etc.).

3. **State Injection**: After each step, the system checks for completed document processing. When a document finishes processing:
   - Extracted data is injected into the graph state
   - The document type is added to the `verification_queue`

4. **Priority Routing**: The Router detects pending verifications and temporarily diverts the flow to the `verification_node` before resuming the standard journey.

5. **Synchronization Point (Step 14)**: The review node acts as a barrier, ensuring all background processing is complete before generating the final summary.

## Router Logic

The Router (`graph/router.py`) implements deterministic, rule-based routing with the following priority hierarchy:

### Decision Priority

1. **Safety Guard**: If `max_steps_guard` exceeds the configured limit (25), terminate the graph to prevent infinite loops.

2. **Completion Check**: If the `finished` flag is set to `True`, route to the finish node.

3. **Verification Priority**: If `verification_queue` contains pending items, route to `verification_node` immediately (interrupting the sequential flow).

4. **Sequential Progression**: If `current_step` maps to a valid node in `STEP_NODE_MAP`, route to that node.

5. **Default Termination**: If no conditions are met, route to the finish node.

This deterministic approach ensures predictable behavior and prevents the routing logic from introducing non-deterministic LLM-based decisions into the control flow.

## Key Design Principles

- **Deterministic Routing**: All routing decisions are rule-based, ensuring consistent and predictable behavior.
- **Non-Blocking I/O**: Document processing happens asynchronously to maintain conversation flow.
- **Event-Driven Verification**: The system responds to document processing completion events by dynamically adjusting the route.
- **State Isolation**: All state is managed through the `LoanState` TypedDict, providing a single source of truth.
