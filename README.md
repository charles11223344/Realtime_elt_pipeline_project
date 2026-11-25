# Realtime_elt_pipeline_project
Real-time clickstream feature engineering and ML inference pipeline on AWS (Kinesis, Lambda, DynamoDB, FastAPI, CDK). Features 1-min time-bucketed sliding windows with atomic counters, partial batch failure handling via SQS DLQ, and sub-50ms online inference.

```mermaid
sequenceDiagram
    autonumber
    actor User as Simulated User / Load Test
    participant Kin as Kinesis Data Streams
    participant Lam as Lambda (Feature Aggregation Worker)
    participant DDB as DynamoDB (5-Minute Time Buckets)
    participant SQS as SQS Dead-Letter Queue (DLQ)
    participant API as FastAPI Inference Service

    Note over User, DDB: Phase 1: Real-Time Event Ingestion and Sliding-Window Feature Updates
    User->>Kin: PutRecords (Batch clickstream events)
    Kin->>Lam: Trigger batch retrieval (BatchSize=100)
    
    alt Valid event format (Normal Path)
        Lam->>Lam: In-memory pre-aggregation by (user_id, window_min)
        Lam->>DDB: UpdateItem (Atomic ADD counter + set 1h TTL)
        DDB-->>Lam: Write successful
    else Malformed poison-pill data (Poison Pill)
        Lam->>SQS: SendMessage (Isolate bad data without blocking the main flow)
    else Transient DynamoDB throttling (Throttling)
        Lam-->>Kin: Return batchItemFailures (Retry only failed sequence numbers)
    end

    Note over User, API: Phase 2: Low-Latency Online Inference
    User->>API: GET /predict/{user_id}
    API->>DDB: Query latest 5 minute buckets (SK >= now - 4m)
    DDB-->>API: Return time bucket list
    API->>API: Combine 5-minute window features + calculate feature freshness + run model inference
    API-->>User: Return purchase probability, feature freshness (ms), and API latency (ms)
```