# Yoke API Reference

## Overview

The Yoke API is served by the API Gateway on port 8090. All endpoints
accept and return JSON unless otherwise noted. This document covers
every public endpoint, its parameters, and expected responses.

## Authentication

All endpoints except GET /health require an `Authorization` header
with a Bearer token:

```
Authorization: Bearer <your-api-key>
```

Requests without a valid token receive a 401 Unauthorized response.

## Rate Limiting

All authenticated endpoints are rate-limited to 100 requests per minute
per API key. Exceeding this limit returns a 429 Too Many Requests
response with a `Retry-After` header.

## Upload Limits

The maximum upload size is 10MB per document. Requests exceeding this
limit receive a 413 Payload Too Large response.

## Endpoints

### POST /ask

Submit a question to the agent for retrieval-augmented answering.

**Request body:**

```json
{
  "question": "How do I configure hybrid retrieval?",
  "strategy": "hybrid"
}
```

The `strategy` field is optional and defaults to `"hybrid"`. Valid
values are `"dense"`, `"sparse"`, and `"hybrid"`.

**Response (200 OK):**

```json
{
  "answer": "To configure hybrid retrieval, set the strategy parameter...",
  "sources": [
    {"doc_id": "abc123", "title": "Architecture", "score": 0.92},
    {"doc_id": "def456", "title": "Configuration Guide", "score": 0.87}
  ]
}
```

### POST /ingest

Upload one or more documents for ingestion into the knowledge base.

**Request:** multipart/form-data with one or more file fields.

**Response (202 Accepted):**

```json
{
  "job_id": "job_7f3a1b2c",
  "status": "queued"
}
```

The ingestion runs asynchronously. Use GET /jobs/{job_id} to track
progress.

### GET /jobs/{job_id}

Check the status of an ingestion job.

**Response (200 OK):**

```json
{
  "job_id": "job_7f3a1b2c",
  "status": "completed",
  "documents_processed": 12,
  "errors": []
}
```

The `status` field is one of: `"pending"`, `"running"`, `"completed"`,
or `"failed"`. When status is `"failed"`, the `errors` array contains
descriptions of what went wrong.

### GET /health

Returns the health status of the platform. This endpoint does not
require authentication.

**Response (200 OK):**

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### DELETE /documents/{doc_id}

Remove a document and its embeddings from the index.

**Response (200 OK):**

```json
{
  "doc_id": "abc123",
  "deleted": true
}
```

Returns 404 if the document ID does not exist.

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "YK-003",
    "message": "Rate limit exceeded"
  }
}
```

See the [Troubleshooting](troubleshooting.md) guide for a full list
of error codes and resolutions.
