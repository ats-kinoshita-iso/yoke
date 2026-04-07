# Getting Started with Yoke

## Overview

This guide walks you through setting up a local Yoke development
environment from scratch. By the end, you will have all services
running and be able to submit your first query.

## Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.12+** — required for all Yoke services
- **PostgreSQL 16+** — with the pgvector extension enabled
- **Redis 7+** — used for async job processing

You will also need an Anthropic API key. Sign up at
https://console.anthropic.com to obtain one.

## Installation

Install all dependencies using uv:

```bash
uv sync
```

Then run the initialization command:

```bash
uv run yoke-init
```

This creates the default configuration files and verifies that all
dependencies are correctly installed.

## Environment Variables

### Required

These variables must be set before starting Yoke:

| Variable           | Description                              |
|--------------------|------------------------------------------|
| ANTHROPIC_API_KEY  | Your Anthropic API key                   |
| DATABASE_URL       | PostgreSQL connection string             |
| REDIS_URL          | Redis connection string                  |

Example values:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export DATABASE_URL="postgresql://yoke:yoke@localhost:5432/yoke_db"
export REDIS_URL="redis://localhost:6379/0"
```

### Optional

| Variable                    | Default | Description                    |
|-----------------------------|---------|--------------------------------|
| YOKE_LOG_LEVEL              | INFO    | Log verbosity (DEBUG, INFO, WARNING, ERROR) |
| YOKE_EMBEDDING_BATCH_SIZE   | 64      | Number of embeddings per batch |

## First-Time Setup

After setting your environment variables, run the database migrations:

```bash
yoke-migrate
```

This creates the required tables, indexes, and extensions in PostgreSQL
including the pgvector extension and HNSW indexes.

Next, ingest your initial documents:

```bash
yoke-ingest --dir ./docs
```

This recursively scans the `./docs` directory for supported file types
(Markdown, plain text, PDF) and queues them for embedding and indexing.

## Verifying the Installation

Once all services are running, check the health endpoint:

```
GET /health
```

A successful response looks like:

```json
{"status": "ok", "version": "0.1.0"}
```

If the response returns a non-200 status code, check that PostgreSQL
and Redis are running and reachable at the URLs specified in your
environment variables.

## Next Steps

- Read the [API Reference](api-reference.md) for endpoint documentation
- Read the [Architecture](architecture.md) doc for system design details
- See [Troubleshooting](troubleshooting.md) if you encounter issues
