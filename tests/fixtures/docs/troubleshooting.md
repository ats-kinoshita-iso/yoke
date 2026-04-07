# Troubleshooting Yoke

## Overview

This document lists common errors, their causes, and how to resolve
them. Each error is identified by a unique code prefixed with "YK-".

## Error Codes

### YK-001: Connection refused on port 5432

**Message:** `Connection refused on port 5432`

**Cause:** PostgreSQL is not running or is not listening on port 5432.

**Resolution:**
1. Verify PostgreSQL is running: `systemctl status postgresql`
2. Check that it is listening on port 5432: `ss -tlnp | grep 5432`
3. Confirm your DATABASE_URL environment variable points to the correct
   host and port
4. Restart PostgreSQL if needed: `systemctl restart postgresql`

### YK-002: Embedding dimension mismatch

**Message:** `Embedding dimension mismatch: expected 1024, got <N>`

**Cause:** The configured embedding model does not produce 1024-dimension
vectors. Yoke requires the voyage-3 model which outputs 1024 dimensions.

**Resolution:**
1. Verify your embedding model is set to voyage-3
2. If you recently changed models, you must re-index all documents by
   running `yoke-ingest --reindex`
3. Check that no stale embeddings remain from a previous model by
   running `yoke-db check-dimensions`

### YK-003: Rate limit exceeded

**Message:** `Rate limit exceeded`

**Cause:** Your API key has sent more than 100 requests in the current
one-minute window.

**Resolution:**
1. Wait 60 seconds for the rate limit window to reset
2. Implement exponential backoff in your client
3. Upgrade your plan for higher rate limits
4. Check for runaway loops in your application that may be sending
   duplicate requests

### YK-004: Context window exceeded

**Message:** `Context window exceeded: input is <N> tokens, maximum is 180000`

**Cause:** The combined input (question + retrieved documents) exceeds
the 180,000-token context window.

**Resolution:**
1. Split large documents into chunks under 50,000 tokens each before
   ingestion
2. Reduce the number of retrieved documents by lowering the `top_k`
   parameter
3. Use the `max_tokens` parameter on the /ask endpoint to constrain
   input size
4. Re-ingest oversized documents with `yoke-ingest --chunk-size 50000`

### YK-005: Redis connection timeout

**Message:** `Redis connection timeout after 5000ms`

**Cause:** The Redis server is unreachable or not running.

**Resolution:**
1. Check that Redis is running: `systemctl status redis`
2. Verify your REDIS_URL environment variable is correct
3. Test connectivity: `redis-cli -u $REDIS_URL ping`
4. Check firewall rules if Redis is on a remote host
5. Ensure Redis 7+ is installed (older versions may have compatibility
   issues)

## Performance Issues

### Slow Queries

If queries are taking longer than expected:

1. Enable DEBUG logging by setting `YOKE_LOG_LEVEL=DEBUG` in your
   environment
2. Check the retrieval logs to see if hybrid retrieval is falling back
   to dense-only mode (this happens when the BM25 index is unavailable)
3. Verify the HNSW index is built by running `yoke-db check-indexes`
4. Monitor PostgreSQL query performance with `pg_stat_statements`

### Memory Issues

If the Retrieval Service is consuming excessive memory:

1. Reduce YOKE_EMBEDDING_BATCH_SIZE from the default of 64 to 16
2. Restart the Retrieval Service after changing the batch size
3. Monitor memory usage: the Retrieval Service should use under 2GB
   with a batch size of 16
4. For very large corpora (over 1 million documents), consider enabling
   disk-based indexing with `YOKE_INDEX_MODE=disk`

## Getting Help

If your issue is not listed here:

1. Check the logs at DEBUG level for detailed error context
2. Search existing GitHub issues for similar problems
3. Open a new issue with the error code, full stack trace, and your
   Yoke version (`yoke --version`)
