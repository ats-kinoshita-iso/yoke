# Yoke — eval runner targets
#
# Usage:
#   make eval-fast      Local models only, ~2 min (free, for dev iteration)
#   make eval-full      API models, ~4 min (costs ~$0.10, for pre-commit)
#   make eval-compare   Run both, show comparison table
#
# Prerequisites:
#   - Ollama running locally with gemma4:e4b loaded (for eval-fast)
#   - ANTHROPIC_API_KEY set (for eval-full)
#   - OPENAI_API_KEY set (for evals that use embeddings)

PYTEST = uv run python -m pytest
EVAL_DIR = evals
RESULTS_DIR = evals/results

# Which evals to run (all files that start with phase or model)
EVAL_FILES = \
	$(EVAL_DIR)/phase1_ingestion_eval.py \
	$(EVAL_DIR)/phase1_pipeline_eval.py \
	$(EVAL_DIR)/phase2_retrieval_eval.py \
	$(EVAL_DIR)/model_comparison.py

COMMON_FLAGS = -v -s --tb=short

.PHONY: eval-fast eval-full eval-compare eval-clean

# ---------------------------------------------------------------------------
# eval-fast: local models for generation + judging (~2 min, free)
# ---------------------------------------------------------------------------
eval-fast:
	@echo "=== eval-fast: using ollama/gemma4:e4b for generation + judging ==="
	$(PYTEST) $(EVAL_FILES) $(COMMON_FLAGS) --all-local \
		-p no:randomly 2>&1 | tee $(RESULTS_DIR)/eval-fast.log
	@echo ""
	@echo "=== eval-fast complete — results in $(RESULTS_DIR)/ ==="

# ---------------------------------------------------------------------------
# eval-full: API models for generation + judging (~4 min, ~$0.10)
# ---------------------------------------------------------------------------
eval-full:
	@echo "=== eval-full: using Claude API models ==="
	$(PYTEST) $(EVAL_FILES) $(COMMON_FLAGS) \
		-p no:randomly 2>&1 | tee $(RESULTS_DIR)/eval-full.log
	@echo ""
	@echo "=== eval-full complete — results in $(RESULTS_DIR)/ ==="

# ---------------------------------------------------------------------------
# eval-compare: run both, then show side-by-side summary
# ---------------------------------------------------------------------------
eval-compare: $(RESULTS_DIR)
	@echo "=== Running eval-fast (local) ==="
	-$(PYTEST) $(EVAL_FILES) $(COMMON_FLAGS) --all-local \
		-p no:randomly 2>&1 | tee $(RESULTS_DIR)/eval-fast.log
	@echo ""
	@echo "=== Running eval-full (API) ==="
	-$(PYTEST) $(EVAL_FILES) $(COMMON_FLAGS) \
		-p no:randomly 2>&1 | tee $(RESULTS_DIR)/eval-full.log
	@echo ""
	@echo "============================================================"
	@echo "  Comparison: eval-fast (local) vs eval-full (API)"
	@echo "============================================================"
	@echo ""
	@echo "--- eval-fast summary ---"
	@grep -E "(PASSED|FAILED|ERROR|average|Avg|Model Comparison)" \
		$(RESULTS_DIR)/eval-fast.log 2>/dev/null || echo "  (no results)"
	@echo ""
	@echo "--- eval-full summary ---"
	@grep -E "(PASSED|FAILED|ERROR|average|Avg|Model Comparison)" \
		$(RESULTS_DIR)/eval-full.log 2>/dev/null || echo "  (no results)"
	@echo ""
	@echo "Full logs: $(RESULTS_DIR)/eval-fast.log, $(RESULTS_DIR)/eval-full.log"
	@echo "JSON results: $(RESULTS_DIR)/*.json"

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

eval-clean:
	rm -f $(RESULTS_DIR)/*.log $(RESULTS_DIR)/*.json
