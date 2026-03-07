# Todo

## Long Conversation Optimization

The current implementation is a simple design for benchmarking purposes. For long conversation optimization, the following approaches can be considered:

1. **Incremental Prefill** — Only prefill newly added tokens, preserving the KV Cache from previous turns instead of clearing it entirely.
2. **Prompt Cache** — Cache the KV states for the fixed system prompt, and only recompute the changed portions each turn.
3. **Expand Capacity** — Increase `KV_CACHE_CAPACITY` beyond 2048 (trade-off: larger state files and slower transfer).
4. **Sliding Window + Summarization** — Compress or summarize early context before it gets overwritten by the circular buffer.
