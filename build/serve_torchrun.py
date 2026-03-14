#!/usr/bin/env python3
"""
torchrun-based OpenAI-compatible vLLM server with continuous batching.

TP-agnostic: parallelism degree is set via torchrun parameters
(--nnodes, --nproc-per-node). All ranks run identical LLM engines
with external_launcher executor. Rank 0 hosts the HTTP API,
broadcasts requests via GLOO to all ranks.

Uses engine.add_request() + engine.step() for continuous batching
(new requests can join mid-generation, unlike static llm.chat()).

Protocol: Rank 0 broadcasts one message per engine step:
  {"new_requests": [...], "do_step": True/False}
All ranks add new requests, then call engine.step() if do_step=True.
This keeps engines perfectly synchronized (NCCL sync inside step()).

Requires: VLLM_ENABLE_V1_MULTIPROCESSING=0

Usage:
    torchrun --nnodes=2 --nproc-per-node=1 \\
        --node-rank=0 --master-addr=192.168.0.117 --master-port=29500 \\
        serve_torchrun.py --model /path/to/model --served-model-name my-model
"""

import argparse
import concurrent.futures
import os
import queue
import re
import sys
import threading
import time
import uuid

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


def parse_args():
    parser = argparse.ArgumentParser(
        description="torchrun OpenAI-compatible vLLM server"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--kv-cache-memory-bytes", type=str, default="10G")
    parser.add_argument("--speculative-model", type=str, default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs and torch.compile")
    parser.add_argument("--no-cuda-graphs", action="store_true",
                        help="Disable CUDA graphs but keep torch.compile")
    parser.add_argument("--load-format", type=str, default=None,
                        help="Model loading format (e.g. fastsafetensors)")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--reasoning-parser", type=str, default=None)
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser", type=str, default=None)
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument("--kv-cache-dtype", type=str, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    return parser.parse_args()


def parse_bytes(val: str) -> int:
    """Parse human-readable byte string (e.g. '10G', '512M')."""
    val = val.strip()
    if val.upper().endswith("G"):
        return int(val[:-1]) * 1024**3
    elif val.upper().endswith("M"):
        return int(val[:-1]) * 1024**2
    return int(val)


def create_llm(args) -> LLM:
    """Create LLM engine with external_launcher executor."""
    tp_size = int(os.environ.get("WORLD_SIZE", "1"))

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="external_launcher",
        seed=42,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    if args.kv_cache_memory_bytes:
        kwargs["kv_cache_memory_bytes"] = parse_bytes(args.kv_cache_memory_bytes)

    if args.quantization:
        kwargs["quantization"] = args.quantization

    if args.speculative_model:
        kwargs["speculative_config"] = {
            "model": args.speculative_model,
            "num_speculative_tokens": args.num_speculative_tokens or 1,
        }

    if args.enable_expert_parallel:
        kwargs["enable_expert_parallel"] = True

    if args.enforce_eager:
        kwargs["enforce_eager"] = True

    if args.no_cuda_graphs:
        from vllm.config import CompilationConfig
        kwargs["compilation_config"] = CompilationConfig(cudagraph_mode="none")

    if args.load_format:
        kwargs["load_format"] = args.load_format

    if args.enable_prefix_caching:
        kwargs["enable_prefix_caching"] = True

    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype

    if args.max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs

    return LLM(**kwargs)


# Communication between HTTP thread and main generate loop
_request_queue: queue.Queue = queue.Queue()
_response_map: dict[str, queue.Queue] = {}
_response_map_lock = threading.Lock()


def create_app(args) -> FastAPI:
    """Create FastAPI app (runs in Rank 0 only)."""
    app = FastAPI(title="vLLM torchrun server")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: dict):
        messages = request.get("messages", [])

        # Extract sampling parameters
        sampling_dict = {}
        for key in ("temperature", "max_tokens", "top_p", "top_k",
                     "seed", "repetition_penalty", "stop"):
            if key in request:
                sampling_dict[key] = request[key]

        # Default: temperature=0 for deterministic output
        if "temperature" not in sampling_dict:
            sampling_dict["temperature"] = 0.0

        # Enqueue request and wait for response
        req_id = uuid.uuid4().hex[:12]
        resp_q: queue.Queue = queue.Queue()
        with _response_map_lock:
            _response_map[req_id] = resp_q

        _request_queue.put((req_id, messages, sampling_dict))

        try:
            result = resp_q.get(timeout=300)
        finally:
            with _response_map_lock:
                _response_map.pop(req_id, None)

        if isinstance(result, Exception):
            return JSONResponse(
                status_code=500,
                content={"error": str(result)},
            )

        text, prompt_tokens, completion_tokens = result
        return {
            "id": f"chatcmpl-{req_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def _tokenize_chat(llm, messages):
    """Apply chat template and return token IDs."""
    # llm.llm_engine.tokenizer is a CachedTokenizer (HF PreTrainedTokenizer)
    # It has apply_chat_template directly — do NOT unwrap to ._tokenizer
    tokenizer = llm.llm_engine.tokenizer
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    # BatchEncoding (transformers 5.x): isinstance(x, dict) is False!
    if hasattr(result, 'input_ids'):
        return list(result.input_ids)
    if isinstance(result, list):
        return result
    return list(result)


def _add_to_engine(llm, req_id, token_ids, sampling_dict):
    """Add a pre-tokenized request to the engine."""
    params = SamplingParams(**sampling_dict)
    params.output_kind = RequestOutputKind.FINAL_ONLY
    llm.llm_engine.add_request(req_id, {"prompt_token_ids": token_ids}, params)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output."""
    stripped = _THINK_RE.sub("", text).strip()
    # If stripping removed everything, return original (safety net)
    return stripped if stripped else text


def run_rank0(llm: LLM, gloo_group, args):
    """Rank 0: HTTP server + continuous batching engine loop.

    Protocol: Every iteration does exactly one broadcast + one step.
    When idle (no inflight, no new requests), rank 0 blocks on the
    HTTP queue without broadcasting (workers block on broadcast_object_list).
    Once work exists, every loop iteration:
      1. Drain new HTTP requests, tokenize
      2. Broadcast {new_requests} to workers
      3. Both ranks: add_request() + step()  (NCCL-synced)
      4. Rank 0: route finished outputs to HTTP
    """
    app = create_app(args)

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=args.port, log_level="warning"
        ),
        daemon=True,
    )
    server_thread.start()
    print(f"[Rank 0] HTTP server listening on port {args.port} (continuous batching)")

    engine = llm.llm_engine
    inflight: set[str] = set()

    while True:
        # --- Phase 1: Collect new requests ---
        new_items = []

        if not engine.has_unfinished_requests():
            # Engine is idle — block until an HTTP request arrives.
            # Workers are also blocked on broadcast_object_list, so no desync.
            while True:
                try:
                    raw = _request_queue.get(timeout=60.0)
                    req_id, messages, sampling_dict = raw
                    token_ids = _tokenize_chat(llm, messages)
                    new_items.append({
                        "req_id": req_id,
                        "token_ids": token_ids,
                        "sampling": sampling_dict,
                    })
                    break
                except queue.Empty:
                    continue  # Keep waiting

        # Drain additional pending requests (non-blocking)
        while True:
            try:
                raw = _request_queue.get_nowait()
                req_id, messages, sampling_dict = raw
                token_ids = _tokenize_chat(llm, messages)
                new_items.append({
                    "req_id": req_id,
                    "token_ids": token_ids,
                    "sampling": sampling_dict,
                })
            except queue.Empty:
                break

        # --- Phase 2: Broadcast to workers ---
        # Workers receive this, add the same requests, then step.
        data = [{"new_requests": new_items}]
        dist.broadcast_object_list(data, src=0, group=gloo_group)

        # --- Phase 3: Add new requests to engine ---
        for item in new_items:
            try:
                _add_to_engine(llm, item["req_id"], item["token_ids"],
                               item["sampling"])
                inflight.add(item["req_id"])
            except Exception as e:
                print(f"[Rank 0] add_request error: {e}", file=sys.stderr)
                with _response_map_lock:
                    resp_q = _response_map.get(item["req_id"])
                if resp_q:
                    resp_q.put(e)

        if len(new_items) > 0:
            print(f"[Rank 0] +{len(new_items)} requests "
                  f"(inflight: {len(inflight)})", flush=True)

        # --- Phase 4: Step (both ranks step in lockstep via NCCL) ---
        # Timeout prevents system freeze if engine.step() hangs (e.g. NCCL deadlock)
        STEP_TIMEOUT = 120  # seconds
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(engine.step)
                try:
                    step_outputs = future.result(timeout=STEP_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    msg = (f"[Rank 0] engine.step() timed out after "
                           f"{STEP_TIMEOUT}s — possible NCCL deadlock. "
                           f"Failing {len(inflight)} inflight requests.")
                    print(msg, file=sys.stderr, flush=True)
                    err = TimeoutError(msg)
                    for req_id in list(inflight):
                        with _response_map_lock:
                            resp_q = _response_map.get(req_id)
                        if resp_q:
                            resp_q.put(err)
                    inflight.clear()
                    continue
        except Exception as e:
            print(f"[Rank 0] Engine step error: {e}", file=sys.stderr)
            for req_id in list(inflight):
                with _response_map_lock:
                    resp_q = _response_map.get(req_id)
                if resp_q:
                    resp_q.put(e)
            inflight.clear()
            continue

        # --- Phase 5: Route finished outputs to HTTP ---
        for output in step_outputs:
            if output.finished:
                req_id = output.request_id
                inflight.discard(req_id)
                try:
                    raw_text = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    # Fallback: if detokenizer produced empty text but tokens exist,
                    # decode manually (workaround for Step 3.5 tuple/list bug)
                    if not raw_text and token_ids:
                        tokenizer = llm.llm_engine.tokenizer
                        ids_list = list(token_ids)
                        # Debug: show first 20 token IDs and their individual decodes
                        print(f"[Rank 0] DEBUG token_ids[:{min(20,len(ids_list))}]: {ids_list[:20]}", flush=True)
                        for tid in ids_list[:10]:
                            print(f"  id={tid} -> {repr(tokenizer.decode([tid], skip_special_tokens=False))}", flush=True)
                        raw_text = tokenizer.decode(ids_list, skip_special_tokens=False)
                        print(f"[Rank 0] Detokenizer fallback (no skip): {len(ids_list)} tokens -> {repr(raw_text[:300])}", flush=True)
                        raw_text_clean = tokenizer.decode(ids_list, skip_special_tokens=True)
                        print(f"[Rank 0] Detokenizer fallback (skip):    {repr(raw_text_clean[:300])}", flush=True)
                        raw_text = raw_text_clean if raw_text_clean else raw_text
                    # Strip <think>...</think> blocks (reasoning models)
                    text = _strip_thinking(raw_text)
                    if raw_text != text or not text:
                        print(f"[Rank 0] raw={repr(raw_text[:200])} -> stripped={repr(text[:200])}", flush=True)
                    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
                    completion_tokens = len(token_ids)
                    result = (text, prompt_tokens, completion_tokens)
                except Exception as e:
                    result = e

                with _response_map_lock:
                    resp_q = _response_map.get(req_id)
                if resp_q:
                    resp_q.put(result)


def run_worker(llm: LLM, gloo_group, rank: int):
    """Rank 1+: receive broadcasts, add requests, step engine in lockstep.

    Mirrors rank 0's protocol exactly:
    - Block on broadcast_object_list (when rank 0 is idle, workers are too)
    - Receive new_requests, add them to engine
    - Call engine.step() (NCCL sync inside ensures lockstep with rank 0)
    """
    print(f"[Rank {rank}] Worker loop started (continuous batching)")

    engine = llm.llm_engine

    while True:
        # Block until rank 0 broadcasts
        data = [None]
        dist.broadcast_object_list(data, src=0, group=gloo_group)

        msg = data[0]
        if msg is None:
            continue
        if msg.get("shutdown"):
            print(f"[Rank {rank}] Shutdown received")
            break

        # Add new requests (identical to rank 0)
        for item in msg.get("new_requests", []):
            try:
                _add_to_engine(llm, item["req_id"], item["token_ids"],
                               item["sampling"])
            except Exception as e:
                print(f"[Rank {rank}] add_request error: {e}", file=sys.stderr)

        # Always step when rank 0 steps (NCCL sync inside)
        # Use has_unfinished_requests OR new_requests to decide
        if engine.has_unfinished_requests() or msg.get("new_requests"):
            STEP_TIMEOUT = 120
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(engine.step)
                    try:
                        future.result(timeout=STEP_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        print(f"[Rank {rank}] engine.step() timed out after "
                              f"{STEP_TIMEOUT}s", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Rank {rank}] Engine step error: {e}", file=sys.stderr)


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"[Rank {rank}/{world_size}] Initializing LLM engine...")
    llm = create_llm(args)
    print(f"[Rank {rank}/{world_size}] LLM engine ready.")

    # --- DIAGNOSTIC: Quick generation test ---
    print(f"[Rank {rank}] Running generation test...")
    try:
        from vllm import SamplingParams as _SP
        sp = _SP(max_tokens=30, temperature=0)
        outputs = llm.generate(["Hello! What is 2+2?"], sp)
        if rank == 0:
            for o in outputs:
                print(f"[DIAG] Output: {repr(o.outputs[0].text[:100])}")
                print(f"[DIAG] IDs: {list(o.outputs[0].token_ids[:10])}")
        print(f"[Rank {rank}] Generation test done.")
    except Exception as e:
        print(f"[Rank {rank}] DIAG Error: {e}")

    # Create GLOO group for CPU-based object broadcasts.
    # Use infinite timeout — workers idle-wait for rank 0 HTTP requests.
    from datetime import timedelta
    gloo_group = dist.new_group(backend="gloo", timeout=timedelta(days=365))

    if rank == 0:
        run_rank0(llm, gloo_group, args)
    else:
        run_worker(llm, gloo_group, rank)


if __name__ == "__main__":
    main()
