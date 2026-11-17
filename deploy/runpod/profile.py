"""Measures the vLLM servers so the simulated GPUs can match them.

Run on the pod with the three servers up and nothing else using the GPU:

    source deploy/runpod/env.sh && python deploy/runpod/profile.py

It fits the same formulas the simulated GPUs use (internal/gpu) and prints
the values to put in system.DefaultConfig:

    embed / rerank batch:  Base + PerToken x tokens in flight
    LLM decode step:       StepBase + DecodePerSeq x running sequences
    LLM prefill:           PrefillPerToken x prompt tokens
"""

import asyncio
import os
import random
import statistics
import time

import aiohttp
import numpy as np

LLM = (f"http://127.0.0.1:{os.environ.get('LLM_PORT', '8000')}", os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
EMBED = (f"http://127.0.0.1:{os.environ.get('EMBED_PORT', '8002')}", os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5"))
RERANK = (f"http://127.0.0.1:{os.environ.get('RERANK_PORT', '8003')}", os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))

WORDS = ("time year people way day man thing woman life child world school state family student group "
         "country problem hand part place case week company system program question work government number "
         "night point home water room mother area money story fact month lot right study book eye job word "
         "business issue side kind head house service friend father power hour game line end member law car "
         "city community name president team minute idea kid body information back parent face others level "
         "office door health person art war history party result change morning reason research girl guy "
         "moment air teacher force education").split()

REPEATS = 5


def text(rng, n):
    """About n tokens of random words (random so vLLM's prefix cache can't help)."""
    return " ".join(rng.choice(WORDS) for _ in range(n))


async def tokens(session, server, s):
    url, model = server
    async with session.post(f"{url}/tokenize", json={"model": model, "prompt": s}) as r:
        return (await r.json())["count"]


def fit(xs, ys):
    """Least-squares y = a + b x; returns (a, b)."""
    b, a = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    return a, b


async def wall(coros):
    t = time.perf_counter()
    await asyncio.gather(*coros)
    return time.perf_counter() - t


async def embed_one(session, s):
    url, model = EMBED
    async with session.post(f"{url}/v1/embeddings", json={"model": model, "input": s}) as r:
        r.raise_for_status()
        await r.read()


async def rerank_one(session, q, docs):
    url, model = RERANK
    async with session.post(f"{url}/rerank", json={"model": model, "query": q, "documents": docs}) as r:
        r.raise_for_status()
        await r.read()


async def profile_embed(session, rng):
    # 256-token texts (bge-base takes at most 512), more of them at once.
    per = await tokens(session, EMBED, text(rng, 250))
    xs, ys = [], []
    for n in (1, 4, 16, 32, 64):
        ts = [await wall([embed_one(session, text(rng, 250)) for _ in range(n)]) for _ in range(REPEATS)]
        xs.append(n * per)
        ys.append(statistics.median(ts))
        print(f"  embed   {n:3d} x {per} tokens = {n * per:6d}: {ys[-1] * 1000:7.1f} ms")
    return fit(xs, ys)


async def profile_rerank(session, rng):
    # Workload-shaped requests: a 20-token query and 8 docs of ~150 tokens.
    q = text(rng, 20)
    docs = [text(rng, 150) for _ in range(8)]
    per = sum([await tokens(session, RERANK, q + " " + d) for d in docs])
    xs, ys = [], []
    for n in (1, 2, 4, 8, 12):
        ts = []
        for _ in range(REPEATS):
            reqs = [(text(rng, 20), [text(rng, 150) for _ in range(8)]) for _ in range(n)]
            ts.append(await wall([rerank_one(session, q, d) for q, d in reqs]))
        xs.append(n * per)
        ys.append(statistics.median(ts))
        print(f"  rerank  {n:3d} x {per} tokens = {n * per:6d}: {ys[-1] * 1000:7.1f} ms")
    return fit(xs, ys)


async def stream(session, prompt, out, stamps):
    url, model = LLM
    body = {"model": model, "prompt": prompt, "max_tokens": out, "min_tokens": out,
            "ignore_eos": True, "stream": True}
    async with session.post(f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.content:
            if line.startswith(b"data:") and b"[DONE]" not in line and b'"text"' in line:
                stamps.append(time.perf_counter())


async def profile_decode(session, rng):
    # B streams of 128 tokens each; the step time is the gap between tokens in
    # the middle of the run, when all B are decoding together.
    xs, ys = [], []
    for b in (1, 16, 64, 128, 256):
        all_stamps = [[] for _ in range(b)]
        await asyncio.gather(*[stream(session, text(rng, 300), 128, s) for s in all_stamps])
        gaps = []
        for s in all_stamps:
            mid = s[len(s) // 4: 3 * len(s) // 4]
            gaps += [y - x for x, y in zip(mid, mid[1:])]
        xs.append(b)
        ys.append(statistics.median(gaps))
        print(f"  decode  {b:3d} sequences: {ys[-1] * 1000:6.2f} ms per step")
    return fit(xs, ys)


async def profile_prefill(session, rng):
    # Time to first token for one request alone, at growing prompt sizes.
    xs, ys = [], []
    for n in (256, 1024, 2048, 4096):
        ts = []
        for _ in range(REPEATS):
            p = text(rng, n)
            stamps = []
            t = time.perf_counter()
            await stream(session, p, 1, stamps)
            ts.append(stamps[0] - t)
        xs.append(await tokens(session, LLM, p))
        ys.append(statistics.median(ts))
        print(f"  prefill {xs[-1]:5d} tokens: {ys[-1] * 1000:7.1f} ms to first token")
    return fit(xs, ys)


def us(seconds):
    return f"{seconds * 1e6:.0f}µs"


def ms(seconds):
    return f"{seconds * 1e3:.1f}ms"


async def main():
    rng = random.Random(1)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        # Warm up every server once.
        await embed_one(session, text(rng, 50))
        await rerank_one(session, text(rng, 10), [text(rng, 50)])
        await stream(session, text(rng, 50), 4, [])

        print("Embed")
        eb, ep = await profile_embed(session, rng)
        print("Rerank")
        rb, rp = await profile_rerank(session, rng)
        print("LLM decode")
        sb, sd = await profile_decode(session, rng)
        print("LLM prefill")
        _, pp = await profile_prefill(session, rng)

    print()
    print("Fitted simulator settings (system.DefaultConfig):")
    print(f"  Embed:  Base {ms(eb)}  PerToken {us(ep)}")
    print(f"  Rerank: Base {ms(rb)}  PerToken {us(rp)}")
    print(f"  LLM:    StepBase {ms(sb)}  DecodePerSeq {us(sd)}  PrefillPerToken {us(pp)}")


asyncio.run(main())
