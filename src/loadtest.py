"""End-to-end latency test against a running service.

    python -m src.loadtest --url http://localhost:8000 --n 2000 --concurrency 1 8 [--explain]

Sends real held-out records (artifacts/sample_requests.csv) and reports p50/p95/p99 and throughput.
Run it against the container to get numbers that mean something for your hardware.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run(url, payloads, n, conc, explain):
    lat = []

    def worker(ids):
        with httpx.Client(base_url=url, timeout=10) as c:
            out = []
            for i in ids:
                t = time.perf_counter()
                r = c.post("/predict", params={"explain": str(explain).lower()}, json=payloads[i % len(payloads)])
                out.append((time.perf_counter() - t) * 1000)
                assert r.status_code == 200, r.text
            return out

    with httpx.Client(base_url=url) as c:  # warm-up
        for p in payloads[:50]:
            c.post("/predict", json=p)
    chunks = [range(k, n, conc) for k in range(conc)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        for part in ex.map(worker, chunks):
            lat += part
    wall = time.perf_counter() - t0
    return {"concurrency": conc, "explain": explain, "requests": len(lat),
            "p50_ms": round(pct(lat, .5), 2), "p95_ms": round(pct(lat, .95), 2),
            "p99_ms": round(pct(lat, .99), 2), "mean_ms": round(statistics.mean(lat), 2),
            "throughput_rps": round(len(lat) / wall, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(ART / "sample_requests.csv").drop(columns=["offline_probability"])
    payloads = [{k: (int(v) if k == "EDUCATION" or (k.startswith("PAY_") and not k.startswith("PAY_AMT")) else float(v))
                 for k, v in r.items()} for r in df.to_dict("records")]
    res = [run(a.url, payloads, a.n, c, e) for c in a.concurrency for e in (False, True)]
    for r in res:
        print(r)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
