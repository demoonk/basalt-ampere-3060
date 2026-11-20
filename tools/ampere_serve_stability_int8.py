"""Serve stability, second sample: same 10 requests, larger expert cache and int8 KV."""
from __future__ import annotations
import json
import subprocess
import sys
import time
import urllib.request

PROMPT = ("Write a detailed paragraph about the engineering trade-offs of running a 34 GiB "
          "mixture-of-experts model on a 12 GB GPU. Vary the wording each time.")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PORT = 8110
EXE = r"C:\Neurolib\Tools\basalt-ampere-3060\bin\basalt.exe"
CFG = r"C:\Neurolib\Tools\basalt-ampere-3060\stab-cfg-v05-b.json"
LOG = r"C:\Neurolib\Tools\basalt-ampere-3060\stab-v05-b.log"

json.dump({"version": 2,
           "server": {"host": "127.0.0.1", "port": PORT},
           "engine": {"kv": "int8", "max-context": 16384, "spec": 8, "expert-cache": 3000},
           "models": [{"model": r"C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt",
                       "gpu": "0", "parallel": 1, "fit-max-tokens": True}]}, open(CFG, "w"))

subprocess.run(["cmd", "/c", "taskkill /f /im basalt.exe >nul 2>&1"], timeout=30)
time.sleep(5)
subprocess.Popen(["cmd", "/c", f"{EXE} serve --config {CFG} > {LOG} 2>&1"])

ready = False
t0 = time.time()
while time.time() - t0 < 900:
    try:
        if b'"loaded": true' in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10).read():
            ready = True
            break
    except Exception:
        pass
    time.sleep(5)
if not ready:
    print("server never ready")
    raise SystemExit(1)

models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=30).read())["data"]
ok, fails, decodes = 0, 0, []
for i in range(N):
    body = json.dumps({"model": models[0]["id"], "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 600, "temperature": 0.0}).encode()
    try:
        r = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"}), timeout=1200).read())
        tm = r.get("timings", {})
        ok += 1
        decodes.append(tm.get("predicted_per_second"))
        text = (r.get("content") or r.get("choices", [{}])[0].get("message", {}).get("content", ""))
        print(f"v05b #{i+1}: {tm.get('predicted_per_second')} tok/s drafts {tm.get('draft_n_accepted')}/{tm.get('draft_n')} :: {text[:220]}")
    except Exception as exc:
        fails += 1
        print(f"v05b #{i+1} FAILED: {exc}")
        time.sleep(8)
print(json.dumps({"ok": ok, "fail": fails,
                  "mean": round(sum(decodes) / len(decodes), 2) if decodes else None,
                  "min": min(decodes) if decodes else None, "max": max(decodes) if decodes else None}, indent=2))
subprocess.run(["cmd", "/c", "taskkill /f /im basalt.exe >nul 2>&1"], timeout=30)
