"""Early cells (0-3) of the Kaggle [L3] calibration notebook (v2).

Cell sources are split across this module and ``kaggle_cells_late.py`` to stay
under the repository's 400-line-per-file rule (AGENTS.md section 51). The
notebook is a build artifact written to ``_local/`` (gitignored); the generator
is ``make_kaggle_notebook.py``.

v2 fixes three measurement defects found in the first real T4 run:
  1. Cell 2 timed the very first CUDA forward (kernel init) as a 256-token
     sample -> 164 tok/s; a mean over 5 lengths then produced a meaningless
     4152 tok/s scalar. v2 warms up, repeats 3x, takes the median, and reports
     a per-length curve instead of one number.
  2. Cell 4 never released the held tensors (``for t in held: del t`` only
     unbinds the loop variable), so the caching allocator kept the memory and
     produced the nonsense alternating curve 105 / 0 / 26 / 0 contexts.
  3. Cell 6 averaged incompatible quantities and labelled free VRAM as
     ``capacity_bytes`` (the mock's capacity is a KV budget, not total VRAM).

Evidence level stays [L3] micro-benchmark / parameter calibration. Nothing in
this notebook is an end-to-end PhaseWeaver result.
"""

from __future__ import annotations

INTRO_LINES = [
    "# PhaseWeaver [L3] 参数标定 v2 — Kaggle GPU T4 x2\n",
    "\n",
    "只标定仿真器参数，不跑 vLLM/SGLang，不声称端到端性能。\n",
    "\n",
    "Settings: Accelerator = `GPU T4 x2`、Internet = `On`、Persistence = `Files only`。\n",
    "**Cell 0 必须先跑**（定义 `OUT` / `save` / `check`）；其余 cell 各自 `try/except`，\n",
    "失败不阻塞后续，结果立即写 `/kaggle/working/phaseweaver_l3/`。\n",
    "Cell 6 可单独运行（自带同样的辅助函数），只跑过部分 cell 也能出 `calibration.json`。\n",
    "\n",
    "按顺序 Run All（或 Save & Run All 后台跑）。每个 cell 打印 `[PASS]`/`[FAIL]`。\n",
    "本版相对 v1：Cell 2 加 warmup + 3 次取中位 + 按长度曲线；Cell 4 修张量未释放、\n",
    "并预留权重显存；Cell 6 不再产出无效的单一 prefill 常数。\n",
    "\n",
    "**已知口径限制**：Cell 2/5 的速率来自 0.5B 模型，Cell 1 的 57344 B/token 来自 7B。\n",
    "两者不能合并成同一份 `BackendCaps`，`calibration.json` 会显式标注。",
]

CELL_0 = r"""# Cell 0 / 环境体检 — 必跑，约 30 秒
import json, subprocess, sys, time
from pathlib import Path
import torch

OUT = Path('/kaggle/working/phaseweaver_l3')
OUT.mkdir(parents=True, exist_ok=True)

def save(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    print('[saved]', p)

def check(name, ok, detail):
    print(('[PASS] ' if ok else '[FAIL] ') + name + ' :: ' + str(detail))
    return ok

env = {
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'devices': [],
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
for i in range(env['device_count']):
    p = torch.cuda.get_device_properties(i)
    env['devices'].append({
        'index': i, 'name': p.name,
        'compute_capability': str(p.major) + '.' + str(p.minor),
        'total_memory_gib': round(p.total_memory / 1024**3, 2),
        'sm_count': p.multi_processor_count,
    })
try:
    env['nvidia_smi'] = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
         '--format=csv,noheader'], capture_output=True, text=True, timeout=20
    ).stdout.strip()
except Exception as exc:
    env['nvidia_smi'] = 'unavailable: ' + str(exc)
save('00_env.json', env)
print(json.dumps(env, indent=2))
ok = bool(env['cuda_available'] and env['devices'] and env['devices'][0]['total_memory_gib'] >= 14)
check('GPU 体检', ok, 'T4 应为 14.56 GiB / sm_75；FAIL 时检查 Accelerator 后 Restart & Run All')
if not ok:
    print('STOP: 后续 cell 都需要 CUDA，先修好环境。')
"""

CELL_1 = r"""# Cell 1 / KV 字节标定 — 只下载 config.json，不下权重，秒级
from transformers import AutoConfig

MODELS = [
    'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
]
DTYPE_BYTES = 2  # T4 sm_75 无 bf16，用 fp16；每元素字节数与 bf16 相同
rows = []
for mid in MODELS:
    try:
        cfg = AutoConfig.from_pretrained(mid)
        n_layers = int(cfg.num_hidden_layers)
        n_kv = int(getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads))
        head_dim = int(getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads))
        bpt = 2 * n_layers * n_kv * head_dim * DTYPE_BYTES
        rows.append({
            'model': mid, 'layers': n_layers, 'kv_heads': n_kv,
            'head_dim': head_dim, 'bytes_per_token': bpt,
            'kib_per_token': round(bpt / 1024, 1),
            'kv_gib_at_8k': round(bpt * 8192 / 1024**3, 3),
        })
    except Exception as exc:
        rows.append({'model': mid, 'error': str(exc)})
kv = {'dtype': 'float16', 'formula': '2 * layers * kv_heads * head_dim * dtype_bytes', 'rows': rows}
save('01_kv_bytes.json', kv)
for r in rows:
    print(r)
seven = next((r for r in rows if '7B' in r['model'] and 'bytes_per_token' in r), None)
check('mock 的 57344 B/token 对齐真值',
      bool(seven and seven['bytes_per_token'] == 57344),
      '7B 实测 = ' + str(seven['bytes_per_token'] if seven else 'unknown'))
"""

CELL_2 = r"""# Cell 2 / prefill 与 decode 速率实测 — warmup + 3 次取中位，按长度出曲线
# 真机 prefill 随 prompt 长度非线性；mock 却用常数 rate。这里给曲线，不给假常数。
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, gc, statistics

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
PROMPT_LENS = (512, 1024, 2048, 4096)
DECODE_STEPS = 32
REPEATS = 3
WARMUP = 4
rates = {
    'model': MODEL_ID, 'dtype': 'float16', 'device': 'cuda:0',
    'repeats': REPEATS, 'warmup': WARMUP, 'rows': [], 'error': None,
    'model_mismatch': (
        'rates from 0.5B; bytes_per_token=57344 is 7B. '
        'Do NOT merge these into one BackendCaps without a same-model run.'
    ),
    'prefill_shape': 't(n) is superlinear; mock assumes t(n) = n / rate (linear)',
}
past = None
try:
    AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation='sdpa'
    ).to('cuda:0').eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        warm = torch.randint(10, 1000, (1, 256), device='cuda:0')
        for _ in range(WARMUP):
            model(input_ids=warm, use_cache=True)
        torch.cuda.synchronize()
        for plen in PROMPT_LENS:
            pf, dc = [], []
            for _ in range(REPEATS):
                ids = torch.randint(10, 1000, (1, plen), device='cuda:0')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(input_ids=ids, use_cache=True)
                torch.cuda.synchronize()
                pf.append(plen / (time.perf_counter() - t0))
                past = out.past_key_values
                last = ids[:, -1:]
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for _ in range(DECODE_STEPS):
                    step = model(input_ids=last, past_key_values=past, use_cache=True)
                    past = step.past_key_values
                    last = step.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                dc.append(DECODE_STEPS / (time.perf_counter() - t1))
            rates['rows'].append({
                'prompt_tokens': plen,
                'prefill_tok_per_s': round(statistics.median(pf), 1),
                'prefill_samples': [round(v, 1) for v in pf],
                'decode_tok_per_s': round(statistics.median(dc), 1),
                'decode_samples': [round(v, 1) for v in dc],
            })
            print(rates['rows'][-1])
    rates['peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    del model
    past = None
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    rates['error'] = repr(exc)
    print('[WARN] Cell 2 失败，其余 cell 仍可跑：', rates['error'])
save('02_rates.json', rates)
check('prefill/decode 实测', rates['error'] is None and bool(rates['rows']),
      str(len(rates['rows'])) + ' 组 / error=' + str(rates['error']))
"""

CELL_3 = r"""# Cell 3 / CPU<->GPU 带宽实测 — 只需 torch
import torch, statistics, time
SIZE_MIB = 256
REPEATS = 7
bw = {'size_mib': SIZE_MIB, 'repeats': REPEATS, 'samples_gb_s': [], 'error': None}
try:
    x = torch.empty(SIZE_MIB * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
    try:
        cpu = torch.empty(SIZE_MIB * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        pin = True
    except Exception:
        cpu = torch.empty(SIZE_MIB * 1024 * 1024, dtype=torch.uint8)
        pin = False
    bw['pinned_host_memory'] = pin
    for _ in range(2):
        cpu.copy_(x)
        x.copy_(cpu)
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu.copy_(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        x.copy_(cpu)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        bw['samples_gb_s'].append({
            'd2h': round(SIZE_MIB * 1024**2 / (t1 - t0) / 1e9, 2),
            'h2d': round(SIZE_MIB * 1024**2 / (t2 - t1) / 1e9, 2),
        })
    bw['d2h_gb_s'] = round(statistics.median(s['d2h'] for s in bw['samples_gb_s']), 2)
    bw['h2d_gb_s'] = round(statistics.median(s['h2d'] for s in bw['samples_gb_s']), 2)
    bw['mem_bw_bps'] = int(min(bw['d2h_gb_s'], bw['h2d_gb_s']) * 1e9)
    del x, cpu
    torch.cuda.empty_cache()
except Exception as exc:
    bw['error'] = repr(exc)
save('03_bandwidth.json', bw)
print(json.dumps({k: v for k, v in bw.items() if k != 'samples_gb_s'}, indent=2))
check('PCIe 带宽实测', bw['error'] is None,
      'mock 用 12.88 GB/s，实测 min(d2h,h2d) = ' + str(bw.get('mem_bw_bps')) + ' B/s')
"""

CELLS = [CELL_0, CELL_1, CELL_2, CELL_3]
