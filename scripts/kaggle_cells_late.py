"""Late cells (4-6) of the Kaggle [L3] calibration notebook (v2).

Split from ``kaggle_cells.py`` to stay under the 400-line-per-file rule
(AGENTS.md section 51). See that module's docstring for the v2 fix list.
"""

from __future__ import annotations

CELL_4 = r"""# Cell 4 / 容量上限实测 —「同显存能多装几个 Agent」的真机判据
# v2: 真正释放 held 张量；用占位张量预留权重显存，回答「模型加载后还剩多少 KV」
import torch, json
CTX_TOKENS = (2048, 4096, 8192, 16384)
BPT = 57344                         # Cell 1 的 7B 真值
WEIGHT_GIB = (0.0, 1.0, 6.2, 14.2)  # 空卡 / 0.5B / 3B / 7B fp16 权重量级
GUARD_GIB = 1.0
cap = {
    'bytes_per_token': BPT, 'weight_reserve_gib': list(WEIGHT_GIB),
    'rows': [], 'error': None,
    'note': 'mock capacity is a KV budget, not total VRAM; 14.2 GiB reserve is 7B fp16',
}
try:
    free0, total0 = torch.cuda.mem_get_info(0)
    cap['free_gib_at_start'] = round(free0 / 1024**3, 2)
    cap['total_gib'] = round(total0 / 1024**3, 2)
    for wgib in WEIGHT_GIB:
        reserve = None
        try:
            if wgib:
                reserve = torch.empty(int(wgib * 1024**3), dtype=torch.uint8, device='cuda:0')
        except RuntimeError as exc:
            cap['rows'].append({'weight_gib': wgib, 'max_contexts': None,
                                'stopped_by': 'weight reserve OOM: ' + str(exc)[:90]})
            print(cap['rows'][-1])
            continue
        for ntok in CTX_TOKENS:
            per_ctx = ntok * BPT
            held, count = [], 0
            while count < 256:
                free_now, _ = torch.cuda.mem_get_info(0)
                if free_now < per_ctx + GUARD_GIB * 1024**3:
                    break
                try:
                    held.append(torch.empty(per_ctx, dtype=torch.uint8, device='cuda:0'))
                except RuntimeError:
                    break
                count += 1
            row = {
                'weight_gib': wgib, 'context_tokens': ntok,
                'bytes_per_context_mib': round(per_ctx / 1024**2, 1),
                'max_contexts': count,
            }
            cap['rows'].append(row)
            print(row)
            held.clear()          # v1 的 `for t in held: del t` 只解绑循环变量，张量从未释放
            torch.cuda.empty_cache()
        reserve = None
        torch.cuda.empty_cache()
except Exception as exc:
    cap['error'] = repr(exc)
    print('[WARN] Cell 4 失败：', cap['error'])
save('04_capacity.json', cap)
n8k = [r['max_contexts'] for r in cap['rows'] if r.get('context_tokens') == 8192]
check('容量上限实测', bool(n8k), '8k 上下文可装份数（按权重预留）= ' + str(n8k))
"""

CELL_5 = r"""# Cell 5 / 多并发解码衰减 alpha — batched forward 近似共享吞吐竞争
from transformers import AutoModelForCausalLM
import torch, gc, time, statistics
alpha = {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'rows': [], 'error': None,
         'method': 'batched decode; per-stream rate = base / (1 + a*(n-1))'}
past = None
try:
    model = AutoModelForCausalLM.from_pretrained(
        alpha['model'], torch_dtype=torch.float16, attn_implementation='sdpa'
    ).to('cuda:0').eval()
    with torch.inference_mode():
        for batch in (1, 2, 4, 8):
            ids = torch.randint(10, 1000, (batch, 1), device='cuda:0')
            past = None
            for _ in range(3):
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                ids = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(24):
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                ids = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            alpha['rows'].append({'batch': batch,
                                  'tok_per_s_total': round(batch * 24 / dt, 1),
                                  'tok_per_s_per_stream': round(24 / dt, 1)})
            print(alpha['rows'][-1])
    if len(alpha['rows']) > 1:
        base = alpha['rows'][0]['tok_per_s_per_stream']
        fits = [(base / r['tok_per_s_per_stream'] - 1) / (r['batch'] - 1)
                for r in alpha['rows'][1:]]
        alpha['contention_alpha_median'] = round(statistics.median(fits), 4)
        print('fitted alpha =', alpha['contention_alpha_median'], ' (mock default 0.15)')
    del model
    past = None
    gc.collect()
    torch.cuda.empty_cache()
except Exception as exc:
    alpha['error'] = repr(exc)
    print('[WARN] Cell 5 跳过：', alpha['error'])
save('05_alpha.json', alpha)
"""

CELL_6 = r"""# Cell 6 / 汇总 — 单独运行也能出结果；不产出无效的单一 prefill 常数
import json, statistics
from pathlib import Path

OUT = Path('/kaggle/working/phaseweaver_l3')
OUT.mkdir(parents=True, exist_ok=True)
def save(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    print('[saved]', p)
def check(name, ok, detail):
    print(('[PASS] ' if ok else '[FAIL] ') + name + ' :: ' + str(detail))
    return ok
def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

env, kv, rates, bw, cap, alpha = (
    load('00_env.json'), load('01_kv_bytes.json'), load('02_rates.json'),
    load('03_bandwidth.json'), load('04_capacity.json'), load('05_alpha.json'),
)
raw = rates.get('rows', [])
clean = [r for r in raw if r.get('prefill_tok_per_s', 0) >= 1000]  # 丢弃 v1 冷启动行
prefill_by_len = {r['prompt_tokens']: r['prefill_tok_per_s'] for r in clean}
decode = [r['decode_tok_per_s'] for r in clean if r.get('decode_tok_per_s')]
decode_med = round(statistics.median(decode), 1) if decode else None
seven_bpt = next((r['bytes_per_token'] for r in kv.get('rows', [])
                  if '7B' in r.get('model', '') and 'bytes_per_token' in r), 57344)
cal = {
    'evidence': ('[L3] micro-benchmark / parameter calibration, '
                 'NOT an end-to-end PhaseWeaver result'),
    'notebook_version': 2,
    'hardware': env.get('devices', []),
    'bytes_per_token': seven_bpt,
    'bytes_per_token_source': 'Qwen2.5-7B-Instruct config.json (exact)',
    'prefill_tok_per_s_measured': None,
    'prefill_rate_by_prompt_tokens': prefill_by_len,
    'prefill_scalar_note': ('no single scalar: prefill time is superlinear in prompt '
                            'length; mock assumes linear'),
    'decode_tok_per_s_measured': decode_med,
    'mem_bw_bps_measured': bw.get('mem_bw_bps'),
    'contention_alpha_measured': alpha.get('contention_alpha_median'),
    'free_vram_gib_measured': cap.get('free_gib_at_start'),
    'capacity_note': 'free VRAM is NOT the mock KV budget; see capacity_curve',
    'kv_bytes_models': kv.get('rows', []),
    'capacity_curve': cap.get('rows', []),
    'alpha_curve': alpha.get('rows', []),
    'dropped_cold_start_rows': len(raw) - len(clean),
    'rate_model': rates.get('model'),
    'model_mismatch': rates.get('model_mismatch',
        '0.5B rates + 7B bytes_per_token: not one coherent BackendCaps'),
    'mock_defaults': {
        'bytes_per_token': 57344, 'prefill_tok_per_s': 3200,
        'decode_tok_per_s': 45, 'mem_bw_bps': int(12*1024**3),
        'contention_alpha': 0.15,
    },
}
save('calibration.json', cal)
print(json.dumps(cal, indent=2))
if cal['dropped_cold_start_rows']:
    print('[WARN] 检测到 v1 冷启动行已丢弃；请重跑 Cell 2 获得带 warmup 的干净数据。')
print('[WARN] ' + str(cal['model_mismatch']))
lines = [
    '# [L3] 微基准标定 v2 — 真实 GPU，非端到端性能结论',
    '',
    '| 参数 | mock 默认 | 实测 |',
    '|---|---|---|',
    '| bytes_per_token | 57344 | ' + str(cal['bytes_per_token']) + ' |',
    '| prefill_tok_per_s | 3200 | 按长度: ' + str(prefill_by_len) + ' |',
    '| decode_tok_per_s | 45 | ' + str(decode_med) + ' |',
    '| mem_bw_bps | ' + str(int(12*1024**3)) + ' | ' + str(cal['mem_bw_bps_measured']) + ' |',
    '| contention_alpha | 0.15 | ' + str(cal['contention_alpha_measured']) + ' |',
    '',
    'prefill 无单一常数；不要回填进 `BackendCaps.prefill_tok_per_s`。',
]
(OUT / 'calibration.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
print()
print('下载：右侧 Output -> /kaggle/working/phaseweaver_l3/ 全部文件')
print('本地回灌：把 calibration.json 放到 runs/kaggle-l3/ 后跑')
print('  python scripts/probe_calibration.py')
check('标定完成', bool(decode_med or bw.get('mem_bw_bps')),
      '至少拿到一个真机参数即可回灌')
"""

LATE_CELLS = [CELL_4, CELL_5, CELL_6]
