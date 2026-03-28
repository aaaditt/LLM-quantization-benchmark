"""
Project 3: LLM Interface Optimization — Capstone
Benchmarks Phi-3 Mini 3.8B at different quantization levels:
  - FP16 on CPU  (baseline — no GPU)
  - INT8 on GPU  (moderate compression)
  - INT4 on GPU  (maximum compression, fits in 4GB VRAM)

Measures: tokens/sec, VRAM usage, response quality, latency
"""

import torch
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MODEL_ID    = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 200
NUM_RUNS       = 3   # average over multiple prompts

BENCHMARK_PROMPTS = [
    "Explain what a neural network is in simple terms.",
    "What is the difference between machine learning and deep learning?",
    "How does GPU parallelism speed up matrix multiplication?",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_vram_used():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def get_vram_total():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def measure_tokens_per_second(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS):
    """Run inference on one prompt, return (tokens_per_sec, output_text, latency_ms)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Timed run
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    new_tokens   = output.shape[1] - input_len
    tokens_per_s = new_tokens / elapsed
    latency_ms   = elapsed * 1000
    output_text  = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    return tokens_per_s, output_text, latency_ms, new_tokens


# ─────────────────────────────────────────────
# BENCHMARK A SINGLE CONFIGURATION
# ─────────────────────────────────────────────
def run_benchmark(config_name, load_fn, device_str):
    from transformers import AutoTokenizer
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {config_name}")
    print(f"{'='*60}")

    clear_gpu()
    vram_before = get_vram_used()

    print(f"  Loading model... (this may take a minute)")
    t_load = time.time()
    model, tokenizer = load_fn()
    load_time = time.time() - t_load

    vram_after  = get_vram_used()
    vram_used   = vram_after - vram_before

    print(f"  Load time     : {load_time:.1f}s")
    print(f"  VRAM used     : {vram_used:.2f} GB")

    all_tps     = []
    all_latency = []
    outputs     = []

    for i, prompt in enumerate(BENCHMARK_PROMPTS):
        print(f"\n  Prompt {i+1}: \"{prompt[:55]}...\"")
        tps, text, lat, ntok = measure_tokens_per_second(model, tokenizer, prompt, device_str)
        all_tps.append(tps)
        all_latency.append(lat)
        outputs.append(text)
        print(f"    → {ntok} tokens | {tps:.1f} tok/s | {lat:.0f}ms")
        print(f"    → \"{text[:120].strip()}...\"")

    avg_tps     = np.mean(all_tps)
    avg_latency = np.mean(all_latency)

    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Avg tokens/sec : {avg_tps:.1f}")
    print(f"  Avg latency    : {avg_latency:.0f}ms")
    print(f"  VRAM footprint : {vram_used:.2f} GB")

    # Cleanup
    del model
    clear_gpu()

    return {
        "name"        : config_name,
        "avg_tps"     : avg_tps,
        "avg_latency" : avg_latency,
        "vram_gb"     : vram_used,
        "load_time"   : load_time,
        "all_tps"     : all_tps,
        "outputs"     : outputs,
    }


# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
def load_fp16_cpu():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer

def load_int8_gpu():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer

def load_int4_gpu():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────
def plot_results(results):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "LLM Optimization Benchmark — Phi-3 Mini 3.8B\nFP16 (CPU) vs INT8 (GPU) vs INT4 (GPU)",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    names  = [r["name"] for r in results]
    tps    = [r["avg_tps"] for r in results]
    lats   = [r["avg_latency"] for r in results]
    vrams  = [r["vram_gb"] for r in results]
    colors = ["#4C72B0", "#55A868", "#DD4444"][:len(results)]

    # 1. Tokens per second
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(names, tps, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax1.set_title("Tokens / Second", fontweight="bold")
    ax1.set_ylabel("Tokens/sec")
    for bar, v in zip(bars, tps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    if len(results) >= 3:
        speedup = tps[-1] / tps[0]
        ax1.text(len(results)-1, tps[-1]*0.5, f"{speedup:.1f}x\nfaster",
                 ha="center", color="white", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # 2. Latency
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(names, lats, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax2.set_title("Avg Response Latency", fontweight="bold")
    ax2.set_ylabel("Latency (ms)")
    for bar, v in zip(bars2, lats):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    # 3. VRAM usage
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(names, vrams, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax3.axhline(y=get_vram_total(), color="red", linestyle="--", linewidth=1.5,
                label=f"GTX 1650 VRAM limit ({get_vram_total():.1f}GB)")
    ax3.set_title("VRAM Footprint", fontweight="bold")
    ax3.set_ylabel("VRAM (GB)")
    ax3.legend(fontsize=8)
    for bar, v in zip(bars3, vrams):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.2f}GB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)

    # 4. Per-prompt tokens/sec breakdown
    ax4 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(BENCHMARK_PROMPTS))
    width = 0.25
    short_prompts = [p[:40] + "..." for p in BENCHMARK_PROMPTS]
    for i, (r, c) in enumerate(zip(results, colors)):
        ax4.bar(x + i*width, r["all_tps"], width, label=r["name"], color=c,
                edgecolor="black", linewidth=0.6)
    ax4.set_title("Tokens/sec per Prompt", fontweight="bold")
    ax4.set_ylabel("Tokens/sec")
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(short_prompts, fontsize=8)
    ax4.legend()
    ax4.grid(True, axis="y", alpha=0.3)

    # 5. Summary stats table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    table_data = [["Metric"] + [r["name"] for r in results]]
    table_data.append(["Avg tok/s"] + [f"{r['avg_tps']:.1f}" for r in results])
    table_data.append(["Latency"] + [f"{r['avg_latency']:.0f}ms" for r in results])
    table_data.append(["VRAM"] + [f"{r['vram_gb']:.2f}GB" for r in results])
    table_data.append(["Load time"] + [f"{r['load_time']:.0f}s" for r in results])
    if len(results) >= 2:
        table_data.append(["Speedup vs FP16"] + ["1x"] + [f"{r['avg_tps']/results[0]['avg_tps']:.1f}x" for r in results[1:]])

    tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F9FA")
    ax5.set_title("Summary Table", fontweight="bold", pad=10)

    plt.savefig("llm_benchmark.png", dpi=150, bbox_inches="tight")
    print("\n  Chart saved → llm_benchmark.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Project 3: LLM Quantization Benchmark")
    print(f"  Model : {MODEL_ID}")
    print("=" * 60)
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {get_vram_total():.1f} GB")

    results = []

    # ── FP16 on CPU ──────────────────────────
    print("\n[1/3] FP16 on CPU (baseline)...")
    r1 = run_benchmark("FP16\n(CPU)", load_fp16_cpu, "cpu")
    results.append(r1)

    # ── INT8 on GPU ──────────────────────────
    print("\n[2/3] INT8 on GPU...")
    try:
        r2 = run_benchmark("INT8\n(GPU)", load_int8_gpu, "cuda")
        results.append(r2)
    except Exception as e:
        print(f"  INT8 failed: {e}")

    # ── INT4 on GPU ──────────────────────────
    print("\n[3/3] INT4 on GPU (NF4 double quant)...")
    try:
        r3 = run_benchmark("INT4\n(GPU)", load_int4_gpu, "cuda")
        results.append(r3)
    except Exception as e:
        print(f"  INT4 failed: {e}")

    # ── Final summary ─────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<20} {'Tok/s':>8} {'Latency':>10} {'VRAM':>8}")
    print(f"  {'─'*50}")
    for r in results:
        name = r["name"].replace("\n", " ")
        print(f"  {name:<20} {r['avg_tps']:>8.1f} {r['avg_latency']:>9.0f}ms {r['vram_gb']:>7.2f}GB")

    if len(results) >= 2:
        best = max(results, key=lambda x: x["avg_tps"])
        base = results[0]
        print(f"\n  Best config  : {best['name'].replace(chr(10),' ')}")
        print(f"  Speedup vs FP16 CPU : {best['avg_tps']/base['avg_tps']:.1f}x")

    plot_results(results)
