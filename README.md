"""
sorting_benchmark.py
====================
Experimental Comparison of Sorting Algorithms
Algorithms Course — Python 3.12

Implements all 10 sorting algorithms studied in the paper, benchmarks them
across 3 input distributions × 6 input sizes, and writes:
  - results.csv          (180 rows: 10 algs × 6 sizes × 3 distributions)
  - results_report.txt   (ASCII bar-chart report matching the paper)

Run:
    python sorting_benchmark.py

Expected runtime: 10–20 minutes on a single modern CPU core (quadratic
algorithms at n = 10,000 dominate the total time).
"""

import time
import random
import csv
import sys
import math

# ── Recursion limit for Merge Sort / Quick Sort at large n ──────────────────
sys.setrecursionlimit(50_000)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Experimental parameters (must match the paper) ───────────────────────────
INPUT_SIZES       = [100, 500, 1_000, 2_000, 5_000, 10_000]
DISTRIBUTIONS     = ["random", "sorted", "reverse"]
NUM_TRIALS        = 5          # 5-run average
KEY_RANGE_FACTOR  = 10         # keys drawn from [0, 10n)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. SORTING ALGORITHM IMPLEMENTATIONS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── 1.1  Quadratic algorithms ─────────────────────────────────────────────

def bubble_sort(arr):
    """
    Bubble Sort with early-exit flag.
    Best case O(n), average/worst O(n²).
    """
    a = arr[:]
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:          # already sorted → O(n) best case
            break
    return a


def selection_sort(arr):
    """
    Selection Sort.
    Always Θ(n²/2) comparisons; O(n) swaps.
    """
    a = arr[:]
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a


def insertion_sort(arr):
    """
    Insertion Sort.
    O(n) best case (sorted input), O(n²) worst case (reverse).
    """
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# ── 1.2  Sub-quadratic comparison sorts ───────────────────────────────────

def merge_sort(arr):
    """
    Merge Sort (divide-and-conquer).
    Θ(n log n) all cases; O(n) auxiliary space.
    """
    if len(arr) <= 1:
        return arr[:]
    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """
    Quick Sort with median-of-three pivot selection.
    Average O(n log n); worst case O(n²) for adversarial input (rare with
    median-of-three).
    """
    a = arr[:]
    _quick_sort_helper(a, 0, len(a) - 1)
    return a

def _median_of_three(a, lo, hi):
    mid = (lo + hi) // 2
    # Sort lo, mid, hi in-place so a[mid] is the median
    if a[lo] > a[mid]:
        a[lo], a[mid] = a[mid], a[lo]
    if a[lo] > a[hi]:
        a[lo], a[hi] = a[hi], a[lo]
    if a[mid] > a[hi]:
        a[mid], a[hi] = a[hi], a[mid]
    # Place pivot at hi-1
    a[mid], a[hi - 1] = a[hi - 1], a[mid]
    return a[hi - 1]

def _quick_sort_helper(a, lo, hi):
    if hi - lo < 1:
        return
    if hi - lo == 1:
        if a[lo] > a[hi]:
            a[lo], a[hi] = a[hi], a[lo]
        return
    pivot = _median_of_three(a, lo, hi)
    i = lo
    j = hi - 1
    while True:
        i += 1
        while a[i] < pivot:
            i += 1
        j -= 1
        while a[j] > pivot:
            j -= 1
        if i >= j:
            break
        a[i], a[j] = a[j], a[i]
    # Restore pivot
    a[i], a[hi - 1] = a[hi - 1], a[i]
    _quick_sort_helper(a, lo, i - 1)
    _quick_sort_helper(a, i + 1, hi)


def heap_sort(arr):
    """
    Heap Sort (max-heap).
    O(n log n) all cases; O(1) auxiliary space.
    """
    a = arr[:]
    n = len(a)
    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(a, i, n)
    # Extract elements
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        _sift_down(a, 0, i)
    return a

def _sift_down(a, root, end):
    while True:
        child = 2 * root + 1
        if child >= end:
            break
        if child + 1 < end and a[child] < a[child + 1]:
            child += 1
        if a[root] < a[child]:
            a[root], a[child] = a[child], a[root]
            root = child
        else:
            break


def shell_sort(arr):
    """
    Shell Sort (halving gap sequence).
    Average O(n^(3/2)); in-place, no recursion.
    """
    a   = arr[:]
    n   = len(a)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = a[i]
            j    = i
            while j >= gap and a[j - gap] > temp:
                a[j] = a[j - gap]
                j   -= gap
            a[j] = temp
        gap //= 2
    return a


# ── 1.3  Non-comparison integer sorts ─────────────────────────────────────

def counting_sort(arr):
    """
    Counting Sort.
    O(n + k) time and space; integers only.
    """
    if not arr:
        return []
    a      = arr[:]
    lo, hi = min(a), max(a)
    k      = hi - lo + 1
    count  = [0] * k
    for x in a:
        count[x - lo] += 1
    result = []
    for i, c in enumerate(count):
        result.extend([i + lo] * c)
    return result


def radix_sort(arr):
    """
    Radix Sort — LSD, base-10, using Counting Sort as stable sub-routine.
    O(nk) where k = number of digits; effectively O(n) for bounded integers.
    """
    if not arr:
        return []
    a      = arr[:]
    max_val = max(a)
    exp    = 1
    while max_val // exp > 0:
        a   = _counting_sort_by_digit(a, exp)
        exp *= 10
    return a

def _counting_sort_by_digit(a, exp):
    n      = len(a)
    output = [0] * n
    count  = [0] * 10
    for x in a:
        idx = (x // exp) % 10
        count[idx] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        idx           = (a[i] // exp) % 10
        count[idx]   -= 1
        output[count[idx]] = a[i]
    return output


# ── 1.4  Hybrid — Python built-in Timsort ─────────────────────────────────

def timsort(arr):
    """
    Python's built-in sort (Timsort).
    O(n) best case, O(n log n) average/worst; implemented in optimised C.
    """
    return sorted(arr)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. ALGORITHM REGISTRY                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ALGORITHMS = {
    "Bubble Sort"    : bubble_sort,
    "Selection Sort" : selection_sort,
    "Insertion Sort" : insertion_sort,
    "Merge Sort"     : merge_sort,
    "Quick Sort"     : quick_sort,
    "Heap Sort"      : heap_sort,
    "Shell Sort"     : shell_sort,
    "Counting Sort"  : counting_sort,
    "Radix Sort"     : radix_sort,
    "Tim Sort"       : timsort,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. INPUT GENERATION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_input(n, distribution, rng):
    """
    Generate a list of n integers according to the specified distribution.
    Keys are drawn from [0, 10n) as described in Section 3.2 of the paper.
    """
    population = list(range(KEY_RANGE_FACTOR * n))
    base = rng.sample(population, n)
    if distribution == "random":
        return base
    elif distribution == "sorted":
        return sorted(base)
    elif distribution == "reverse":
        return sorted(base, reverse=True)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. BENCHMARKING HARNESS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def time_algorithm(sort_fn, arr, trials=NUM_TRIALS):
    """
    Time sort_fn on a fresh copy of arr for `trials` runs.
    Returns arithmetic mean in milliseconds.
    """
    times = []
    for _ in range(trials):
        data  = arr[:]          # fresh copy every run
        start = time.perf_counter()
        sort_fn(data)
        end   = time.perf_counter()
        times.append((end - start) * 1_000)   # convert to ms
    return sum(times) / len(times)


def run_benchmark():
    """
    Run the full benchmark matrix:
      10 algorithms × 6 sizes × 3 distributions × 5 trials
    Returns a list of result dicts.
    """
    rng     = random.Random(SEED)
    results = []

    total   = len(ALGORITHMS) * len(INPUT_SIZES) * len(DISTRIBUTIONS)
    done    = 0

    print("=" * 65)
    print("  Experimental Comparison of Sorting Algorithms")
    print("  Algorithms Course — Python Benchmark")
    print("=" * 65)
    print(f"  Algorithms   : {len(ALGORITHMS)}")
    print(f"  Input sizes  : {INPUT_SIZES}")
    print(f"  Distributions: {DISTRIBUTIONS}")
    print(f"  Trials/cell  : {NUM_TRIALS}")
    print(f"  Total cells  : {total}")
    print("=" * 65)
    print()

    for dist in DISTRIBUTIONS:
        print(f"── Distribution: {dist.upper()} ──────────────────────────────")
        for n in INPUT_SIZES:
            arr = generate_input(n, dist, rng)
            for name, fn in ALGORITHMS.items():
                avg_ms = time_algorithm(fn, arr)
                results.append({
                    "algorithm"   : name,
                    "n"           : n,
                    "distribution": dist,
                    "avg_ms"      : round(avg_ms, 3),
                })
                done += 1
                pct  = 100 * done / total
                print(f"  [{pct:5.1f}%]  {name:<16s}  n={n:>6,}  "
                      f"{dist:<8s}  {avg_ms:>10.3f} ms")

    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. CSV OUTPUT                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def write_csv(results, path="results.csv"):
    fieldnames = ["algorithm", "n", "distribution", "avg_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  ✓ Results written to '{path}'  ({len(results)} rows)")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. ASCII REPORT (mirrors paper Figures 1–4)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BAR_WIDTH = 40   # characters

def _bar(value, max_value):
    filled = int(round(BAR_WIDTH * value / max_value)) if max_value > 0 else 0
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def write_report(results, path="results_report.txt"):
    # Index results for easy lookup
    lookup = {}
    for r in results:
        lookup[(r["algorithm"], r["n"], r["distribution"])] = r["avg_ms"]

    alg_names = list(ALGORITHMS.keys())
    lines     = []
    sep       = "─" * 72

    def h(text):
        lines.append(text)

    # ── Figure 1 & 2 & 3: bar charts at n = 10,000 ───────────────────────
    n_large = 10_000
    for dist in DISTRIBUTIONS:
        h("")
        h(f"Figure — {dist.upper()} Input  (n = {n_large:,}, 5-run average)")
        h(sep)
        h(f"  {'Algorithm':<16s}  {'Avg (ms)':>10s}   Relative bar")
        h(sep)

        row_data = [(name, lookup.get((name, n_large, dist), 0.0))
                    for name in alg_names]
        max_val  = max(v for _, v in row_data) or 1.0

        for name, val in sorted(row_data, key=lambda x: x[1]):
            bar = _bar(val, max_val)
            h(f"  {name:<16s}  {val:>10.2f}   {bar}")

        h(sep)
        h(f"  (n = {n_large:,} — bars scaled to maximum value)")
        h("")

    # ── Figure 4: growth table, random input ─────────────────────────────
    h("")
    h("Figure — Growth Table, Random Input (all times in ms)")
    h(sep)
    header = f"  {'Algorithm':<16s}" + "".join(f"  {f'n={s}':>9s}"
                                                 for s in INPUT_SIZES)
    h(header)
    h(sep)
    for name in alg_names:
        row = f"  {name:<16s}"
        for s in INPUT_SIZES:
            val  = lookup.get((name, s, "random"), 0.0)
            row += f"  {val:>9.3f}"
        h(row)
    h(sep)
    h("")

    # ── Summary table at n = 10,000 ───────────────────────────────────────
    h("")
    h("Summary Table — Average Sorting Time (ms) at n = 10,000")
    h(sep)
    h(f"  {'Algorithm':<16s}  {'Random':>10s}  {'Sorted':>10s}  {'Reverse':>10s}")
    h(sep)
    for name in alg_names:
        r  = lookup.get((name, 10_000, "random"),  0.0)
        s  = lookup.get((name, 10_000, "sorted"),  0.0)
        rv = lookup.get((name, 10_000, "reverse"), 0.0)
        h(f"  {name:<16s}  {r:>10.2f}  {s:>10.2f}  {rv:>10.2f}")
    h(sep)
    h("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  ✓ Report  written to '{path}'")
    print()
    print(text)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. CORRECTNESS VERIFICATION                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def verify_all():
    """
    Quick smoke test: each algorithm must produce the same result as
    Python's built-in sorted() on a small random list.
    """
    rng  = random.Random(0)
    test = rng.sample(range(1_000), 50)
    ref  = sorted(test)
    ok   = True
    for name, fn in ALGORITHMS.items():
        out = fn(test)
        if out != ref:
            print(f"  ✗ CORRECTNESS FAIL: {name}")
            ok = False
    if ok:
        print("  ✓ All 10 algorithms passed correctness check.")
    return ok


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. MAIN                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print()
    print("  Running correctness verification …")
    if not verify_all():s
        sys.exit(1)
    print()

    results = run_benchmark()
    write_csv(results)
    write_report(results)

    print("  Done.")