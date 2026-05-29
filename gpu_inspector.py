import torch

p = torch.cuda.get_device_properties(0)

print(f"\nGPU: {p.name}")
print(f"{'='*50}")

for attr in sorted(dir(p)):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(p, attr)
        if callable(val):
            continue
        print(f"  {attr:<45} {val}")
    except:
        pass

print(f"\n── DERIVED ──")
print(f"  total_warps:     {p.multi_processor_count * (p.max_threads_per_multi_processor // 32)}")
print(f"  peak_bandwidth:  {p.memory_clock_rate * 1000 * p.memory_bus_width / 8 / 1e9:.0f} GB/s")
