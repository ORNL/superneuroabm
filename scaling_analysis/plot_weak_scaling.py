#!/usr/bin/env python3
"""
Plot weak scaling results from job 3939840
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from job 3939840
nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
gpus = nodes * 8
neurons = gpus * 5000  # 5000 neurons per GPU
sim_time = np.array([23.732, 29.335, 33.934, 42.308, 47.950,
                      239.507, 69.448, 113.992, 88.659, 97.406])

# Calculate weak scaling efficiency (relative to 1 node baseline)
baseline = sim_time[0]
efficiency = (baseline / sim_time) * 100

# Ideal weak scaling (constant time)
ideal_time = np.ones_like(sim_time) * baseline

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Simulation Time vs GPUs
ax1.plot(gpus, sim_time, 'o-', linewidth=2, markersize=8,
         label='Actual (non-contiguous nodes)', color='#d62728')
ax1.plot(gpus, ideal_time, '--', linewidth=2, alpha=0.7,
         label='Ideal weak scaling', color='#2ca02c')

# Highlight the node allocation issue
ax1.axvspan(40, 48, alpha=0.2, color='red',
            label='Non-contiguous allocation spike')

# Smooth region (1-5 nodes)
ax1.plot(gpus[:5], sim_time[:5], 'o-', linewidth=3, markersize=10,
         color='#1f77b4', label='Contiguous portion (1-5 nodes)')

ax1.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax1.set_ylabel('Simulation Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Weak Scaling: Simulation Time\n(5000 neurons/GPU, 10 ticks)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='upper left')
ax1.set_xticks(gpus)

# Add annotation for the spike
ax1.annotate('Node allocation issue\n(5× slowdown)',
             xy=(48, 239.507), xytext=(60, 200),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Plot 2: Weak Scaling Efficiency
ax2.plot(gpus, efficiency, 'o-', linewidth=2, markersize=8,
         color='#ff7f0e', label='Weak scaling efficiency')
ax2.axhline(y=100, color='#2ca02c', linestyle='--', linewidth=2,
            label='Ideal (100%)', alpha=0.7)
ax2.axhline(y=80, color='gray', linestyle=':', linewidth=1.5,
            label='80% threshold', alpha=0.5)

# Highlight good scaling region
ax2.plot(gpus[:5], efficiency[:5], 'o-', linewidth=3, markersize=10,
         color='#1f77b4', label='Good scaling (1-5 nodes)')

ax2.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weak Scaling Efficiency (%)', fontsize=12, fontweight='bold')
ax2.set_title('Weak Scaling Efficiency\n(η = T₁/Tₙ × 100%)',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xticks(gpus)
ax2.set_ylim(0, 110)

# Add efficiency values as text
for i, (g, e) in enumerate(zip(gpus, efficiency)):
    if i < 5:  # Good scaling region
        ax2.text(g, e+3, f'{e:.1f}%', ha='center', fontsize=9,
                fontweight='bold', color='#1f77b4')
    elif i == 5:  # Spike
        ax2.text(g, e+3, f'{e:.1f}%', ha='center', fontsize=9,
                fontweight='bold', color='red')

plt.tight_layout()

# Save figure
output_path = '/lustre/orion/lrn088/proj-shared/objective3/xxz/superneuroabm/tests/output/weak_scaling_3939840.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also create a summary table
print("\n" + "="*70)
print("WEAK SCALING RESULTS SUMMARY (Job 3939840)")
print("="*70)
print(f"{'Nodes':<8} {'GPUs':<8} {'Neurons':<10} {'Time (s)':<12} {'Efficiency':<12} {'Notes'}")
print("-"*70)
for i, (n, g, neu, t, e) in enumerate(zip(nodes, gpus, neurons, sim_time, efficiency)):
    note = ""
    if i < 5:
        note = "✓ Good"
    elif i == 5:
        note = "✗ Non-contiguous"
    else:
        note = "~ Variable"
    print(f"{n:<8} {g:<8} {neu:<10} {t:<12.3f} {e:<12.1f} {note}")
print("="*70)

# Calculate statistics for 1-5 node range (good scaling)
good_efficiency = efficiency[:5]
print(f"\n1-5 Nodes Statistics (Contiguous Region):")
print(f"  Average efficiency: {np.mean(good_efficiency):.1f}%")
print(f"  Min efficiency: {np.min(good_efficiency):.1f}%")
print(f"  Time range: {sim_time[0]:.1f}s - {sim_time[4]:.1f}s")
print(f"  Slowdown factor: {sim_time[4]/sim_time[0]:.2f}x (for 5x workers)")

print(f"\nNode 6 (Non-contiguous allocation):")
print(f"  Efficiency: {efficiency[5]:.1f}%")
print(f"  Time: {sim_time[5]:.1f}s")
print(f"  Slowdown vs Node 5: {sim_time[5]/sim_time[4]:.1f}x")
print(f"  Nodes used: 10376,10397,10400,10403-10409 (fragmented!)")

plt.show()
