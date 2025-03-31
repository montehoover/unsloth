import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from brokenaxes import brokenaxes

SIZE_LIMIT = 2000

fig = plt.figure(figsize=(4.5, 2.75))
bax = brokenaxes(xlims=((0, SIZE_LIMIT), (SIZE_LIMIT * 1.15, SIZE_LIMIT * 1.15**3)), hspace=.05)

bax.axs[0].errorbar(
  [1.5, 3.0, 7.0],
  [61.35, 66.57, 85.61],
  [4.50, 3.68, 3.68],
  label='Qwen2.5 Base',
  color='#1f77b4',
  capsize=5,
  ms=4,
  fmt='D',
  zorder=6
)

bax.axs[0].errorbar(
  [1.5, 3.0, 7.0],
  [75.48, 92.96, 94.65],
  [4.32, 2.11, 2.86],
  label='Qwen2.5 SFT',
  color='#e377c2',
  capsize=5,
  ms=4,
  fmt='D',
  zorder=7
)

bax.axs[0].errorbar(
  [1.5, 3.0, 7.0],
  [95.58, 95.81, 96.13],
  [0.55, 0.38, 1.21],
  label='Qwen2.5 GRPO',
  color='#2ca02c',
  capsize=5,
  ms=4,
  fmt='D',
  zorder=8
)

# Gemma2 2B: 17.10% w/ std 2.44%. 75% of time doesn't follow format
bax.axs[0].errorbar(
  [9.0, 27.0],
  [91.23, 92.19],
  [1.15, 1.34],
  label='Gemma2',
  color='#bcbd22',
  capsize=5,
  ms=4,
  fmt='v',
  zorder=5
)

bax.axs[0].errorbar(
  [7.3, 24.0, 46.7, 141],
  [85.03, 86.58, 74.45, 82.58],
  [2.32, 1.95, 2.58, 1.66],
  label='Mistral',
  color='#ff7f0e',
  capsize=5,
  ms=4,
  fmt='s',
  zorder=4
)

bax.axs[0].errorbar(
  [3.0, 8.0, 70.0, 405],
  [75.74, 81.10, 86.39, 90.65],
  [3.14, 2.61, 1.02, 0.97],
  label='Llama3',
  color='#17becf',
  capsize=5,
  ms=4,
  fmt='^',
  zorder=3
)


# bax.axs[0].errorbar(
#   [1.0, 8.0],
#   [43.71, 54.19],
#   [2.79, 1.29],
#   label='LlamaGuard 3',
#   color='#8c564b',
#   capsize=5,
#   ms=4,
#   fmt='o',
#   zorder=2
# )

# bax.axs[0].errorbar(
#   [1.0, 3.0, 8.0],
#   [42.90, 42.71, 42.19],
#   [0.43, 0.48, 0.43],
#   label='GuardReasoner',
#   color='#8c564b',
#   capsize=5,
#   ms=4,
#   fmt='o',
#   zorder=2
# )

bax.axs[0].errorbar(
  [673],
  [95.29],
  [1.15],
  label='DeepSeek R1',
  color='#7f7f7f',
  capsize=5,
  ms=4,
  fmt='o',
  zorder=2
)

bax.axs[0].legend(loc='lower right', fontsize=6)

bax.axs[1].errorbar(
  [SIZE_LIMIT*1.15**2],
  [97.90],
  [1.22/2],
  label='GPT-4o-mini',
  color='#9467bd',
  capsize=5,
  ms=4,
  fmt='^',
  zorder=2
)

bax.axs[1].errorbar(
  [SIZE_LIMIT*1.15**2],
  [95.55],
  [2.44/2],
  label='GPT-4o',
  color='#9467bd',
  capsize=5,
  ms=4,
  fmt='o',
  zorder=2
)

bax.axs[1].legend(loc='lower right', fontsize=6)

# for size, accuracy, error, name, label in points:
#   if accuracy is None: continue
#   bax.errorbar(size, accuracy, color='#1f77b4', yerr=error/2, fmt='o', ms=4, capsize=5, label=label)


bax.axhline(y=97.90, color='#9467bd', linestyle='--', linewidth=1)
bax.grid()
bax.axs[0].set_xlim(1, SIZE_LIMIT)
bax.axs[0].set_ylim(55, 100)
bax.axs[1].set_ylim(55, 100)
bax.axs[0].set_xscale('log')
# bax.axs[0].set_xticks([1.5, 3.0, 7.0], ['1.5B', '3B', '7B'], rotation=30)
bax.axs[0].set_xticks([1.0, 3.0, 9.0, 27.0, 70.0, 200, 600], ['1B', '3B', '9B', '27B', '70B', '200B', '600B'], rotation=30)
bax.axs[0].minorticks_off()
bax.axs[1].set_xticks([SIZE_LIMIT*1.15**2], ['Closed-\nSource'], rotation=30)
bax.set_xlabel('Model Size', labelpad=30)
bax.set_ylabel('Test Set Accuracy')
fig.tight_layout()
for handle in bax.diag_handles:
    handle.remove()
plt.savefig('pareto.png', dpi=400)