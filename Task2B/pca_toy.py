import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 200 made-up people: arm span tracks height, plus some individual variation
height = rng.normal(170, 10, 200)                 # cm
arm_span = height + rng.normal(0, 3, 200)         # cm
X = np.column_stack([height, arm_span])           # table: 200 rows x 2 columns

# PCA via SVD
X_centered = X - X.mean(axis=0)                   # subtract the average person
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

explained = S**2 / np.sum(S**2)
print("Principal components (rows):\n", np.round(Vt, 3))
print("Fraction of variance explained:", np.round(explained, 3))

scores = U * S                                    # each person's coordinates along PC1, PC2

# Plot the data with the two component directions drawn from the center
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(X_centered[:, 0], X_centered[:, 1], s=10, alpha=0.6)
for k, color in zip([0, 1], ["tab:red", "tab:green"]):
    length = 2 * S[k] / np.sqrt(len(X))           # arrow length ~ spread along that direction
    axes[0].arrow(0, 0, Vt[k, 0] * length, Vt[k, 1] * length, width=0.4, color=color,
                  label=f"PC{k + 1} ({explained[k]:.0%})")
axes[0].set_xlabel("Height − average (cm)")
axes[0].set_ylabel("Arm span − average (cm)")
axes[0].set_aspect("equal")
axes[0].legend()
axes[0].set_title("Original measurements")

axes[1].scatter(scores[:, 0], scores[:, 1], s=10, alpha=0.6)
axes[1].set_xlabel("PC1 score")
axes[1].set_ylabel("PC2 score")
axes[1].set_aspect("equal")
axes[1].set_title("Same data in PC coordinates")
plt.tight_layout()
plt.show()