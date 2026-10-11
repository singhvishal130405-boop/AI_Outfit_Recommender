import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor("#0d1117")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1280); ax.set_ylim(0, 640); ax.axis("off")

ax.text(80, 490, "AI Outfit Recommender", color="white", fontsize=46, fontweight="bold")
ax.text(80, 430, "From a clothing photo to matching outfit pieces",
        color="#9da7b3", fontsize=20)

labels = ["Clothing\nimage", "Model 1\nMobileNetV2 CNN",
          "Model 2\nCompatibility network", "Outfit\nrecommendations"]
xs = [80, 376, 673, 970]
for x, lab in zip(xs, labels):
    ax.add_patch(FancyBboxPatch((x, 200), 230, 110, boxstyle="round,pad=0,rounding_size=14",
                                fc="#161b22", ec="#30363d", lw=2))
    ax.text(x + 115, 255, lab, color="white", fontsize=15, ha="center", va="center")
for x in xs[:-1]:
    ax.annotate("", xy=(x + 230 + 58, 255), xytext=(x + 230 + 8, 255),
                arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2.5))

ax.text(80, 110, "TensorFlow  |  Keras  |  Python", color="#58a6ff", fontsize=18)
ax.text(80, 70, "Article accuracy 85.6%  |  Usage 90.8%  |  Rule-based compatibility labels",
        color="#9da7b3", fontsize=14)

fig.savefig("assets/social_preview.png", dpi=100, facecolor=fig.get_facecolor())
print("saved assets/social_preview.png")