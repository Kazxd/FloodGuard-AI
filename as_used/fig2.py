import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = [0, 1, 2, 5, 10, 20]
# (extra distance %, exposure avoided %) for w = 1, 2, 5, 10, 20; three seeds (26, 123, 2026), all start points
data = {
 "Moderate": [
  [(1.2147,7.1306),(2.0707,8.9355),(4.4392,11.3907),(5.3893,11.8040),(6.3543,11.9978)],
  [(1.1999,7.0443),(1.8946,8.3352),(4.6034,11.1827),(5.7267,11.7007),(6.8191,11.9434)],
  [(1.3060,7.9372),(1.7346,8.8368),(3.8583,10.9089),(4.6729,11.2812),(5.8585,11.5420)]],
 "Heavy": [
  [(0.5377,2.3869),(1.0328,2.9922),(2.5149,3.6441),(2.8533,3.7109),(3.4325,3.7808)],
  [(0.6608,2.5508),(1.0994,3.0861),(2.4601,3.7167),(2.8908,3.8036),(3.2640,3.8464)],
  [(0.6703,3.2895),(1.0781,3.7707),(2.6572,4.4978),(3.1106,4.5803),(3.8312,4.6728)]],
 "Extreme": [
  [(0.2744,0.6464),(0.3524,0.7266),(0.3908,0.7408),(0.4110,0.7440),(0.4855,0.7498)],
  [(0.1976,0.4793),(0.2609,0.5315),(0.4170,0.5918),(0.4372,0.5951),(0.5862,0.6069)],
  [(0.3199,0.7920),(0.3903,0.8630),(0.4688,0.8907),(0.4925,0.8946),(0.5718,0.9008)]],
}
means = {}
for name, seeds in data.items():
    pts = [(0.0, 0.0)]
    for i in range(5):
        pts.append((st.mean(s[i][0] for s in seeds), st.mean(s[i][1] for s in seeds)))
    means[name] = pts

fig, ax = plt.subplots(figsize=(3.4, 2.7))
colors = {"Moderate": "#1f4e9c", "Heavy": "#d9801a", "Extreme": "#2e8b57"}
markers = {"Moderate": "o", "Heavy": "s", "Extreme": "^"}
for name, pts in means.items():
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker=markers[name], ms=3.5, lw=1.2, color=colors[name], label=name)
for name in ("Moderate", "Heavy"):
    pts = means[name]
    for idx, label in ((1, "1"), (3, "5"), (5, "20")):
        ax.annotate(f"w={label}", pts[idx], textcoords="offset points", xytext=(4, -9 if idx == 1 else 3), fontsize=6)
ax.set_xlim(-0.2, 7.6)
ax.set_ylim(-0.5, 13.2)
ax.set_xlabel("Extra distance vs shortest route (%)", fontsize=7.5)
ax.set_ylabel("Flood exposure avoided (%)", fontsize=7.5)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=7, loc="lower right", frameon=True)
fig.tight_layout(pad=0.4)
fig.savefig("fig2_tradeoff.png", dpi=300)
for k, v in means.items():
    print(k, [(round(a, 2), round(b, 2)) for a, b in v])
