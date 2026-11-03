import copy, re, statistics as st
from lxml import etree
import fig2  # reuses the averaged data (also rewrites fig2_tradeoff.png)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
ns = {"w": W}
def q(tag): return "{%s}%s" % (W, tag)

tree = etree.parse("unpacked/word/document.xml")
root = tree.getroot()
body = root.find("w:body", ns)
B = list(body)          # original block order, indices as listed earlier

def ptext(p): return "".join(p.itertext())

def set_text(p, text):
    runs = p.findall("w:r", ns)
    r0 = runs[0]
    for r in runs[1:]: p.remove(r)
    for ch in list(r0):
        if ch.tag != q("rPr"): r0.remove(ch)
    t = etree.SubElement(r0, q("t")); t.text = text; t.set("{%s}space" % XML, "preserve")

def set_run_text(r, text):
    for ch in list(r):
        if ch.tag != q("rPr"): r.remove(ch)
    t = etree.SubElement(r, q("t")); t.text = text; t.set("{%s}space" % XML, "preserve")

def clone_after(ref, template, text):
    new = copy.deepcopy(template)
    set_text(new, text)
    ref.addnext(new)
    return new

def append_text(p, extra):
    last = p.findall("w:r", ns)[-1]
    t = last.find("w:t", ns)
    t.text = t.text + extra
    t.set("{%s}space" % XML, "preserve")

BODY, SUB, SECT = B[7], B[30], B[29]       # body paragraph, subsection heading, section heading templates

# ---------------- Abstract and keywords ----------------
abstract = ("Flooding disrupts road networks exactly when residents most need to evacuate, yet shortest-distance route planners ignore flood risk. "
 "This paper presents FloodGuard Lite, a compact system that couples supervised machine learning with graph search to recommend safer evacuation routes. "
 "A classifier estimates, for every road segment, the probability of flooding from rainfall, river level, elevation, distance from the river, flood history and road class; "
 "these probabilities become edge costs, and Dijkstra and A-star search find the route that best balances distance against risk, either to a chosen destination or to the best of several automatically selected shelters. "
 "The study area is Velachery, Chennai, modelled from OpenStreetMap roads and waterways with elevation taken from a digital elevation model. "
 "Because no public dataset provides per-road flood labels, the classifier is trained on a documented synthetic dataset of 6,000 samples whose true flood probabilities are known; a Random Forest reaches a ROC AUC of 0.798 against a theoretical ceiling of 0.841. "
 "On the Velachery network, with 900 random start points per scenario, risk-aware routing with flood weight w = 5 reduced the expected flooded distance by 11.2%, 4.0% and 0.7% under moderate, heavy and extreme rainfall scenarios, at the cost of routes only 4.3%, 2.5% and 0.4% longer. "
 "Gains flattened beyond w = 5 and shrank as severity rose, because predicted risk saturates across the network. "
 "A-star returned the same optimal cost as Dijkstra while expanding about 24% fewer nodes. "
 "All risk results hold under the simulated risk model and do not measure real-world flood prediction accuracy.")
set_run_text(B[4].findall("w:r", ns)[1], abstract)
set_run_text(B[5].findall("w:r", ns)[1], "flood prediction, evacuation routing, random forest, A-star search, shortest path, digital elevation model, decision support")

# ---------------- Introduction: contributions ----------------
intro = ("The prototype works in two coupled stages. A classifier, compared between a Decision Tree and a Random Forest, estimates a flood probability for each road segment. "
 "These probabilities then become edge costs on a graph of the road network, and the safest route between a start and a destination is found with Dijkstra\u2019s algorithm and A* search under a configurable cost function. "
 "Results are shown in a Streamlit dashboard with a Folium map. The main contributions are: (1) an end-to-end pipeline that links per-road flood probability to route cost; "
 "(2) a documented synthetic dataset that stores the true flood probability of every sample, giving a theoretical ceiling for model evaluation; "
 "(3) a flood-aware cost function that keeps the straight-line heuristic admissible, so A* remains optimal; "
 "(4) a quantitative evaluation of risk-aware routing on the real Velachery road network, with elevation from a digital elevation model, that reports the distance cost and the flood exposure avoided under three rainfall scenarios; "
 "and (5) an automated test suite that checks these properties.")
set_text(B[9], intro)

# ---------------- Objectives ----------------
t23 = ptext(B[23]); set_text(B[23], t23.replace("five objectives", "six objectives"))
t27 = ptext(B[27]); set_text(B[27], re.sub(r";\s*and\s*$", ";", t27))
t28 = ptext(B[28]); t28 = t28.rstrip(); t28 = t28[:-1] + "; and" if t28.endswith(".") else t28 + "; and"; set_text(B[28], t28)
clone_after(B[28], B[28], "6) to quantify, on the real Velachery road network, the extra distance and the flood exposure avoided by risk-aware routing under different rainfall scenarios.")

# ---------------- Method ----------------
append_text(B[31], " For Velachery, the resulting graph has 1,341 junctions and 3,288 directed road segments.")
set_text(B[33],
 "Distance from the river, elevation and historical flood frequency are not stored in the road graph, so they are attached to each road. "
 "Distance is measured from the road midpoint to the nearest waterway or water body, using the rivers, canals, streams, drains, lakes and wetlands that OpenStreetMap records within 2 km of the study area; "
 "the outline of each lake or wetland is treated as a line. If no waterway data can be downloaded, a synthetic river along the southern edge of the area is used. "
 "Elevation is read from a digital elevation model (DEM) obtained from OpenTopography [21], sampled by bilinear interpolation at the road midpoint and clipped to 0.5\u201325 m, the range used in training; "
 "over the junctions of the Velachery network it spans about 2 to 18 m with a mean of about 9 m. If no DEM file is present, elevation is generated synthetically so that the system and its tests still run offline. "
 "No public source provides per-road flood history, so historical flood frequency remains synthetic: it is a Poisson count whose rate is higher for low ground and for roads near the river. "
 "A two-way road is one physical road, so both of its directed edges receive identical feature values.")
append_text(B[35], " The training samples are independent of the Velachery graph: their elevation is drawn from a normal distribution with mean 8 m and standard deviation 3 m, which is close to the range of the real terrain but is not a sample of it.")

# citation renumbering in Method (new [21] DEM shifts old [21]-[24])
shift = {"21": "22", "22": "23", "23": "24", "24": "25"}
for i in (40, 45):
    for t in B[i].iter(q("t")):
        t.text = re.sub(r"\[(2[1-4])\]", lambda m: "[%s]" % shift[m.group(1)], t.text or "")

# H. Routing experiment design (after Dashboard paragraph B[51])
h_head = clone_after(B[51], SUB, "H. Routing Experiment Design")
h_par = clone_after(h_head, BODY,
 "To quantify the effect of risk-aware routing on the real network, three weather scenarios are simulated: moderate (100 mm of rainfall, river 2 m above normal), heavy (200 mm, 4 m) and extreme (300 mm, 6 m). "
 "For each scenario the model assigns a flood probability to every road, and random start junctions (excluding the five shelters) are routed to the lowest-cost shelter for flood weights w \u2208 {0, 1, 2, 5, 10, 20}; w = 0 is the plain shortest route and serves as the baseline. "
 "Each route is scored by its length and by its exposure, the sum over its segments of length times predicted flood probability, which is the expected number of flooded metres travelled. "
 "We report the extra distance and the share of baseline exposure avoided, pooled over routes, for all starts and for at-risk starts, whose outgoing roads have a mean flood probability of at least 0.5. "
 "The experiment uses three random seeds (26, 123 and 2026) of 300 start points each. Dijkstra and A* are compared separately on 100 start\u2013shelter pairs in the heavy scenario at w = 5.")

# ---------------- Results ----------------
t59 = ptext(B[59]).replace("about 95% of the ceiling of 0.841", "which is 87% of the way from chance level (0.5) to the ceiling of 0.841")
set_text(B[59], t59)
set_text(B[60], "B. Routing Correctness Tests")

c_head = clone_after(B[61], SUB, "C. Risk-Aware Routing on the Velachery Network")
c1 = clone_after(c_head, BODY,
 "Table II and Fig. 2 summarise the routing experiment on the Velachery network (1,341 junctions, 3,288 directed segments, five shelters), which combines OpenStreetMap roads and waterways with DEM elevation. "
 "Values are means over three seeds of 300 start points each, that is, 900 routes per scenario.")

# Table II (clone of Table I)
cap1 = copy.deepcopy(B[55]); set_text(cap1, "TABLE II"); c1.addnext(cap1)
cap2 = copy.deepcopy(B[56]); set_text(cap2, "Extra distance and flood exposure avoided versus flood weight"); cap1.addnext(cap2)
tbl = copy.deepcopy(B[57]); cap2.addnext(tbl)
widths = [750] + [805] * 5
for gc, w in zip(tbl.find("w:tblGrid", ns).findall("w:gridCol", ns), widths): gc.set(q("w"), str(w))
rows = tbl.findall("w:tr", ns)
assert len(rows) == 4, len(rows)
header = ["Scenario", "w = 1", "w = 2", "w = 5", "w = 10", "w = 20"]
table = [header]
for name in ("Moderate", "Heavy", "Extreme"):
    pts = fig2.means[name][1:]
    table.append([name] + ["%.1f/%.1f" % (a, b) for a, b in pts])
for row, vals in zip(rows, table):
    for tc, val, w in zip(row.findall("w:tc", ns), vals, widths):
        tc.find("w:tcPr/w:tcW", ns).set(q("w"), str(w))
        set_text(tc.find("w:p", ns), val)
tbl.find("w:tblPr/w:tblW", ns).set(q("w"), str(sum(widths)))
note = copy.deepcopy(B[58]); set_text(note, "Cells give extra distance (%) / flood exposure avoided (%) relative to the shortest route (w = 0); all start points; mean of three seeds."); tbl.addnext(note)

c2 = clone_after(note, BODY,
 "At the default weight w = 5, risk-aware routing reduced the expected flooded distance by 11.2% in the moderate scenario, 4.0% in the heavy scenario and 0.7% in the extreme scenario, at the cost of routes 4.3%, 2.5% and 0.4% longer; "
 "the mean risk along the route fell from 0.318 to 0.271, from 0.629 to 0.589 and from 0.798 to 0.788. "
 "The results were stable across seeds: the exposure avoided at w = 5 ranged from 10.9% to 11.4% in the moderate scenario and from 3.6% to 4.5% in the heavy scenario. "
 "Most of the benefit comes from small weights. In the moderate scenario w = 1 already avoids 7.4% of exposure for 1.2% extra distance, whereas raising w from 5 to 20 adds about 2 points of extra distance for less than 1 further point of exposure avoided. "
 "Exposure never increased, and distance never decreased, as w grew, as Section IV-E implies.")

# Figure 2 (clone of Figure 1 paragraph)
fig_par = copy.deepcopy(B[48]); c2.addnext(fig_par)
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
cx = 2857500; cy = int(cx * 2.7 / 3.4)
for el in fig_par.iter("{%s}extent" % WP, "{%s}ext" % A):
    el.set("cx", str(cx)); el.set("cy", str(cy))
dp = next(fig_par.iter("{%s}docPr" % WP)); dp.set("id", "2"); dp.set("name", "fig2"); dp.set("descr", "Extra distance versus flood exposure avoided"); dp.set("title", "Trade-off")
next(fig_par.iter("{%s}blip" % A)).set("{%s}embed" % R, "rId9")
fig_cap = copy.deepcopy(B[49]); set_text(fig_cap, "Fig. 2. Trade-off between extra distance and flood exposure avoided for w = 0, 1, 2, 5, 10 and 20 (all start points; mean of three seeds)."); fig_par.addnext(fig_cap)

c3 = clone_after(fig_cap, BODY,
 "For at-risk start points the results were similar: at w = 5, 9.3% of exposure was avoided for 4.2% extra distance in the moderate scenario (152 routes in total) and 3.9% for 2.6% in the heavy scenario (740 routes). "
 "In the extreme scenario nearly every start (299 or 300 of 300) was at risk, so the results equal those for all starts. "
 "The benefit shrinks as severity rises because predicted risk saturates: with a mean road risk of about 0.8 in the extreme scenario, few safer alternatives exist. "
 "Detours also lower average exposure without removing the worst crossing; the highest segment risk on a route changed only from 0.82 to 0.81 in the heavy scenario at w = 5.")

d_head = clone_after(c3, SUB, "D. Dijkstra versus A* on the Real Network")
d1 = clone_after(d_head, BODY,
 "On 100 start\u2013shelter pairs in the heavy scenario at w = 5, A* returned exactly the same optimal cost as Dijkstra in every case while expanding 24% fewer nodes (143.9 versus 190.2 on average). "
 "Its median run time was not lower (0.48 ms versus 0.41 ms), because the straight-line heuristic adds a per-node cost that a graph of this size does not repay. "
 "A complete query across all five shelters took about 6\u201316 ms depending on the weight, which is fast enough for interactive use.")

# Software verification: rename and update (original B[62], B[63])
set_text(B[62], "E. Software Verification")
t63 = ptext(B[63])
t63 = t63.replace("The project includes 184 automated tests", "The project includes over 200 automated tests").replace("graph building, cost,", "graph building, DEM sampling, cost,").replace("risk assignment, map output", "risk assignment, the routing experiment, map output")
set_text(B[63], t63)
# move the new C/D subsections before "C. Software Verification": B[62] currently follows B[61]; reorder
sv_head, sv_par = B[62], B[63]
body.remove(sv_head); body.remove(sv_par)
d1.addnext(sv_head); sv_head.addnext(sv_par)

# ---------------- Limitations ----------------
set_text(B[65],
 "The main limitation is that the flood labels and the historical flood frequency are synthetic. The labels are generated from a logistic rule with noise because no public dataset provides per-road flood records for Velachery, "
 "so the classifier scores and the routing benefits are conditional on that rule: they show that the pipeline works end to end, not that it predicts real floods or that the reported exposure reductions would occur in practice. "
 "The road network, waterways and elevation are real, so the study is semi-synthetic. In training, river level rises with rainfall, whereas the dashboard lets the user set them independently, so extreme combinations are extrapolations.")
l2 = clone_after(B[65], BODY,
 "Several further simplifications apply. The DEM is a surface model that includes buildings and vegetation and cannot resolve local drainage or underpasses. "
 "Flood risk is a static snapshot for a given rainfall and river level, with no dynamics, drainage capacity or road closures, and roads are straight segments between junctions. "
 "Shelters are high-ground junctions standing in for real facilities, and the planner considers neither traffic nor travel time. Only one area was studied and only the proportional cost form was evaluated. "
 "Because predicted risk depends strongly on rainfall, which applies everywhere, risk saturates in severe scenarios and limits what any routing method can gain, and detours reduce average exposure without removing the worst crossing. "
 "The flood weight w has no single correct value; the diminishing returns above w = 5 suggest a range of 2 to 5.")
l3 = clone_after(l2, BODY,
 "Future work includes labelling roads from observed flooding, for example with flood extents derived from Sentinel-1 imagery for past Chennai floods, together with spatial cross-validation; "
 "real flood records and live rainfall and gauge feeds; real shelters from OpenStreetMap; a bare-earth elevation model; hard road-blocking thresholds and multi-objective routing; and more models with threshold tuning for higher recall.")

# ---------------- Conclusion ----------------
set_text(B[67],
 "FloodGuard Lite shows how a flood-probability model and a graph search algorithm can be combined into a decision-support tool for evacuation routing, addressing the gap between flood prediction and route planning identified in the literature. "
 "A single cost function ties the two together, keeps A* optimal, and gives the user direct control over the trade-off between distance and risk. "
 "On the Velachery network, risk-aware routing avoided about 11% of the expected flooded distance in a moderate scenario for roughly 4% extra distance, with smaller gains under more severe scenarios where predicted risk is high almost everywhere. "
 "With synthetic labels, the Random Forest recovers most of the learnable signal, and the interactive dashboard makes the behaviour easy to inspect. "
 "Replacing the synthetic labels and flood history with observed flood data is the main step needed to move from a teaching prototype towards a practical tool.")

# ---------------- References ----------------
def ref(i, text): set_text(B[68 + i], "[%d]\t%s" % (i, text))
ref(1, "\u201cPerformance comparison of machine learning methods for flood prediction,\u201d Procedia Computer Science, 2024.")
ref(3, "S. P. Kadiyala and W. L. Woo, \u201cFlood prediction and analysis on the relevance of features using explainable artificial intelligence,\u201d arXiv:2201.05046, 2022.")
ref(4, "\u201cFlood susceptibility prediction using tree-based machine learning models in the GBA,\u201d Sustainable Cities and Society, vol. 97, Art. no. 104744, 2023.")
ref(10, "F. Yuan, Y. Xu, W. Mobley, R. Blessing, H. Farahmand, and S. Dong, \u201cPredicting road flooding risk with machine learning approaches using crowdsourced reports and fine-grained traffic data,\u201d arXiv:2108.13265, 2021.")
ref(12, "B. Li et al., \u201cHigh-resolution flood numerical model and Dijkstra algorithm based risk avoidance routes planning,\u201d Water Resources Management, vol. 37, pp. 3243\u20133258, 2023, doi: 10.1007/s11269-023-03500-5.")
ref(13, "Y. Zhu et al., \u201cOptimal evacuation route planning of urban personnel at different risk levels of flood disasters based on the improved 3D Dijkstra\u2019s algorithm,\u201d Sustainability, vol. 14, no. 16, Art. no. 10250, 2022, doi: 10.3390/su141610250.")
ref(15, "\u201cIntegrated optimization of emergency evacuation routing for dam failure-induced flooding: A coupled flood\u2013road network modeling approach,\u201d Applied Sciences, vol. 15, no. 8, Art. no. 4518, 2025, doi: 10.3390/app15084518.")
ref(17, "S. Ghosh, A. Mallick, A. Chowdhury, K. De Sarkar, and J. Mukherjee, \u201cGraph theory applications for advanced geospatial modelling and decision-making,\u201d Applied Geomatics, vol. 16, no. 4, pp. 799\u2013812, 2024, doi: 10.1007/s12518-024-00586-3.")
ref(18, "\u201cImproving community resiliency and emergency response with artificial intelligence,\u201d arXiv:2005.14212, 2020.")
# renumber old 21-24 -> 22-25 and insert new [21]
old = {21: B[89], 22: B[90], 23: B[91], 24: B[92]}
for n, p in old.items():
    txt = ptext(p); txt = re.sub(r"^\[\d+\]", "[%d]" % (n + 1), txt); set_text(p, txt)
new21 = copy.deepcopy(B[88]); set_text(new21, "[21]\tOpenTopography, \u201cOpenTopography: High-resolution topography data and tools.\u201d [Online]. Available: https://opentopography.org (accessed Oct. 2026).")
B[88].addnext(new21)

tree.write("unpacked/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# relationship for the new image
rels = open("unpacked/word/_rels/document.xml.rels", encoding="utf8").read()
rels = rels.replace("</Relationships>", '<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig2_tradeoff.png"/></Relationships>')
open("unpacked/word/_rels/document.xml.rels", "w", encoding="utf8").write(rels)
print("edited")
