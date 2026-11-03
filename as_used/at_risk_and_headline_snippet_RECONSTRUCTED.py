# RECONSTRUCTED: this inline snippet was run once in a shell and never saved. It is reproduced here
# verbatim from the conversation. Numbers were typed in by hand from the three summary.csv files
# (seeds 26, 123, 2026). Replace by experiments/aggregate_seeds.py, which reads the CSVs directly.
import statistics as st
# (extra_dist, exposure_avoided, base_risk, w5_risk) at w=5, all starts; seeds 26, 123, 2026
all5 = {
 "moderate": [(4.4392,11.3907,0.320201,0.271668),(4.6034,11.1827,0.315023,0.267482),(3.8583,10.9089,0.319926,0.274437)],
 "heavy":    [(2.5149,3.6441,0.632007,0.594036),(2.4601,3.7167,0.622342,0.584824),(2.6572,4.4978,0.633253,0.589116)],
 "extreme":  [(0.3908,0.7408,0.800439,0.791417),(0.4170,0.5918,0.792110,0.784152),(0.4688,0.8907,0.800149,0.789321)],
}
for k,v in all5.items():
    for i,name in enumerate(["extra","avoided","base","w5"]):
        xs=[t[i] for t in v]
        print(k,name,round(st.mean(xs),3),round(min(xs),3),round(max(xs),3))
    print(k,"risk drop",round(st.mean([t[2]-t[3] for t in v]),4), "ratio", round(st.mean([t[1]/t[0] for t in v]),2))
at = {"moderate":[(50,4.5635,9.2885),(46,4.7953,8.7538),(56,3.2987,9.7345)],
      "heavy":[(251,2.5436,3.6973),(243,2.5375,3.7314),(246,2.6808,4.2881)]}
for k,v in at.items():
    print(k,"at-risk n",sum(t[0] for t in v),"extra",round(st.mean(t[1] for t in v),2),"avoid",round(st.mean(t[2] for t in v),2),
          min(t[2] for t in v),max(t[2] for t in v))
for w,vals in {"w1":[(1.2147,7.1306),(1.1999,7.0443),(1.3060,7.9372)],"w20":[(6.3543,11.9978),(6.8191,11.9434),(5.8585,11.5420)]}.items():
    print("moderate",w,round(st.mean(a for a,b in vals),2),round(st.mean(b for a,b in vals),2))
