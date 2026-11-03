import json
r=json.load(open('out/validation.json'))
print({k:r[k] for k in ['n_edges','km','n_blocks','explicit_negative_edges_2015']})
f=lambda x:'-' if x is None else (f'{x:.3f}' if isinstance(x,float) else str(x))
for ev,d in r['events'].items():
  print('\n==',ev,'pos',d['positives'],'prev %.3f'%d['prevalence'],'scen',d['scenario_used'])
  for m,v in d['models'].items():
    ci=v.get('auc_ci'); dc=v.get('diff_ci')
    print(f"{m:30s} AUC {v['roc_auc']:.3f} [{f(ci[0]) if ci else ''},{f(ci[1]) if ci else ''}] AP {v['pr_auc']:.3f} Brier {f(v.get('brier'))} ECE {f(v.get('ece'))} slope {f(v.get('calib_slope'))} int {f(v.get('calib_intercept'))} dRF {f(v.get('diff_vs_ref'))} [{f(dc[0]) if dc else ''},{f(dc[1]) if dc else ''}] p {f(v.get('p_two_sided'))}")
  print('RF hist seeds min/med/max',[round(x,3) for x in d['rf_auc_over_20_history_seeds']],'by scen',{k:round(v,3) for k,v in d['rf_auc_by_scenario'].items()})
for s in r['label_sensitivity_2015']: print({k:round(v,3) if isinstance(v,float) else v for k,v in s.items()})
