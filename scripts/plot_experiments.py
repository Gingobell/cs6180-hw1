"""Plot recorded runs; rejects incomplete runs and mismatched comparison budgets."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('runs',nargs='+',type=Path)
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
runs=[]
for folder in a.runs:
    c=json.loads((folder/'config.json').read_text())
    s=json.loads((folder/'summary.json').read_text())
    rows=[json.loads(t) for t in (folder/'metrics.jsonl').read_text().splitlines()]
    assert s['status']=='complete' and s['updates']==c['optimizer_updates']
    assert [r['step'] for r in rows if r['kind']=='update']==list(range(1,s['updates']+1))
    runs.append((folder,c,s,[r for r in rows if r['kind']=='eval']))
reference=runs[0][1]
for _,c,s,_ in runs:
    for field in ('profile','seed','data_seed','dropout_seed','optimizer_updates','batch_size','eval_interval','eval_iters','dtype','hardware','compile','tf32','data_sha256','weight_decay','betas','grad_clip','warmup_iters'):
        assert c[field]==reference[field],f'Mismatched {field}'
    assert {k:v for k,v in c['model'].items() if k!='variant'}=={k:v for k,v in reference['model'].items() if k!='variant'}
    assert s['train_sample_sha256']==runs[0][2]['train_sample_sha256']
a.out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for folder,c,s,rows in runs:
    label=f"{c['variant']} (lr={c['learning_rate']:g})"
    for ax,split in zip(axes,('train','val')):
        ax.plot([r['step'] for r in rows],[r[split+'_loss'] for r in rows],label=label,linewidth=1.5)
for ax,title in zip(axes,('Training loss (evaluation mode)','Validation loss')):
    ax.set(xlabel='Completed optimizer updates',ylabel='Cross-entropy (nats)',title=title)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('SMOKE CHECK — not formal results' if reference['profile']=='smoke' else 'Shakespeare character-level language modeling')
fig.savefig(a.out/'loss_curves.png',dpi=180);fig.savefig(a.out/'loss_curves.pdf');plt.close(fig)
with (a.out/'comparison.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['run','variant','profile','learning_rate','min_lr','parameters','updates','best_val_loss','best_step','final_val_loss','delta_best_vs_first_run'])
    for folder,c,s,_ in runs:
        writer.writerow([str(folder),c['variant'],c['profile'],c['learning_rate'],c['min_lr'],c['parameters'],s['updates'],s['best_val_loss'],s['best_step'],s['final_val_loss'],s['best_val_loss']-runs[0][2]['best_val_loss']])
print(a.out/'loss_curves.png')
