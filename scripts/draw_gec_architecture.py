"""Draw the proposed Q3 architecture; SVG and PNG share node/edge definitions."""
from html import escape
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1040, 850
fig, ax = plt.subplots(figsize=(10.4, 8.5))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0,W), ylim=(H,0)); ax.axis('off')
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="850" viewBox="0 0 1040 850">',
       '<style>text {font-family:Arial,Helvetica,sans-serif;}</style>',
       '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#2563eb"/></marker></defs>',
       '<rect width="1040" height="850" fill="white"/>']
def text(x,y,s,size=15,bold=False,color='#111827'):
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="middle" font-size="{size}" font-weight="{600 if bold else 400}" fill="{color}">{escape(s)}</text>')
    ax.text(x,y,s,ha='center',va='center',fontsize=size*.72,weight='bold' if bold else 'normal',color=color)
def rect(x,y,w,h,fill='#ffffff',dash=False):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#d1d5db" stroke-width="1.5"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',facecolor=fill,edgecolor='#d1d5db',linewidth=1,linestyle='--' if dash else '-'))
def node(x,y,title,sub,fill='#eff6ff'):
    rect(x,y,320,70,fill)
    text(x+160,y+25,title,17,True)
    text(x+160,y+50,sub,14)
def arrow(points):
    d='M '+' L '.join(f'{x},{y}' for x,y in points)
    svg.append(f'<path d="{d}" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow)"/>')
    p=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=13,color='#2563eb',linewidth=1.3))
text(520,32,'Grammar correction: encoder–decoder Transformer',22,True)
rect(50,275,360,235,'#f8fafc',True)
rect(600,275,360,355,'#f8fafc',True)
text(145,292,'Encoder × 6',15,True)
text(695,292,'Decoder × 6',15,True)
node(70,70,'Source tokens','Erroneous sentence + EOS')
node(620,70,'Target prefix','BOS + preceding target tokens')
node(70,180,'Input representation','Token embedding + sinusoidal position')
node(620,180,'Input representation','Token embedding + sinusoidal position')
node(70,320,'Self-attention','Bidirectional; padding masked')
node(70,430,'Feed-forward','512 → 2048 → 512; ReLU')
node(70,550,'Encoder memory H','All source positions')
node(620,320,'Self-attention','Causal mask + padding mask')
node(620,430,'Cross-attention','Decoder queries; encoder keys / values')
node(620,550,'Feed-forward','512 → 2048 → 512; ReLU')
node(620,700,'Linear + softmax','Next-token distribution over vocabulary')
for x in (230,780):
    arrow([(x,140),(x,180)])
    arrow([(x,250),(x,320)])
    arrow([(x,390),(x,430)])
arrow([(230,500),(230,550)])
arrow([(780,500),(780,550)])
arrow([(780,620),(780,700)])
arrow([(390,585),(500,585),(500,465),(620,465)])
rect(514,431,90,24)
text(559,443,'Memory H',13)
text(230,675,'Each attention / FFN sublayer:',15,True)
text(230,700,'residual connection + dropout + LayerNorm',14)
text(780,799,'Generate until EOS, then detokenize.',15)
text(230,760,'Training: gold target prefix.',14)
text(230,785,'Inference: previously generated prefix.',14)
svg.append('</svg>')
(OUT/'gec_architecture.svg').write_text('\n'.join(svg)+'\n')
fig.savefig(OUT/'gec_architecture.png',dpi=200,facecolor='white')
fig.savefig(OUT/'gec_architecture.pdf',facecolor='white')
plt.close(fig)
# Validate XML even when rsvg-convert is unavailable.
import xml.etree.ElementTree as ET
ET.parse(OUT/'gec_architecture.svg')
print('Saved SVG, PNG, PDF to',OUT)
