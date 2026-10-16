"""Draw an editable, publication-style architecture preview from current scope.

Creates new SVG/PNG figure assets; does not alter any existing image or software.
"""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, PathPatch
from matplotlib.path import Path as MPath

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','axes.unicode_minus':False})
fig=plt.figure(figsize=(19.2,11.2),facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1920); ax.set_ylim(1120,0); ax.axis('off')
NAVY='#16324D'; INK='#20384C'; TEAL='#167D8D'; PALE='#EEF7F8'
AMBER='#AE7421'; GOLD='#FCF5E8'; GREY='#7F8C98'; LIGHT='#F6F8FA'; LINE='#C9D3DC'; RED='#BC4D50'

def text(x,y,s,size=14,color=INK,weight='normal',ha='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',zorder=7,
                   linespacing=1.35,**kw)

def box(x,y,w,h,fill='white',edge=NAVY,dash=False,lw=1.35,r=10,z=3):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=(0,(5,4)) if dash else '-',zorder=z)
    ax.add_patch(p); return p

def arrow(points,color=INK,dash=False,lw=1.6,head=12,z=2):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    p=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=head,color=color,linewidth=lw,
                      linestyle=(0,(5,4)) if dash else '-',zorder=z,joinstyle='round')
    ax.add_patch(p)

def label(x,y,s,size=11.3,color=INK):
    text(x,y,s,size,color,bbox={'facecolor':'white','edgecolor':'none','pad':2.2})

# Title and a clear typographic hierarchy.
text(60,58,'Unified BCI–VLA Interaction Framework',29,NAVY,'bold',ha='left')
text(62,101,'Joint task–channel inference and budgeted diagnostic interaction',15,GREY,ha='left')

# Planned feedback is drawn first and routed outside every main module.
arrow([(1830,746),(1888,746),(1888,126),(500,126),(500,148)],GREY,True,1.35,10)
label(1555,124,'Planned scene / outcome feedback',10.8,GREY)

# Context cards.
box(350,149,300,80,LIGHT,LINE,r=8)
text(500,177,'Scene state & task history',14,NAVY,'bold')
text(500,205,'Objects • relations • goal candidates',11.2,GREY)
box(710,149,300,80,PALE,LINE,r=8)
text(860,177,'Channel / probe models',14,NAVY,'bold')
text(860,205,'Source-specific observation models',11.2,GREY)
box(1060,149,310,80,GOLD,LINE,r=8)
text(1215,177,'Task outcomes & time budget',14,NAVY,'bold')
text(1215,205,'Consequences • elapsed time • budget',11.2,GREY)
text(1040,245,'Current channel, outcome and time models: simulation only',10.5,GREY)

arrow([(500,229),(500,261),(790,261),(790,340)],GREY,False,1.15,10)
arrow([(710,189),(682,189),(682,650),(650,650)],TEAL,False,1.1,10)
arrow([(1215,229),(1393,229),(1393,548),(1320,548)],AMBER,False,1.15,10)

# Panel headings.
text(180,286,'01  HUMAN–BCI',13,NAVY,'bold')
text(500,286,'02  EVIDENCE SEPARATION',13,NAVY,'bold')
text(1040,286,'03  TASK–CHANNEL DIAGNOSTIC CORE',13,NAVY,'bold')
text(1645,286,'04  PLANNED EXECUTION',13,GREY,'bold')
box(710,309,660,473,'white',LINE,lw=1.25,r=14,z=.5)
box(1430,309,430,503,LIGHT,GREY,True,lw=1.3,r=14,z=.5)

# Main evidence-to-decision flow.
arrow([(180,418),(180,455)],INK)
arrow([(290,497),(324,497),(324,400),(350,400)],INK)
arrow([(324,497),(324,530),(350,530)],INK)
ax.add_patch(Circle((324,497),3.5,color=INK,zorder=4))
arrow([(650,415),(668,415),(668,611),(635,611),(635,625)],TEAL,False,1.45)
arrow([(560,580),(560,625)],AMBER,False,1.45)
arrow([(650,682),(695,682),(695,395),(760,395)],TEAL,False,1.7)
arrow([(1040,455),(1040,494)],TEAL,False,1.8)
arrow([(1140,650),(1140,690)],TEAL,False,1.6)
arrow([(855,650),(855,690)],AMBER,False,1.5)
arrow([(1320,721),(1408,721),(1408,383),(1460,383)],GREY,True,1.6)

# Active diagnostic feedback. The evidence inlet and query outlet are separate.
arrow([(760,594),(722,594),(722,802),(180,802),(180,735)],TEAL,False,1.8,13)
label(440,801,'Next task question / known-target probe',12,TEAL)
arrow([(70,695),(44,695),(44,365),(137,365)],TEAL,False,1.4)
text(26,550,'Prompted interaction',10.8,TEAL,rotation=90)

# Human EEG line illustration, no photographs or stock artwork.
ax.add_patch(Circle((180,367),42,facecolor='#F4F7FA',edgecolor=NAVY,linewidth=1.5,zorder=4))
ax.plot([167,167,146],[405,419,430],color=NAVY,lw=1.4,zorder=4)
ax.plot([193,193,214],[405,419,430],color=NAVY,lw=1.4,zorder=4)
ax.plot([142,151,171,193,212,220],[355,337,329,331,340,359],color=TEAL,lw=2.0,zorder=5)
for x,y in [(151,343),(172,333),(194,335),(213,345)]:
    ax.add_patch(Circle((x,y),3.0,facecolor='white',edgecolor=TEAL,linewidth=1.2,zorder=5))
xs=[227+i*1.3 for i in range(36)]
ys=[365+8*math.sin(i*.75)*(1 if 10<i<26 else .45) for i in range(36)]
ax.plot(xs,ys,color=TEAL,lw=1.1,zorder=5)
text(239,430,'User',11,GREY)
box(70,455,220,85,'white',NAVY)
text(180,480,'BCI adapter',17,NAVY,'bold')
text(180,514,'Validated events\nOffline replay',11.7,GREY)
box(70,660,220,75,PALE,TEAL)
text(180,686,'Interaction display',15,TEAL,'bold')
text(180,713,'Task / probe prompts',11.2,GREY)

# Sources stay separate; the known marker is never a robot goal.
box(350,350,300,100,PALE,TEAL)
text(500,375,'Task response',16,TEAL,'bold')
text(500,405,'Unknown goal g',13,INK)
text(500,432,r'$P(y\mid g,z)$',14,TEAL)
box(350,480,300,100,GOLD,AMBER)
text(500,505,'Channel probe',16,AMBER,'bold')
text(500,535,'Known prompt a',13,INK)
text(500,562,r'$P(y\mid a,z)$',14,AMBER)
box(350,625,300,110,'white',NAVY)
text(500,651,'Source-specific likelihoods',14.3,NAVY,'bold')
text(500,680,'TASK / PROBE / CONFIRM',11.7,GREY)
text(500,711,'Noisy confirmation is evidence',11.5,INK)

# Correlated joint belief, not two independent classifiers.
box(760,340,560,115,PALE,TEAL,lw=1.6)
text(988,369,'Joint belief',19,TEAL,'bold')
text(988,408,r'$b_t(g,z)$',23,NAVY)
text(988,439,'Goal g × channel mode z',12,GREY)
for row in range(2):
    for col in range(2):
        ax.add_patch(Rectangle((1205+col*33,362+row*33),29,29,
            facecolor=TEAL if row==col else '#D5E9EC',edgecolor='white',linewidth=.8,zorder=5))
text(1237,439,'correlated',10.4,TEAL)

box(760,494,560,156,PALE,TEAL,lw=1.6)
text(1040,523,'Task-aware diagnostic policy',18,TEAL,'bold')
text(1040,553,'One-step task VoI / bounded lookahead',12.5,NAVY)
text(1040,577,'Expected task loss + interaction time',12,GREY)
for x,w,s in [(780,155,'Ask task'),(945,155,'Probe channel'),(1110,190,'Retry / extend')]:
    box(x,598,w,29,'white','#C7DEE2',r=5,lw=.8)
    text(x+w/2,613,s,11.5,TEAL)
text(1040,638,'Request confirmation    •    Pause / exit',11.6,TEAL)

box(760,690,190,62,GOLD,AMBER,r=8)
text(855,709,'Unknown / expired /',10.8,AMBER)
text(855,735,'no budget → Pause',11.4,AMBER,'bold')
box(970,690,350,62,'white',NAVY,r=8)
text(1145,713,'Proposed task contract',15,NAVY,'bold')
text(1145,738,'Goal • scene version • deadline',11,GREY)
text(1040,769,'Planning output is not execution authorization',11.1,GREY)

# All downstream integration remains planned and explicitly dashed.
box(1460,342,370,82,'white',GREY,True,r=9)
text(1645,368,'Evidence-bound authorization',15.7,NAVY,'bold')
text(1645,398,'Noisy confirmation + version checks',11.8,GREY)
arrow([(1645,424),(1645,455)],GREY,True,1.4)
box(1460,455,370,80,'white',GREY,True,r=9)
text(1645,481,'VLA adapter',18,NAVY,'bold')
text(1645,512,'e.g., π0.5',12.5,GREY)
arrow([(1645,535),(1645,575)],GREY,True,1.4)
box(1460,575,370,85,'white',GREY,True,r=9)
text(1596,617,'Robot execution',16,NAVY,'bold')
arm=[(1750,638),(1750,625),(1777,605),(1763,586),(1793,584)]
ax.plot(*zip(*arm),color=GREY,lw=4.5,solid_capstyle='round',zorder=5)
for x,y in arm[1:-1]:ax.add_patch(Circle((x,y),5,facecolor='white',edgecolor=GREY,lw=1.5,zorder=6))
ax.plot([1737,1763],[642,642],color=GREY,lw=2,zorder=5)
ax.plot([1793,1801,1800],[584,590,599],color=GREY,lw=1.6,zorder=5)
ax.plot([1793,1798],[584,578],color=GREY,lw=1.6,zorder=5)
arrow([(1645,660),(1645,705)],GREY,True,1.4)
box(1460,705,370,80,'white',GREY,True,r=9)
text(1645,734,'Execution monitor',16.5,NAVY,'bold')
text(1645,763,'Outcome checks / controller feedback',11.5,GREY)

# STOP path bypasses belief and utility scoring. No physical stop is claimed.
box(80,839,200,43,'#FCF2F1',RED,r=7)
text(180,860,'STOP request',13,RED,'bold')
ax.plot([280,1408],[860,860],color=RED,lw=1.7,zorder=2)
arrow([(1408,860),(1408,765),(1460,765)],RED,True,1.6,12)
label(840,859,'Independent of diagnostic scoring',11.8,RED)

# The evaluation rail is detached from all online arrows.
text(60,916,'OFFLINE COMPARISON & AUDIT',12,NAVY,'bold',ha='left')
box(60,939,1800,87,LIGHT,LINE,r=10,lw=1.1)
for x in [554,947,1392]:ax.plot([x,x],[956,1010],color=LINE,lw=.8,zorder=4)
text(305,966,'Baseline registry: B0–B10 / M',14,NAVY,'bold')
text(305,997,'B0: external trace pending',11.2,GREY)
text(750,981,'8 mechanism ablations',14,NAVY,'bold')
text(1170,967,'O1: same-information oracle',14,NAVY,'bold')
text(1170,997,'Exact finite-tree reference',11.2,GREY)
text(1625,967,'O2: omniscient reference',14,NAVY,'bold')
text(1625,997,'Evaluation only • not deployable',11.2,GREY)

# Scope legend, kept concise rather than claiming completed live integration.
box(64,1063,41,22,'white',TEAL,r=3,lw=1.4)
text(118,1074,'Implemented offline',11.9,INK,ha='left')
box(410,1063,41,22,'white',GREY,True,r=3,lw=1.3)
text(465,1074,'Planned integration',11.9,INK,ha='left')
text(910,1074,'Finite simulation models; live calibration and robot closed loop are pending.',12,GREY,ha='left')

fig.savefig(OUT/'bci_vla_framework_v1.svg',format='svg',facecolor='white')
fig.savefig(OUT/'bci_vla_framework_v1.png',dpi=220,facecolor='white')
plt.close(fig)
print(OUT/'bci_vla_framework_v1.png')
print(OUT/'bci_vla_framework_v1.svg')
