"""
Homeplate Umpire Scorecard Generator
Run: streamlit run ump_scorecard_app.py
"""
import io, os, csv, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle, Polygon
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from matplotlib.image import imread
from collections import defaultdict
from datetime import datetime
import streamlit as st

# ── Optional local assets ─────────────────────────────────────────────────────
_BASE      = os.path.dirname(os.path.abspath(__file__))
_FONT_PATH = os.path.join(_BASE, "assets", "Sooner-Born-Regular.otf")
_MASK_PATH = os.path.join(_BASE, "assets", "Mask SB.png")
_TM_PATH   = os.path.join(_BASE, "assets", "TM.jpg")
SOONER_BORN = FontProperties(fname=_FONT_PATH) if os.path.exists(_FONT_PATH) else None
MASK_IMG    = imread(_MASK_PATH) if os.path.exists(_MASK_PATH) else None
TM_IMG      = imread(_TM_PATH)   if os.path.exists(_TM_PATH)   else None

# ── Known team color map (auto-color by abbreviation) ─────────────────────────
TEAM_COLORS = {
    "OKL": "#841617", "OU":  "#841617",
    "KAN": "#0051A5", "KU":  "#0051A5",
    "BIN": "#005A9C",
    "MIC": "#00274C",
    "TEX": "#BF5700",
    "LSU": "#461D7C",
    "FLA": "#0021A5",
    "ARK": "#9D2235",
    "ALA": "#9E1B32",
    "TEN": "#FF8200",
    "MSU": "#5D1725",
    "GEO": "#BA0C2F",
    "ORE": "#154733",
    "WAS": "#4B2E83",
    "ASU": "#8C1D40",
    "UCLA": "#2D68C4",
    "USC": "#990000",
    "CAL": "#003262",
    "OSU": "#BB0000",
}

def get_abb(team_id: str) -> str:
    """Extract 2-4 char abbreviation from Trackman team ID like OKL_SOO_SB."""
    skip = {"UNI", "COL", "STA", "COM", "ST"}
    parts = team_id.replace("-", "_").split("_")
    for p in parts:
        if p.upper() not in skip and p.upper() != "SB" and len(p) >= 2:
            return p.upper()
    return parts[0].upper()

def guess_color(abb: str, fallback: str) -> str:
    return TEAM_COLORS.get(abb.upper(), fallback)

def format_date(date_str: str) -> str:
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%A, %m/%d/%Y"), dt.strftime("%m/%d/%Y")
        except Exception:
            pass
    return date_str, date_str

# ── Zone constants ────────────────────────────────────────────────────────────
# A pitch is a strike if any part of the softball touches the zone.
# Softball radius = 1.91 in = 0.159 ft → expand zone by 0.159 ft on all sides.
_XL = 17/24; _YLO = 1.5; _YHI = 1.5 + 22/12; _R = 0.159
def _in_zone(r):
    try: return abs(float(r["PlateLocSide"])) <= _XL+_R and _YLO-_R <= float(r["PlateLocHeight"]) <= _YHI+_R
    except: return False

def _s(r): return float(r["PlateLocSide"])
def _h(r): return float(r["PlateLocHeight"])

# ── Run value (RE shift from wrong call) ──────────────────────────────────────
# Count run values: expected runs added for the OFFENSE at each count (linear weights)
_CRV = {
    (0,0):0.000,
    (1,0):0.031, (0,1):-0.034,
    (2,0):0.084, (1,1):-0.003, (0,2):-0.084,
    (3,0):0.167, (2,1): 0.052, (1,2):-0.055,
                 (3,1): 0.109, (2,2):-0.018,
                               (3,2): 0.018,
}
_RV_WALK = 0.30; _RV_K = -0.27

def pitch_rv(b, s, call):
    """Absolute RE shift caused by a wrong call (higher = more impactful)."""
    b, s = int(b), int(s)
    if call == "StrikeCalled":          # phantom: was ball, called strike
        rv_correct = _RV_WALK if b == 3 else _CRV.get((b+1, s), 0.0)
        rv_actual  = _RV_K    if s == 2 else _CRV.get((b,   s+1), 0.0)
    else:                               # missed: was strike, called ball
        rv_correct = _RV_K    if s == 2 else _CRV.get((b,   s+1), 0.0)
        rv_actual  = _RV_WALK if b == 3 else _CRV.get((b+1, s), 0.0)
    return abs(rv_correct - rv_actual)

# ── Data processing ───────────────────────────────────────────────────────────
def load_csv(file_obj):
    text = file_obj.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)

def compute_stats(subset):
    phantom = [r for r in subset if r["PitchCall"] == "StrikeCalled" and not _in_zone(r)]
    missed  = [r for r in subset if r["PitchCall"] == "BallCalled"   and     _in_zone(r)]
    total = len(subset); correct = total - len(phantom) - len(missed)
    balls   = [r for r in subset if r["PitchCall"] == "BallCalled"]
    strikes = [r for r in subset if r["PitchCall"] == "StrikeCalled"]
    return dict(
        total=total, correct=correct,
        acc=correct/total if total else 0,
        ball_total=len(balls),   ball_correct=len(balls)-len(missed),
        strike_total=len(strikes), strike_correct=len(strikes)-len(phantom),
        ball_acc=(len(balls)-len(missed))/len(balls)        if balls   else 0,
        strike_acc=(len(strikes)-len(phantom))/len(strikes) if strikes else 0,
        phantom=len(phantom), missed=len(missed),
        phantom_rows=phantom, missed_rows=missed,
        phantom_pts=[(_s(r),_h(r),r.get('PitcherTeam','')) for r in phantom],
        missed_pts =[(_s(r),_h(r),r.get('PitcherTeam','')) for r in missed],
        strike_pts =[(_s(r),_h(r)) for r in strikes],
    )

def compute_consistency(called):
    grid = defaultdict(list)
    for r in called:
        try:
            xi = min(int((_s(r)-(-1.7))/(3.4/5)), 4)
            yi = min(int((_h(r)-0.5)/(4.0/5)), 4)
            grid[(xi,yi)].append(r["PitchCall"])
        except: pass
    agree = tot = 0
    for calls in grid.values():
        for i in range(len(calls)):
            for j in range(i+1, len(calls)):
                tot += 1
                if calls[i] == calls[j]: agree += 1
    return agree/tot if tot else 0

INN = {"1":"1st","2":"2nd","3":"3rd","4":"4th","5":"5th","6":"6th","7":"7th"}
def count_lev(r):
    b,s = int(r["Balls"]), int(r["Strikes"])
    if b==3 and s==2: return 5
    if b==3 and s==1: return 4
    if b==3 and s==0: return 3
    if b==2 and s==0: return 2
    if b==0 and s==2: return 2
    return 1

def get_impactful(st_data, n=3):
    wrong = sorted(st_data["phantom_rows"]+st_data["missed_rows"], key=lambda r: -count_lev(r))
    out = []
    for r in wrong[:n]:
        half = r["Top/Bottom"]; inn = INN.get(r["Inning"], r["Inning"]+"th")
        def fmt(nm):
            p = nm.split(",")
            return f"{p[1].strip()} {p[0].strip()}" if len(p) > 1 else nm
        pit = fmt(r["Pitcher"]); bat = fmt(r["Batter"])
        b,s,o = r["Balls"], r["Strikes"], r["Outs"]
        desc = "ball is called a strike" if r["PitchCall"]=="StrikeCalled" else "strike is called a ball"
        hl = "Top of the" if half=="Top" else "Bottom of the"
        sit = f"{o} out{'s' if o!='1' else ''}, {b}-{s} count"
        rv = pitch_rv(b, s, r["PitchCall"])
        out.append((f"{hl} {inn}", f"{pit} to {bat}", sit, desc, _s(r), _h(r), rv, r["PitchCall"]))
    return out

def _imp_color(call):
    return ZONE_BORDER if call == "StrikeCalled" else "#2E75B6"

def compute_favor(called, phantom_rows, missed_rows, home_team):
    okl_phantom = sum(1 for r in phantom_rows if r["PitcherTeam"] == home_team)
    away_phantom = sum(1 for r in phantom_rows if r["PitcherTeam"] != home_team)
    okl_missed  = sum(1 for r in missed_rows  if r["PitcherTeam"] == home_team)
    away_missed  = sum(1 for r in missed_rows  if r["PitcherTeam"] != home_team)
    home_net = okl_phantom + away_missed
    away_net = away_phantom + okl_missed
    return home_net, away_net

def get_score(rows, home_team):
    home = sum(int(r["RunsScored"]) for r in rows if r.get("RunsScored","").strip() and r["BatterTeam"] == home_team)
    away = sum(int(r["RunsScored"]) for r in rows if r.get("RunsScored","").strip() and r["BatterTeam"] != home_team)
    return home, away

def process_game(rows):
    called = [r for r in rows
              if r.get("PitchCall") in ("BallCalled","StrikeCalled")
              and r.get("PlateLocSide","").strip() and r.get("PlateLocHeight","").strip()]
    home_team = rows[0]["HomeTeam"] if rows else ""
    away_team = rows[0]["AwayTeam"] if rows else ""
    home_abb = get_abb(home_team); away_abb = get_abb(away_team)
    date_raw = rows[0].get("Date","") if rows else ""
    date_long, date_short = format_date(date_raw)
    home_score, away_score = get_score(rows, home_team)
    gs = compute_stats(called)
    consistency = compute_consistency(called)
    home_net, away_net = compute_favor(called, gs["phantom_rows"], gs["missed_rows"], home_team)
    # pitchers
    pg = defaultdict(list)
    for r in called: pg[(r["Pitcher"], r["PitcherTeam"])].append(r)
    pitchers = []
    for (pname, pteam), prows in sorted(pg.items()):
        pst = compute_stats(prows)
        pimp = get_impactful(pst, min(3, len(pst["phantom_rows"])+len(pst["missed_rows"])))
        parts = pname.split(",")
        dname = f"{parts[1].strip()} {parts[0].strip()}" if len(parts)>1 else pname
        pabb = get_abb(pteam)
        pitchers.append(dict(name=dname, last=parts[0].strip() if parts else pname,
                             team=pteam, abb=pabb, rows=prows, stats=pst, impactful=pimp))
    # catchers
    cg = defaultdict(list)
    for r in called:
        cname = r.get("Catcher","").strip()
        cteam = r.get("CatcherTeam","").strip()
        if cname: cg[(cname, cteam)].append(r)
    catchers = []
    for (cname, cteam), crows in sorted(cg.items()):
        cst = compute_stats(crows)
        cimp = get_impactful(cst, min(3, len(cst["phantom_rows"])+len(cst["missed_rows"])))
        parts = cname.split(",")
        dname = f"{parts[1].strip()} {parts[0].strip()}" if len(parts)>1 else cname
        cabb = get_abb(cteam) if cteam else ""
        catchers.append(dict(name=dname, last=parts[0].strip() if parts else cname,
                             team=cteam, abb=cabb, rows=crows, stats=cst, impactful=cimp))
    imp_global = get_impactful(gs, 3)
    return dict(
        rows=rows, called=called,
        home_team=home_team, away_team=away_team,
        home_abb=home_abb, away_abb=away_abb,
        date_long=date_long, date_short=date_short,
        home_score=home_score, away_score=away_score,
        gs=gs, consistency=consistency,
        home_net=home_net, away_net=away_net,
        impactful=[(t[0],t[1],t[2],t[3]) for t in imp_global],
        impactful_pts=[(t[4],t[5]) for t in imp_global],
        impactful_rvs=[t[6] for t in imp_global],
        impactful_calls=[t[7] for t in imp_global],
        pitchers=pitchers,
        catchers=catchers,
    )

# ── Palette ───────────────────────────────────────────────────────────────────
BLACK="#111111"; GREY="#555555"; LGREY="#CCCCCC"
ZONE_PINK="#F4CCCC"; ZONE_BORDER="#CC3333"; BG="white"
IMP_COLORS=["#E67E22","#F1C40F","#9B59B6"]

# ── Margin helpers ────────────────────────────────────────────────────────────
ML=0.25/8.5; MR=0.25/8.5; MB=0.25/11; MT=0.25/11
CW=1-ML-MR; CH=1-MB-MT
def m(l,b,w,h): return [ML+l*CW, MB+b*CH, w*CW, h*CH]

# ── Draw primitives ───────────────────────────────────────────────────────────
def title_text(ax, txt):
    if SOONER_BORN:
        ax.text(0.5,0.95,txt,ha='center',va='top',fontsize=30,color=BLACK,fontproperties=SOONER_BORN)
    else:
        ax.text(0.5,0.95,txt,ha='center',va='top',fontsize=26,fontweight='bold',color=BLACK,fontfamily='Arial')

def draw_donut(ax, pct, color, size=0.32):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(Wedge((0.5,0.5),size,0,360,width=size*0.32,facecolor="#E0E0E0",edgecolor="none"))
    ax.add_patch(Wedge((0.5,0.5),size,90-pct*360,90,width=size*0.32,facecolor=color,edgecolor="none"))
    ax.text(0.5,0.54,f"{round(pct*100)}%",ha='center',va='center',fontsize=14,fontweight='bold',color=BLACK,fontfamily='Arial')

def draw_acc_bar(ax, pct, color, label, avg_pct=None):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    bh=0.28; by=0.35
    ax.add_patch(Rectangle((0,by),1,bh,facecolor="#E8E8E8",edgecolor="none"))
    ax.add_patch(Rectangle((0,by),pct,bh,facecolor=color,edgecolor="none"))
    if avg_pct:
        ax.plot([avg_pct,avg_pct],[by-0.06,by+bh+0.06],color=BLACK,lw=1.5,ls='--')
        ax.text(avg_pct,by+bh+0.10,"avg.",ha='center',va='bottom',fontsize=8,color=BLACK,fontfamily='Arial')
    ax.text(1.02,by+bh/2,f"{round(pct*100)}%",ha='left',va='center',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax.text(-0.01,by+bh+0.14,label,ha='left',va='bottom',fontsize=11,fontweight='bold',color=BLACK,fontfamily='Arial')

def draw_zone(ax, subset, p_pts, m_pts, s_pts, imp_pts, imp_colors, team_edge_fn=None, show_labels=True):
    ax.set_facecolor('white'); ax.set_xlim(-2.0,2.6); ax.set_ylim(-0.15,4.8)
    ax.set_aspect('equal'); ax.axis('off')
    if len(s_pts) >= 4:
        xs=np.array([p[0] for p in s_pts]); ys=np.array([p[1] for p in s_pts])
        kde=gaussian_kde(np.vstack([xs,ys]),bw_method=0.55)
        gx=np.linspace(-1.7,1.7,200); gy=np.linspace(0.5,4.5,200)
        GX,GY=np.meshgrid(gx,gy)
        Z=kde(np.vstack([GX.ravel(),GY.ravel()])).reshape(GX.shape)
        ax.contourf(GX,GY,Z,levels=[Z.max()*0.08,Z.max()],colors=[ZONE_PINK],alpha=0.30,zorder=1)
        ax.contour(GX,GY,Z,levels=[Z.max()*0.08],colors=[ZONE_BORDER],linewidths=1.5,linestyles='dotted',zorder=2)
    ax.add_patch(Rectangle((-_XL,_YLO),2*_XL,_YHI-_YLO,linewidth=1.0,edgecolor=BLACK,facecolor='none',zorder=3))
    _R=0.159
    ax.add_patch(Rectangle((-_XL-_R,_YLO-_R),2*_XL+2*_R,_YHI-_YLO+2*_R,linewidth=1.0,edgecolor='orange',alpha=0.65,linestyle='dashed',facecolor='none',zorder=3))
    for xv in (-_XL/3, _XL/3):
        ax.plot([xv,xv],[_YLO,_YHI],color='grey',alpha=0.25,linewidth=0.6,zorder=3)
    for yh in (_YLO+(_YHI-_YLO)/3, _YLO+2*(_YHI-_YLO)/3):
        ax.plot([-_XL,_XL],[yh,yh],color='grey',alpha=0.25,linewidth=0.6,zorder=3)
    # Home plate  (_XL = 17/24 ft = 8.5/12 ft, so all plate dims derive from _XL)
    plate_verts = [(0,0),(_XL,_XL),(_XL,2*_XL),(-_XL,2*_XL),(-_XL,_XL)]
    ax.add_patch(Polygon(plate_verts,closed=True,facecolor='#EFEFEF',edgecolor='#888888',linewidth=0.9,zorder=2))
    # Zone dimension labels
    ax.text(0, _YLO-0.08, '17 in', ha='center', va='top',
            fontsize=6.5, color='black', alpha=0.40, fontfamily='Arial', zorder=4)
    ax.text(_XL+0.09, (_YLO+_YHI)/2, '22 in', ha='left', va='center',
            fontsize=6.5, color='black', alpha=0.40, fontfamily='Arial', rotation=90, zorder=4)
    imp_set={(round(x,3),round(y,3)) for x,y in imp_pts}
    miss_set={(round(x,3),round(y,3)) for x,y,_ in p_pts+m_pts}
    for r in subset:
        try: sx,sy=_s(r),_h(r)
        except: continue
        key=(round(sx,3),round(sy,3))
        if key in miss_set or key in imp_set: continue
        if r["PitchCall"]=="StrikeCalled":
            ax.plot(sx,sy,'o',ms=12,color=ZONE_BORDER,alpha=0.25,markeredgecolor='none',zorder=2)
        elif r["PitchCall"]=="BallCalled":
            ax.plot(sx,sy,'o',ms=12,color='#2E75B6',alpha=0.25,markeredgecolor='none',zorder=2)
    for sx,sy,team in p_pts:
        if (round(sx,3),round(sy,3)) not in imp_set:
            ec = team_edge_fn(team) if team_edge_fn else 'none'
            lbl = get_abb(str(team))[:2] if team else ''
            ax.plot(sx,sy,'o',ms=12,color=ZONE_BORDER,markeredgecolor=ec,markeredgewidth=0.8,zorder=4)
            if show_labels: ax.text(sx,sy,lbl,ha='center',va='center',fontsize=4.5,color='white',fontweight='bold',zorder=5)
    for sx,sy,team in m_pts:
        if (round(sx,3),round(sy,3)) not in imp_set:
            ec = team_edge_fn(team) if team_edge_fn else 'none'
            lbl = get_abb(str(team))[:2] if team else ''
            ax.plot(sx,sy,'o',ms=12,color='#2E75B6',markeredgecolor=ec,markeredgewidth=0.8,zorder=4)
            if show_labels: ax.text(sx,sy,lbl,ha='center',va='center',fontsize=4.5,color='white',fontweight='bold',zorder=5)
    for i,((sx,sy),color) in enumerate(zip(imp_pts,imp_colors),1):
        ax.plot(sx,sy,'o',ms=12,color=color,markeredgecolor='white',markeredgewidth=0.8,zorder=5)
        tc='black' if color=='#F1C40F' else 'white'
        ax.text(sx,sy,str(i),ha='center',va='center',fontsize=7,color=tc,fontweight='bold',zorder=6)

def draw_zone_legend(ax, team_labels=('AB','CD'), team_colors=('gray','gray'), simple=False):
    ax.set_facecolor(BG); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    # Left column: correct calls + EUZ
    ax.plot(0.03,0.82,'o',ms=7,color=ZONE_BORDER,alpha=0.5,markeredgecolor='none')
    ax.text(0.09,0.82,"Correct called strike",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')
    ax.plot(0.03,0.55,'o',ms=7,color='#2E75B6',alpha=0.5,markeredgecolor='none')
    ax.text(0.09,0.55,"Correct called ball",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')
    ax.add_patch(Rectangle((0.005,0.08),0.055,0.18,facecolor=ZONE_PINK,edgecolor=ZONE_BORDER,linewidth=1.2))
    ax.text(0.09,0.17,"Estimated Ump Zone (EUZ)",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')
    if simple:
        # Right column: single dot per call type, no team labels
        ax.plot(0.53,0.72,'o',ms=9,color=ZONE_BORDER,markeredgecolor='none')
        ax.text(0.59,0.72,"Ball called Strike",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')
        ax.plot(0.53,0.28,'o',ms=9,color='#2E75B6',markeredgecolor='none')
        ax.text(0.59,0.28,"Strike called Ball",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')
    else:
        # Right column: missed calls per team
        for y,lbl,ec in zip([0.82,0.60], team_labels, team_colors):
            ax.plot(0.53,y,'o',ms=9,color=ZONE_BORDER,markeredgecolor=ec,markeredgewidth=1.2)
            ax.text(0.53,y,lbl[:2],ha='center',va='center',fontsize=3.5,color='white',fontweight='bold')
            ax.text(0.59,y,f"{lbl} Ball called Strike",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')
        for y,lbl,ec in zip([0.38,0.16], team_labels, team_colors):
            ax.plot(0.53,y,'o',ms=9,color='#2E75B6',markeredgecolor=ec,markeredgewidth=1.2)
            ax.text(0.53,y,lbl[:2],ha='center',va='center',fontsize=3.5,color='white',fontweight='bold')
            ax.text(0.59,y,f"{lbl} Strike called Ball",ha='left',va='center',fontsize=7,color=GREY,fontfamily='Arial')

def draw_tm_logo(fig, l, b, w, h):
    if TM_IMG is not None:
        ax=fig.add_axes(m(l,b,w,h)); ax.imshow(TM_IMG); ax.axis('off')

def draw_powered_by(ax):
    ax.text(0.477,0.18,"Powered by",ha='right',va='center',fontsize=8,fontweight='bold',color=GREY,fontfamily='Arial')

# ── Footer helper ─────────────────────────────────────────────────────────────
def add_footer(fig, game):
    ax=fig.add_axes(m(0.0,0.005,1.0,0.025)); ax.set_facecolor(BG); ax.axis('off')
    ax.text(0.5,0.5,
            f"Accuracy measured on Trackman softball zone (17 in wide, 22 in tall)  |  "
            f"Solid box = Trackman zone  |  Source: Trackman — {game['home_abb']} vs {game['away_abb']}, {game['date_short']}",
            ha='center',va='center',fontsize=5.5,color=LGREY,fontfamily='Arial')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Full Game Scorecard
# ══════════════════════════════════════════════════════════════════════════════
def build_main_page(pdf, game, ump_name, home_color, away_color):
    fig=plt.figure(figsize=(8.5,11),facecolor=BG,dpi=150)
    fig.subplots_adjust(0,0,1,1)
    gs=game["gs"]; T=0.97; TITLE_H=0.092
    imp_colors=IMP_COLORS[:len(game["impactful"])]

    ax_name=fig.add_axes(m(0.0,T-TITLE_H,1.0,TITLE_H)); ax_name.set_facecolor(BG); ax_name.axis('off')
    title_text(ax_name,"Homeplate Umpire Scorecard"); draw_powered_by(ax_name)
    draw_tm_logo(fig,0.490,T-TITLE_H+0.004,0.065,TITLE_H*0.42)

    sec2_b=T-TITLE_H-0.13
    ax_teams=fig.add_axes(m(0.01,sec2_b,0.38,0.13)); ax_teams.set_facecolor(BG); ax_teams.axis('off')
    ax_teams.text(0.0,0.85,game["home_abb"],ha='left',va='top',fontsize=22,fontweight='bold',color=home_color,fontfamily='Arial')
    ax_teams.text(0.38,0.85," vs. ",ha='center',va='top',fontsize=18,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_teams.text(0.70,0.85,game["away_abb"],ha='left',va='top',fontsize=22,fontweight='bold',color=away_color,fontfamily='Arial')
    ax_teams.text(0.0,0.30,str(game["home_score"]),ha='left',va='center',fontsize=26,fontweight='bold',color=home_color,fontfamily='Arial')
    ax_teams.text(0.38,0.30," - ",ha='center',va='center',fontsize=22,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_teams.text(0.70,0.30,str(game["away_score"]),ha='left',va='center',fontsize=26,fontweight='bold',color=away_color,fontfamily='Arial')

    if MASK_IMG is not None:
        ax_mask=fig.add_axes(m(0.38,sec2_b-0.01,0.24,0.14)); ax_mask.imshow(MASK_IMG); ax_mask.axis('off')

    ax_date=fig.add_axes(m(0.50,sec2_b,0.50,0.13)); ax_date.set_facecolor(BG); ax_date.axis('off')
    ax_date.text(1.0,0.62,ump_name,ha='right',va='center',fontsize=17,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_date.text(1.0,0.35,game["date_long"],ha='right',va='center',fontsize=14,color=GREY,fontfamily='Arial')

    soc_b=sec2_b-0.004
    ax_div=fig.add_axes(m(0.04,soc_b-0.004,0.92,0.003)); ax_div.set_facecolor(LGREY); ax_div.axis('off')

    # Favor text
    home_net=game["home_net"]; away_net=game["away_net"]
    if home_net > away_net:
        fav_txt=f"+{home_net-away_net} calls"; fav_label=f"for {game['home_abb']}"; fav_color=home_color
    elif away_net > home_net:
        fav_txt=f"+{away_net-home_net} calls"; fav_label=f"for {game['away_abb']}"; fav_color=away_color
    else:
        fav_txt="Even"; fav_label="no net favor"; fav_color=BLACK

    # 4 Metrics
    met_b=soc_b-0.004-0.175; met_h=0.170; met_w=0.22; offsets=[0.01,0.26,0.51,0.76]
    ax_m1=fig.add_axes(m(offsets[0],met_b,met_w,met_h)); ax_m1.set_facecolor(BG); ax_m1.axis('off')
    ax_m1.text(0.5,0.97,"Overall",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m1.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub1=fig.add_axes(m(offsets[0]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub1,gs["acc"],"#2E75B6")
    ax_m1.text(0.5,0.04,f"Called {gs['correct']} of {gs['total']} taken\npitches correctly",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m2=fig.add_axes(m(offsets[1],met_b,met_w,met_h)); ax_m2.set_facecolor(BG); ax_m2.axis('off')
    ax_m2.text(0.5,0.97,"Total Missed",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m2.text(0.5,0.87,"Calls",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m2.text(0.5,0.56,f"{gs['phantom']+gs['missed']}",ha='center',va='center',fontsize=44,fontweight='bold',color="#CC3333",fontfamily='Arial')
    ax_m2.text(0.5,0.38,"incorrect calls",ha='center',va='center',fontsize=9,color=GREY,fontfamily='Arial')
    ax_m2.text(0.5,0.12,f"{gs['phantom']} phantom strikes\n{gs['missed']} missed strikes",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m3=fig.add_axes(m(offsets[2],met_b,met_w,met_h)); ax_m3.set_facecolor(BG); ax_m3.axis('off')
    ax_m3.text(0.5,0.97,"Overall",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m3.text(0.5,0.87,"Favor",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m3.text(0.5,0.58,fav_txt,ha='center',va='center',fontsize=22,fontweight='bold',color=fav_color,fontfamily='Arial')
    ax_m3.text(0.5,0.41,fav_label,ha='center',va='center',fontsize=13,fontweight='bold',color=fav_color,fontfamily='Arial')
    ax_m3.text(0.5,0.12,f"{gs['phantom']} phantom strikes\n{gs['missed']} missed strikes",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m4=fig.add_axes(m(offsets[3],met_b,met_w,met_h)); ax_m4.set_facecolor(BG); ax_m4.axis('off')
    ax_m4.text(0.5,0.97,"Overall",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m4.text(0.5,0.87,"Consistency",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub4=fig.add_axes(m(offsets[3]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub4,game["consistency"],"#2E75B6")
    ax_m4.text(0.5,0.04,"Similar-location pitches\ncalled consistently",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    div2_b=met_b-0.010
    ax_div2=fig.add_axes(m(0.04,div2_b,0.92,0.003)); ax_div2.set_facecolor(LGREY); ax_div2.axis('off')

    mid_b=div2_b-0.005-0.315; mid_h=0.315; LEG_H=0.064
    ax_zh=fig.add_axes(m(0.00,mid_b+mid_h-0.025,0.48,0.035)); ax_zh.set_facecolor(BG); ax_zh.axis('off')
    ax_zh.text(0.0,0.9,"All Missed Calls",ha='left',va='top',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_zh.text(0.0,0.1,"true zone and Estimated Ump Zone (EUZ)",ha='left',va='bottom',fontsize=8,color=GREY,fontfamily='Arial')
    ax_zone=fig.add_axes(m(0.00,mid_b+LEG_H,0.48,mid_h-0.010-LEG_H))
    def _edge_fn(team): return home_color if get_abb(str(team))==game["home_abb"] else away_color
    draw_zone(ax_zone,game["called"],gs["phantom_pts"],gs["missed_pts"],gs["strike_pts"],game["impactful_pts"],imp_colors,team_edge_fn=_edge_fn)
    _tlbls=(game["home_abb"], game["away_abb"])
    _tcolors=(home_color, away_color)
    ax_zleg=fig.add_axes(m(0.00,mid_b,0.48,LEG_H)); draw_zone_legend(ax_zleg,team_labels=_tlbls,team_colors=_tcolors)

    ax_ih=fig.add_axes(m(0.50,mid_b+mid_h-0.025,0.48,0.035)); ax_ih.set_facecolor(BG); ax_ih.axis('off')
    ax_ih.text(0.0,0.9,"Impactful Missed Calls",ha='left',va='top',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_ih.text(0.0,0.1,"largest leverage by count situation",ha='left',va='bottom',fontsize=8,color=GREY,fontfamily='Arial')
    ax_impact=fig.add_axes(m(0.50,mid_b,0.48,mid_h-0.07))
    ax_impact.set_facecolor(BG); ax_impact.axis('off'); ax_impact.set_xlim(0,1); ax_impact.set_ylim(0,1)
    for i,(inning,matchup,situation,description) in enumerate(game["impactful"]):
        y=0.95-i*0.30; rv=game["impactful_rvs"][i] if i<len(game["impactful_rvs"]) else None
        ax_impact.add_patch(Circle((0.04,y-0.04),0.035,color=imp_colors[i],zorder=2))
        tc='black' if imp_colors[i]=='#F1C40F' else 'white'
        ax_impact.text(0.04,y-0.04,str(i+1),ha='center',va='center',fontsize=10,fontweight='bold',color=tc,fontfamily='Arial',zorder=3)
        ax_impact.text(0.12,y,inning,ha='left',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
        ax_impact.text(0.12,y-0.075,matchup,ha='left',va='top',fontsize=9,color=GREY,fontfamily='Arial')
        sit_line=f"{situation}  ·  {rv:.2f} runs" if rv is not None else situation
        ax_impact.text(0.12,y-0.148,sit_line,ha='left',va='top',fontsize=8.5,color=GREY,fontfamily='Arial')
        ax_impact.text(0.12,y-0.218,description,ha='left',va='top',fontsize=8.5,fontweight='bold',color=BLACK,fontfamily='Arial')

    div3_b=mid_b-0.010
    ax_div3=fig.add_axes(m(0.04,div3_b,0.92,0.003)); ax_div3.set_facecolor(LGREY); ax_div3.axis('off')

    bar_b=div3_b-0.005-0.115; bar_h=0.110
    ax_bar1=fig.add_axes(m(0.04,bar_b,0.42,bar_h))
    draw_acc_bar(ax_bar1,gs["ball_acc"],"#2E75B6","Accuracy on called balls",avg_pct=0.90)
    ax_bar2=fig.add_axes(m(0.54,bar_b,0.42,bar_h))
    draw_acc_bar(ax_bar2,gs["strike_acc"],"#CC3333","and called strikes",avg_pct=0.82)

    div4_b=bar_b-0.010
    ax_div4=fig.add_axes(m(0.04,div4_b,0.92,0.003)); ax_div4.set_facecolor(LGREY); ax_div4.axis('off')

    sum_b=div4_b-0.005-0.075
    ax_sum=fig.add_axes(m(0.0,sum_b,1.0,0.075)); ax_sum.set_facecolor(BG); ax_sum.axis('off')
    ax_sum.text(0.0,1.0,"Summary",ha='left',va='top',fontsize=11,fontweight='bold',color=BLACK,fontfamily='Arial')
    summary=(
        f"{ump_name} called {gs['correct']} of {gs['total']} pitches correctly ({gs['acc']*100:.1f}%) "
        f"in {game['home_abb']}'s home game against {game['away_abb']} on {game['date_short']}. "
        "A pitch is a strike if any part of the softball (radius 0.159 ft) touches the zone. "
        f"{'He' if True else 'The ump'} was {'highly ' if gs['ball_acc']>0.95 else ''}accurate on called balls "
        f"({gs['ball_acc']*100:.1f}%, {gs['ball_correct']}/{gs['ball_total']}), "
        f"{'but struggled' if gs['strike_acc']<0.75 else 'and accurate'} on called strikes "
        f"({gs['strike_acc']*100:.1f}%, {gs['strike_correct']}/{gs['strike_total']}), "
        f"issuing {gs['phantom']} ball{'s' if gs['phantom']!=1 else ''} called strike{'s' if gs['phantom']!=1 else ''} and {gs['missed']} strike{'s' if gs['missed']!=1 else ''} called a ball. "
        f"Overall consistency of {game['consistency']*100:.1f}% reflects "
        f"{'good' if game['consistency']>0.78 else 'moderate'} reliability on similar-location pitches, "
        f"with calls leaning {fav_label} ({fav_txt} net)."
    )
    ax_sum.text(0.0,0.68,summary,ha='left',va='top',fontsize=9,color=BLACK,fontfamily='Arial',
                wrap=True,transform=ax_sum.transAxes,multialignment='left')

    add_footer(fig, game)
    pdf.savefig(fig,facecolor=BG); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Pitcher Workbook
# ══════════════════════════════════════════════════════════════════════════════
def build_workbook_page(pdf, game, ump_name, home_color, away_color):
    fig=plt.figure(figsize=(8.5,11),facecolor=BG,dpi=150)
    fig.subplots_adjust(0,0,1,1)
    T=0.97; TITLE_H=0.07
    ax_t=fig.add_axes(m(0.0,T-TITLE_H,1.0,TITLE_H)); ax_t.set_facecolor(BG); ax_t.axis('off')
    title_text(ax_t,"Pitcher Breakdown"); draw_powered_by(ax_t)
    draw_tm_logo(fig,0.490,T-TITLE_H+0.004,0.065,TITLE_H*0.42)

    ax_sub=fig.add_axes(m(0.0,T-TITLE_H-0.028,1.0,0.025)); ax_sub.set_facecolor(BG); ax_sub.axis('off')
    ax_sub.text(0.5,0.5,f"{ump_name}  |  {game['home_abb']} vs {game['away_abb']}  |  {game['date_long']}",
                ha='center',va='center',fontsize=10,color=GREY,fontfamily='Arial')

    ax_div=fig.add_axes(m(0.04,T-TITLE_H-0.035,0.92,0.003)); ax_div.set_facecolor(LGREY); ax_div.axis('off')

    COL_STARTS=[0.35,0.49,0.58,0.67,0.76,0.85,0.93]
    COL_LABELS=["Called","Correct","Accuracy","Ball Acc","Strike Acc","Phantom","Missed"]
    hdr_b=T-TITLE_H-0.075
    ax_hdr=fig.add_axes(m(0.0,hdr_b,1.0,0.035)); ax_hdr.set_facecolor(BG); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)
    for cx,lbl in zip(COL_STARTS,COL_LABELS):
        ax_hdr.text(cx,0.5,lbl,ha='center',va='center',fontsize=8,fontweight='bold',color=GREY,fontfamily='Arial')

    ax_hdiv=fig.add_axes(m(0.04,hdr_b-0.004,0.92,0.002)); ax_hdiv.set_facecolor(LGREY); ax_hdiv.axis('off')

    pitchers=game["pitchers"]; n=len(pitchers)
    avail_h=hdr_b-0.01-0.04; row_h=avail_h/n

    def p_color(p):
        return home_color if p["abb"]==game["home_abb"] else away_color

    for i,p in enumerate(pitchers):
        row_b=hdr_b-0.01-(i+1)*row_h; pst=p["stats"]; pc=p_color(p)
        ax_bar=fig.add_axes(m(0.0,row_b+0.005,0.007,row_h-0.01)); ax_bar.set_facecolor(pc); ax_bar.axis('off')
        ax_info=fig.add_axes(m(0.012,row_b,0.33,row_h)); ax_info.set_facecolor(BG); ax_info.axis('off')
        ax_info.text(0.0,0.80,p["name"],ha='left',va='top',fontsize=13,fontweight='bold',color=pc,fontfamily='Arial')
        ax_info.text(0.0,0.50,p["abb"],ha='left',va='top',fontsize=10,color=GREY,fontfamily='Arial')
        bh=0.14; by=0.12
        ax_info.add_patch(Rectangle((0,by),0.85,bh,facecolor='#E8E8E8',edgecolor='none'))
        ax_info.add_patch(Rectangle((0,by),pst["ball_acc"]*0.85,bh,facecolor='#2E75B6',edgecolor='none'))
        ax_info.add_patch(Rectangle((0,by-0.18),0.85,bh,facecolor='#E8E8E8',edgecolor='none'))
        ax_info.add_patch(Rectangle((0,by-0.18),pst["strike_acc"]*0.85,bh,facecolor=ZONE_BORDER,edgecolor='none'))
        ax_info.text(0.87,by+bh/2,f"Ball {pst['ball_acc']*100:.0f}%",va='center',fontsize=7,color=GREY,fontfamily='Arial')
        ax_info.text(0.87,by-0.18+bh/2,f"Str {pst['strike_acc']*100:.0f}%",va='center',fontsize=7,color=GREY,fontfamily='Arial')
        vals=[str(pst["total"]),str(pst["correct"]),
              f"{pst['acc']*100:.1f}%",f"{pst['ball_acc']*100:.1f}%",
              f"{pst['strike_acc']*100:.1f}%",str(pst["phantom"]),str(pst["missed"])]
        vcols=[BLACK,BLACK,BLACK,BLACK,BLACK,"#CC3333","#2E75B6"]
        ax_vals=fig.add_axes(m(0.0,row_b,1.0,row_h)); ax_vals.set_facecolor('none'); ax_vals.axis('off')
        ax_vals.set_xlim(0,1); ax_vals.set_ylim(0,1)
        for cx,val,vc in zip(COL_STARTS,vals,vcols):
            ax_vals.text(cx,0.55,val,ha='center',va='center',fontsize=11,fontweight='bold',color=vc,fontfamily='Arial')
        if i<n-1:
            ax_rdiv=fig.add_axes(m(0.04,row_b,0.92,0.002)); ax_rdiv.set_facecolor(LGREY); ax_rdiv.axis('off')

    add_footer(fig, game)
    pdf.savefig(fig,facecolor=BG); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3+ — Individual Pitcher Pages
# ══════════════════════════════════════════════════════════════════════════════
def build_pitcher_page(pdf, game, p, ump_name, home_color, away_color):
    fig=plt.figure(figsize=(8.5,11),facecolor=BG,dpi=150)
    fig.subplots_adjust(0,0,1,1)
    pst=p["stats"]; pimp=p["impactful"]
    pimp_pts=[(t[4],t[5]) for t in pimp]; pimp_labels=[(t[0],t[1],t[2],t[3]) for t in pimp]
    pimp_rvs=[t[6] for t in pimp]
    pimp_calls=[t[7] for t in pimp]
    pimp_colors=[_imp_color(c) for c in pimp_calls]
    pc=home_color if p["abb"]==game["home_abb"] else away_color

    T=0.97; TITLE_H=0.092

    # ── Title bar ────────────────────────────────────────────────────────────
    ax_name=fig.add_axes(m(0.0,T-TITLE_H,1.0,TITLE_H)); ax_name.set_facecolor(BG); ax_name.axis('off')
    title_text(ax_name,"Pitcher Report"); draw_powered_by(ax_name)
    draw_tm_logo(fig,0.490,T-TITLE_H+0.004,0.065,TITLE_H*0.42)

    # ── Header: pitcher name / mask / ump+date ────────────────────────────────
    sec2_b=T-TITLE_H-0.13
    ax_teams=fig.add_axes(m(0.01,sec2_b,0.38,0.13)); ax_teams.set_facecolor(BG); ax_teams.axis('off')
    ax_teams.text(0.0,0.85,p["name"],ha='left',va='top',fontsize=20,fontweight='bold',color=pc,fontfamily='Arial')
    ax_teams.text(0.0,0.30,p["abb"],ha='left',va='center',fontsize=22,fontweight='bold',color=pc,fontfamily='Arial')
    if MASK_IMG is not None:
        ax_mask=fig.add_axes(m(0.38,sec2_b-0.01,0.24,0.14)); ax_mask.imshow(MASK_IMG); ax_mask.axis('off')
    ax_date=fig.add_axes(m(0.50,sec2_b,0.50,0.13)); ax_date.set_facecolor(BG); ax_date.axis('off')
    ax_date.text(1.0,0.62,ump_name,ha='right',va='center',fontsize=17,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_date.text(1.0,0.35,game["date_long"],ha='right',va='center',fontsize=14,color=GREY,fontfamily='Arial')

    soc_b=sec2_b-0.004
    ax_div=fig.add_axes(m(0.04,soc_b-0.004,0.92,0.003)); ax_div.set_facecolor(LGREY); ax_div.axis('off')

    # ── 4 Metrics row ─────────────────────────────────────────────────────────
    met_b=soc_b-0.004-0.175; met_h=0.170; met_w=0.22; offsets=[0.01,0.26,0.51,0.76]

    ax_m1=fig.add_axes(m(offsets[0],met_b,met_w,met_h)); ax_m1.set_facecolor(BG); ax_m1.axis('off')
    ax_m1.text(0.5,0.97,"Overall",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m1.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub1=fig.add_axes(m(offsets[0]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub1,pst["acc"],"#2E75B6")
    ax_m1.text(0.5,0.04,f"Called {pst['correct']} of {pst['total']} taken\npitches correctly",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m2=fig.add_axes(m(offsets[1],met_b,met_w,met_h)); ax_m2.set_facecolor(BG); ax_m2.axis('off')
    ax_m2.text(0.5,0.97,"Total Missed",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m2.text(0.5,0.87,"Calls",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m2.text(0.5,0.56,f"{pst['phantom']+pst['missed']}",ha='center',va='center',fontsize=44,fontweight='bold',color="#CC3333",fontfamily='Arial')
    ax_m2.text(0.5,0.38,"incorrect calls",ha='center',va='center',fontsize=9,color=GREY,fontfamily='Arial')
    ax_m2.text(0.5,0.12,f"{pst['phantom']} phantom strikes\n{pst['missed']} missed strikes",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m3=fig.add_axes(m(offsets[2],met_b,met_w,met_h)); ax_m3.set_facecolor(BG); ax_m3.axis('off')
    ax_m3.text(0.5,0.97,"Ball",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m3.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub3=fig.add_axes(m(offsets[2]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub3,pst["ball_acc"],"#2E75B6")
    ax_m3.text(0.5,0.04,f"{pst['ball_correct']} of {pst['ball_total']} called balls\ncorrect",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m4=fig.add_axes(m(offsets[3],met_b,met_w,met_h)); ax_m4.set_facecolor(BG); ax_m4.axis('off')
    ax_m4.text(0.5,0.97,"Strike",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m4.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub4=fig.add_axes(m(offsets[3]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub4,pst["strike_acc"],"#CC3333")
    ax_m4.text(0.5,0.04,f"{pst['strike_correct']} of {pst['strike_total']} called strikes\ncorrect",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    div2_b=met_b-0.010
    ax_div2=fig.add_axes(m(0.04,div2_b,0.92,0.003)); ax_div2.set_facecolor(LGREY); ax_div2.axis('off')

    # ── Zone + Impactful row ───────────────────────────────────────────────────
    mid_b=div2_b-0.005-0.315; mid_h=0.315; LEG_H=0.064

    ax_zh=fig.add_axes(m(0.00,mid_b+mid_h-0.025,0.48,0.035)); ax_zh.set_facecolor(BG); ax_zh.axis('off')
    ax_zh.text(0.0,0.9,"All Missed Calls",ha='left',va='top',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_zh.text(0.0,0.1,"true zone and Estimated Ump Zone (EUZ)",ha='left',va='bottom',fontsize=8,color=GREY,fontfamily='Arial')
    ax_zone=fig.add_axes(m(0.00,mid_b+LEG_H,0.48,mid_h-0.010-LEG_H))
    def _edge_fn(team): return home_color if get_abb(str(team))==game["home_abb"] else away_color
    draw_zone(ax_zone,p["rows"],pst["phantom_pts"],pst["missed_pts"],pst["strike_pts"],pimp_pts,pimp_colors,team_edge_fn=_edge_fn,show_labels=False)
    _tlbls=(game["home_abb"], game["away_abb"])
    _tcolors=(home_color, away_color)
    ax_zleg=fig.add_axes(m(0.00,mid_b,0.48,LEG_H)); draw_zone_legend(ax_zleg,team_labels=_tlbls,team_colors=_tcolors,simple=True)

    ax_ih=fig.add_axes(m(0.50,mid_b+mid_h-0.025,0.48,0.035)); ax_ih.set_facecolor(BG); ax_ih.axis('off')
    ax_ih.text(0.0,0.9,"Impactful Missed Calls",ha='left',va='top',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_ih.text(0.0,0.1,"largest leverage by count situation",ha='left',va='bottom',fontsize=8,color=GREY,fontfamily='Arial')
    ax_impact=fig.add_axes(m(0.50,mid_b,0.48,mid_h-0.07))
    ax_impact.set_facecolor(BG); ax_impact.axis('off'); ax_impact.set_xlim(0,1); ax_impact.set_ylim(0,1)
    if pimp_labels:
        for i,(inning,matchup,situation,description) in enumerate(pimp_labels):
            y=0.95-i*0.30; rv=pimp_rvs[i] if i<len(pimp_rvs) else None
            ax_impact.add_patch(Circle((0.04,y-0.04),0.035,color=pimp_colors[i],zorder=2))
            tc='white'
            ax_impact.text(0.04,y-0.04,str(i+1),ha='center',va='center',fontsize=10,fontweight='bold',color=tc,fontfamily='Arial',zorder=3)
            ax_impact.text(0.12,y,inning,ha='left',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
            ax_impact.text(0.12,y-0.075,matchup,ha='left',va='top',fontsize=9,color=GREY,fontfamily='Arial')
            sit_line=f"{situation}  ·  {rv:.2f} runs" if rv is not None else situation
            ax_impact.text(0.12,y-0.148,sit_line,ha='left',va='top',fontsize=8.5,color=GREY,fontfamily='Arial')
            ax_impact.text(0.12,y-0.218,description,ha='left',va='top',fontsize=8.5,fontweight='bold',color=BLACK,fontfamily='Arial')
    else:
        ax_impact.text(0.5,0.5,"No impactful missed calls",ha='center',va='center',fontsize=10,color=LGREY,fontfamily='Arial')

    div3_b=mid_b-0.010
    ax_div3=fig.add_axes(m(0.04,div3_b,0.92,0.003)); ax_div3.set_facecolor(LGREY); ax_div3.axis('off')

    # ── Accuracy bars ─────────────────────────────────────────────────────────
    bar_b=div3_b-0.005-0.115; bar_h=0.110
    ax_bar1=fig.add_axes(m(0.04,bar_b,0.42,bar_h))
    draw_acc_bar(ax_bar1,pst["ball_acc"],"#2E75B6","Accuracy on called balls",avg_pct=0.90)
    ax_bar2=fig.add_axes(m(0.54,bar_b,0.42,bar_h))
    draw_acc_bar(ax_bar2,pst["strike_acc"],"#CC3333","and called strikes",avg_pct=0.82)

    div4_b=bar_b-0.010
    ax_div4=fig.add_axes(m(0.04,div4_b,0.92,0.003)); ax_div4.set_facecolor(LGREY); ax_div4.axis('off')

    # ── Summary ───────────────────────────────────────────────────────────────
    sum_b=div4_b-0.005-0.075
    ax_sum=fig.add_axes(m(0.0,sum_b,1.0,0.075)); ax_sum.set_facecolor(BG); ax_sum.axis('off')
    ax_sum.text(0.0,1.0,"Summary",ha='left',va='top',fontsize=11,fontweight='bold',color=BLACK,fontfamily='Arial')
    ball_q='highly ' if pst['ball_acc']>0.95 else ('' if pst['ball_acc']>=0.85 else 'less ')
    str_q='highly ' if pst['strike_acc']>0.90 else ('' if pst['strike_acc']>=0.80 else 'less ')
    summary=(
        f"{ump_name} called {pst['correct']} of {pst['total']} pitches correctly ({pst['acc']*100:.1f}%) "
        f"when {p['name']} ({p['abb']}) was pitching in {game['home_abb']}'s game against {game['away_abb']} on {game['date_short']}. "
        "A pitch is a strike if any part of the softball (radius 0.159 ft) touches the zone. "
        f"The umpire was {ball_q}accurate on called balls ({pst['ball_acc']*100:.1f}%, {pst['ball_correct']}/{pst['ball_total']}) "
        f"and {str_q}accurate on called strikes ({pst['strike_acc']*100:.1f}%, {pst['strike_correct']}/{pst['strike_total']}), "
        f"issuing {pst['phantom']} ball{'s' if pst['phantom']!=1 else ''} called strike{'s' if pst['phantom']!=1 else ''} "
        f"and {pst['missed']} strike{'s' if pst['missed']!=1 else ''} called a ball."
    )
    ax_sum.text(0.0,0.68,summary,ha='left',va='top',fontsize=9,color=BLACK,fontfamily='Arial',
                wrap=True,transform=ax_sum.transAxes,multialignment='left')

    add_footer(fig, game)
    pdf.savefig(fig,facecolor=BG); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Individual Catcher Pages
# ══════════════════════════════════════════════════════════════════════════════
def build_catcher_page(pdf, game, c, ump_name, home_color, away_color):
    fig=plt.figure(figsize=(8.5,11),facecolor=BG,dpi=150)
    fig.subplots_adjust(0,0,1,1)
    cst=c["stats"]; cimp=c["impactful"]
    cimp_pts=[(t[4],t[5]) for t in cimp]; cimp_labels=[(t[0],t[1],t[2],t[3]) for t in cimp]
    cimp_rvs=[t[6] for t in cimp]
    cimp_calls=[t[7] for t in cimp]
    cimp_colors=[_imp_color(c) for c in cimp_calls]
    cc=home_color if c["abb"]==game["home_abb"] else away_color

    T=0.97; TITLE_H=0.092

    # ── Title bar ────────────────────────────────────────────────────────────
    ax_name=fig.add_axes(m(0.0,T-TITLE_H,1.0,TITLE_H)); ax_name.set_facecolor(BG); ax_name.axis('off')
    title_text(ax_name,"Catcher Report"); draw_powered_by(ax_name)
    draw_tm_logo(fig,0.490,T-TITLE_H+0.004,0.065,TITLE_H*0.42)

    # ── Header: catcher name / mask / ump+date ────────────────────────────────
    sec2_b=T-TITLE_H-0.13
    ax_teams=fig.add_axes(m(0.01,sec2_b,0.38,0.13)); ax_teams.set_facecolor(BG); ax_teams.axis('off')
    ax_teams.text(0.0,0.85,c["name"],ha='left',va='top',fontsize=20,fontweight='bold',color=cc,fontfamily='Arial')
    ax_teams.text(0.0,0.30,c["abb"],ha='left',va='center',fontsize=22,fontweight='bold',color=cc,fontfamily='Arial')
    if MASK_IMG is not None:
        ax_mask=fig.add_axes(m(0.38,sec2_b-0.01,0.24,0.14)); ax_mask.imshow(MASK_IMG); ax_mask.axis('off')
    ax_date=fig.add_axes(m(0.50,sec2_b,0.50,0.13)); ax_date.set_facecolor(BG); ax_date.axis('off')
    ax_date.text(1.0,0.62,ump_name,ha='right',va='center',fontsize=17,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_date.text(1.0,0.35,game["date_long"],ha='right',va='center',fontsize=14,color=GREY,fontfamily='Arial')

    soc_b=sec2_b-0.004
    ax_div=fig.add_axes(m(0.04,soc_b-0.004,0.92,0.003)); ax_div.set_facecolor(LGREY); ax_div.axis('off')

    # ── 4 Metrics row ─────────────────────────────────────────────────────────
    met_b=soc_b-0.004-0.175; met_h=0.170; met_w=0.22; offsets=[0.01,0.26,0.51,0.76]

    ax_m1=fig.add_axes(m(offsets[0],met_b,met_w,met_h)); ax_m1.set_facecolor(BG); ax_m1.axis('off')
    ax_m1.text(0.5,0.97,"Overall",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m1.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub1=fig.add_axes(m(offsets[0]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub1,cst["acc"],"#2E75B6")
    ax_m1.text(0.5,0.04,f"Called {cst['correct']} of {cst['total']} taken\npitches correctly",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m2=fig.add_axes(m(offsets[1],met_b,met_w,met_h)); ax_m2.set_facecolor(BG); ax_m2.axis('off')
    ax_m2.text(0.5,0.97,"Total Missed",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m2.text(0.5,0.87,"Calls",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m2.text(0.5,0.56,f"{cst['phantom']+cst['missed']}",ha='center',va='center',fontsize=44,fontweight='bold',color="#CC3333",fontfamily='Arial')
    ax_m2.text(0.5,0.38,"incorrect calls",ha='center',va='center',fontsize=9,color=GREY,fontfamily='Arial')
    ax_m2.text(0.5,0.12,f"{cst['phantom']} phantom strikes\n{cst['missed']} missed strikes",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m3=fig.add_axes(m(offsets[2],met_b,met_w,met_h)); ax_m3.set_facecolor(BG); ax_m3.axis('off')
    ax_m3.text(0.5,0.97,"Ball",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m3.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub3=fig.add_axes(m(offsets[2]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub3,cst["ball_acc"],"#2E75B6")
    ax_m3.text(0.5,0.04,f"{cst['ball_correct']} of {cst['ball_total']} called balls\ncorrect",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    ax_m4=fig.add_axes(m(offsets[3],met_b,met_w,met_h)); ax_m4.set_facecolor(BG); ax_m4.axis('off')
    ax_m4.text(0.5,0.97,"Strike",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_m4.text(0.5,0.87,"Accuracy",ha='center',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_sub4=fig.add_axes(m(offsets[3]+0.01,met_b+0.026,met_w-0.02,met_h*0.65))
    draw_donut(ax_sub4,cst["strike_acc"],"#CC3333")
    ax_m4.text(0.5,0.04,f"{cst['strike_correct']} of {cst['strike_total']} called strikes\ncorrect",
               ha='center',va='bottom',fontsize=7.5,color=GREY,fontfamily='Arial',linespacing=1.4)

    div2_b=met_b-0.010
    ax_div2=fig.add_axes(m(0.04,div2_b,0.92,0.003)); ax_div2.set_facecolor(LGREY); ax_div2.axis('off')

    # ── Zone + Impactful row ───────────────────────────────────────────────────
    mid_b=div2_b-0.005-0.315; mid_h=0.315; LEG_H=0.064

    ax_zh=fig.add_axes(m(0.00,mid_b+mid_h-0.025,0.48,0.035)); ax_zh.set_facecolor(BG); ax_zh.axis('off')
    ax_zh.text(0.0,0.9,"All Missed Calls",ha='left',va='top',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_zh.text(0.0,0.1,"true zone and Estimated Ump Zone (EUZ)",ha='left',va='bottom',fontsize=8,color=GREY,fontfamily='Arial')
    ax_zone=fig.add_axes(m(0.00,mid_b+LEG_H,0.48,mid_h-0.010-LEG_H))
    def _edge_fn(team): return home_color if get_abb(str(team))==game["home_abb"] else away_color
    draw_zone(ax_zone,c["rows"],cst["phantom_pts"],cst["missed_pts"],cst["strike_pts"],cimp_pts,cimp_colors,team_edge_fn=_edge_fn,show_labels=False)
    _tlbls=(game["home_abb"], game["away_abb"])
    _tcolors=(home_color, away_color)
    ax_zleg=fig.add_axes(m(0.00,mid_b,0.48,LEG_H)); draw_zone_legend(ax_zleg,team_labels=_tlbls,team_colors=_tcolors,simple=True)

    ax_ih=fig.add_axes(m(0.50,mid_b+mid_h-0.025,0.48,0.035)); ax_ih.set_facecolor(BG); ax_ih.axis('off')
    ax_ih.text(0.0,0.9,"Impactful Missed Calls",ha='left',va='top',fontsize=13,fontweight='bold',color=BLACK,fontfamily='Arial')
    ax_ih.text(0.0,0.1,"largest leverage by count situation",ha='left',va='bottom',fontsize=8,color=GREY,fontfamily='Arial')
    ax_impact=fig.add_axes(m(0.50,mid_b,0.48,mid_h-0.07))
    ax_impact.set_facecolor(BG); ax_impact.axis('off'); ax_impact.set_xlim(0,1); ax_impact.set_ylim(0,1)
    if cimp_labels:
        for i,(inning,matchup,situation,description) in enumerate(cimp_labels):
            y=0.95-i*0.30; rv=cimp_rvs[i] if i<len(cimp_rvs) else None
            ax_impact.add_patch(Circle((0.04,y-0.04),0.035,color=cimp_colors[i],zorder=2))
            tc='white'
            ax_impact.text(0.04,y-0.04,str(i+1),ha='center',va='center',fontsize=10,fontweight='bold',color=tc,fontfamily='Arial',zorder=3)
            ax_impact.text(0.12,y,inning,ha='left',va='top',fontsize=10,fontweight='bold',color=BLACK,fontfamily='Arial')
            ax_impact.text(0.12,y-0.075,matchup,ha='left',va='top',fontsize=9,color=GREY,fontfamily='Arial')
            sit_line=f"{situation}  ·  {rv:.2f} runs" if rv is not None else situation
            ax_impact.text(0.12,y-0.148,sit_line,ha='left',va='top',fontsize=8.5,color=GREY,fontfamily='Arial')
            ax_impact.text(0.12,y-0.218,description,ha='left',va='top',fontsize=8.5,fontweight='bold',color=BLACK,fontfamily='Arial')
    else:
        ax_impact.text(0.5,0.5,"No impactful missed calls",ha='center',va='center',fontsize=10,color=LGREY,fontfamily='Arial')

    div3_b=mid_b-0.010
    ax_div3=fig.add_axes(m(0.04,div3_b,0.92,0.003)); ax_div3.set_facecolor(LGREY); ax_div3.axis('off')

    # ── Accuracy bars ─────────────────────────────────────────────────────────
    bar_b=div3_b-0.005-0.115; bar_h=0.110
    ax_bar1=fig.add_axes(m(0.04,bar_b,0.42,bar_h))
    draw_acc_bar(ax_bar1,cst["ball_acc"],"#2E75B6","Accuracy on called balls",avg_pct=0.90)
    ax_bar2=fig.add_axes(m(0.54,bar_b,0.42,bar_h))
    draw_acc_bar(ax_bar2,cst["strike_acc"],"#CC3333","and called strikes",avg_pct=0.82)

    div4_b=bar_b-0.010
    ax_div4=fig.add_axes(m(0.04,div4_b,0.92,0.003)); ax_div4.set_facecolor(LGREY); ax_div4.axis('off')

    # ── Summary ───────────────────────────────────────────────────────────────
    sum_b=div4_b-0.005-0.075
    ax_sum=fig.add_axes(m(0.0,sum_b,1.0,0.075)); ax_sum.set_facecolor(BG); ax_sum.axis('off')
    ax_sum.text(0.0,1.0,"Summary",ha='left',va='top',fontsize=11,fontweight='bold',color=BLACK,fontfamily='Arial')
    ball_q='highly ' if cst['ball_acc']>0.95 else ('' if cst['ball_acc']>=0.85 else 'less ')
    str_q='highly ' if cst['strike_acc']>0.90 else ('' if cst['strike_acc']>=0.80 else 'less ')
    summary=(
        f"{ump_name} called {cst['correct']} of {cst['total']} pitches correctly ({cst['acc']*100:.1f}%) "
        f"while {c['name']} ({c['abb']}) was catching in {game['home_abb']}'s game against {game['away_abb']} on {game['date_short']}. "
        "A pitch is a strike if any part of the softball (radius 0.159 ft) touches the zone. "
        f"The umpire was {ball_q}accurate on called balls ({cst['ball_acc']*100:.1f}%, {cst['ball_correct']}/{cst['ball_total']}) "
        f"and {str_q}accurate on called strikes ({cst['strike_acc']*100:.1f}%, {cst['strike_correct']}/{cst['strike_total']}), "
        f"issuing {cst['phantom']} ball{'s' if cst['phantom']!=1 else ''} called strike{'s' if cst['phantom']!=1 else ''} "
        f"and {cst['missed']} strike{'s' if cst['missed']!=1 else ''} called a ball."
    )
    ax_sum.text(0.0,0.68,summary,ha='left',va='top',fontsize=9,color=BLACK,fontfamily='Arial',
                wrap=True,transform=ax_sum.transAxes,multialignment='left')

    add_footer(fig, game)
    pdf.savefig(fig,facecolor=BG); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PDF generation
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(game, ump_name, home_color, away_color) -> bytes:
    buf=io.BytesIO()
    with PdfPages(buf) as pdf:
        build_main_page(pdf, game, ump_name, home_color, away_color)
        build_workbook_page(pdf, game, ump_name, home_color, away_color)
        for p in game["pitchers"]:
            build_pitcher_page(pdf, game, p, ump_name, home_color, away_color)
        for c in game["catchers"]:
            build_catcher_page(pdf, game, c, ump_name, home_color, away_color)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Umpire Scorecard Generator", page_icon="⚾", layout="wide")

st.title("Homeplate Umpire Scorecard Generator")
st.caption("Upload a Trackman softball CSV and enter the umpire's name to generate a full multi-page PDF scorecard.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Trackman CSV", type=["csv"])
    ump_name = st.text_input("Umpire Name", placeholder="e.g. Ron Burkhart")
    st.divider()
    st.subheader("Team Colors")
    st.caption("Auto-detected from team ID. Override here if needed.")
    home_color_input = st.color_picker("Home Team Color", "#841617")
    away_color_input = st.color_picker("Away Team Color", "#005A9C")
    auto_color = st.checkbox("Auto-detect colors from team ID", value=True)

if uploaded is None:
    st.info("Upload a Trackman CSV in the sidebar to get started.")
    st.markdown("""
**What this generates:**
- **Page 1** — Full game scorecard: accuracy, missed calls, favor, consistency, zone chart, impactful plays
- **Page 2** — Pitcher workbook: all pitchers side-by-side with stats
- **Pages 3+** — Individual pitcher pages: zone chart, accuracy breakdown, impactful calls per pitcher

**Required CSV columns:** `PlateLocSide`, `PlateLocHeight`, `PitchCall`, `Pitcher`, `PitcherTeam`,
`HomeTeam`, `AwayTeam`, `RunsScored`, `Inning`, `Top/Bottom`, `Outs`, `Balls`, `Strikes`, `Date`
    """)
    st.stop()

# Load and process data
with st.spinner("Reading CSV..."):
    try:
        rows = load_csv(uploaded)
        game = process_game(rows)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

# Apply auto-color if checked
if auto_color:
    home_color = guess_color(game["home_abb"], home_color_input)
    away_color = guess_color(game["away_abb"], away_color_input)
else:
    home_color = home_color_input
    away_color = away_color_input

gs = game["gs"]

# Preview stats
st.subheader(f"{game['home_abb']} {game['home_score']} – {game['away_score']} {game['away_abb']}   ·   {game['date_long']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Accuracy", f"{gs['acc']*100:.1f}%", f"{gs['correct']}/{gs['total']}")
col2.metric("Missed Calls", gs["phantom"]+gs["missed"], f"{gs['phantom']} phantom · {gs['missed']} missed")
col3.metric("Ball Accuracy", f"{gs['ball_acc']*100:.1f}%", f"{gs['ball_correct']}/{gs['ball_total']}")
col4.metric("Strike Accuracy", f"{gs['strike_acc']*100:.1f}%", f"{gs['strike_correct']}/{gs['strike_total']}")

st.caption(f"Consistency: {game['consistency']*100:.1f}%  ·  Pitchers: {len(game['pitchers'])}")

with st.expander("Pitcher breakdown"):
    for p in game["pitchers"]:
        pst = p["stats"]
        st.markdown(
            f"**{p['name']}** ({p['abb']}) — "
            f"{pst['correct']}/{pst['total']} correct ({pst['acc']*100:.1f}%)  ·  "
            f"Ball {pst['ball_acc']*100:.1f}%  ·  Strike {pst['strike_acc']*100:.1f}%  ·  "
            f"{pst['phantom']} phantom  ·  {pst['missed']} missed"
        )

st.divider()
if not ump_name.strip():
    st.warning("Enter the umpire's name in the sidebar to generate the PDF.")
    st.stop()

if st.button("Generate Scorecard PDF", type="primary", use_container_width=True):
    with st.spinner("Building PDF..."):
        try:
            pdf_bytes = generate_pdf(game, ump_name.strip(), home_color, away_color)
            safe_ump = ump_name.strip().replace(" ","_")
            filename = f"UmpireScorecard_{game['home_abb']}vs{game['away_abb']}_{game['date_short'].replace('/','')}.pdf"
            st.success(f"PDF ready — {len(game['pitchers'])+2} pages")
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            import traceback; st.code(traceback.format_exc())
