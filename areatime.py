"""
=====================================================================
  PIPELINE ANOMALY / LEAK DETECTOR  —  Pertamina EP Jambi Field
  v5: multi-anomali + KM & maps + filter manual + klasifikasi risiko
      + narasi broadcast + mode sensitif
=====================================================================
Alur: pilih jalur -> upload Excel (kol1=tgl, kol2=jam, kol3..=sensor hulu→hilir,
header bebas, OFF=kosong) -> app deteksi RANGE JAM anomali + KM + maps + narasi.
Metode: HGL elevation-corrected + segment friction-slope vs baseline sehat (top-20%).
Semua hasil = INDIKASI; verifikasi lapangan/mass-balance wajib.
=====================================================================
"""
import os, re
from math import radians, sin, cos, sqrt, atan2
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROUTES = {
    "KAS → TPN": dict(elev_file="kas_elevasi.xlsx", length_km=23.2, diameter_in=6.065,
                      sg=0.85, flow_rate=320, sensor_kp=[0.0, 7.8, 15.4, 23.2],
                      terminal_backpressure=True, history=[3.18, 7.6]),
    "BJG → TPN": dict(elev_file="bjg_elevasi.xlsx", length_km=26.6, diameter_in=4.026,
                      sg=0.85, flow_rate=92, sensor_kp=[0.0, 7.14, 15.4, 19.7],
                      terminal_backpressure=False, history=[14.6, 16.8, 18.8]),
    "KTT → KAS": dict(elev_file="ktt_elevasi.xlsx", length_km=44.6, diameter_in=4.026,
                      sg=0.85, flow_rate=44.31, sensor_kp=[0.0, 44.6],
                      terminal_backpressure=False, history=[]),
    "SG → KAS":  dict(elev_file="sg_elevasi.xlsx", length_km=11.2, diameter_in=3.068,
                      sg=0.85, flow_rate=267, sensor_kp=[0.0, 11.2],
                      terminal_backpressure=False, history=[]),
    "BTJ → BJG": dict(elev_file="btj_elevasi.xlsx", length_km=13.8, diameter_in=4.026,
                      sg=0.85, flow_rate=54.7, sensor_kp=[0.0, 5.5, 9.7],
                      terminal_backpressure=False, history=[]),
}

G = 9.81; PUMP_ON_PSI = 50.0; FRICTION_EXP = 1.85; RESAMPLE = "10min"
MERGE_GAP_MIN = 30

def psi2m(sg): return 6894.76/(sg*1000*G)
def hav(a,b,c,d):
    R=6371000.0;p1,p2=radians(a),radians(c);dp=radians(c-a);dl=radians(d-b)
    x=sin(dp/2)**2+cos(p1)*cos(p2)*sin(dl/2)**2;return 2*R*atan2(sqrt(x),sqrt(1-x))

@st.cache_data(show_spinner=False)
def load_elev(path, L):
    el=pd.read_excel(path); c={x.lower():x for x in el.columns}
    lat=el[c.get("latitude",el.columns[0])].values.astype(float)
    lon=el[c.get("longitude",el.columns[1])].values.astype(float)
    altc=next((c[x] for x in c if "alt" in x),el.columns[-1]); alt=el[altc].values.astype(float)
    d=[0.0]
    for i in range(1,len(el)): d.append(d[-1]+hav(lat[i-1],lon[i-1],lat[i],lon[i]))
    kp=np.array(d)/1000.0
    if kp[-1]>0: kp=kp*(L/kp[-1])
    return kp,alt,lat,lon
def zof(kp,z,x): return float(np.interp(x,kp,z))
def coord(lat,lon,kp,x): return float(np.interp(x,kp,lat)),float(np.interp(x,kp,lon))
def maps_link(lat,lon,kp,x):
    la,lo=coord(lat,lon,kp,x); return f"https://www.google.com/maps?q={la:.6f},{lo:.6f}"

def _hdr_kp(name):
    m=re.findall(r"\d+\.?\d*",str(name)); return float(m[0]) if m else None

def parse_pressure(file, sensor_kp):
    n=len(sensor_kp); df=pd.read_excel(file); cols=list(df.columns)
    low=[str(c).lower() for c in cols]
    date_col=next((cols[i] for i,c in enumerate(low) if "date" in c or "tang" in c),cols[0])
    time_col=next((cols[i] for i,c in enumerate(low) if "time" in c or "waktu" in c or "jam" in c),cols[1])
    scols=[c for c in cols if c not in (date_col,time_col)]
    t=df[time_col].astype(str).str.replace("-",":",regex=False)
    df["_ts"]=pd.to_datetime(df[date_col].astype(str)+" "+t,errors="coerce")
    for c in scols: df[c]=pd.to_numeric(df[c],errors="coerce")
    parsed=[_hdr_kp(c) for c in scols]; mode="positional (kolom = urutan hulu→hilir)"; mapping={}
    if all(p is not None for p in parsed) and len(scols)==n:
        used,ok,tmp=set(),True,{}
        for c,p in zip(scols,parsed):
            cand=[j for j in range(n) if abs(sensor_kp[j]-p)<=0.4 and j not in used]
            if not cand: ok=False;break
            j=min(cand,key=lambda x:abs(sensor_kp[x]-p));used.add(j);tmp[j]=c
        if ok: mapping=tmp;mode="by-header-KP"
    if not mapping:
        for i in range(min(n,len(scols))): mapping[i]=scols[i]
    for j,c in mapping.items(): df[f"S{j}"]=df[c]
    warn=None
    on=df[df.get("S0",pd.Series(dtype=float))>PUMP_ON_PSI] if "S0" in df else df.iloc[0:0]
    if len(on)>30:
        means=[on[f"S{j}"].mean() if f"S{j}" in df else np.nan for j in range(n)]
        seq=[m for m in means if not np.isnan(m)]
        if len(seq)>=3 and not all(seq[i]>=seq[i+1]-3 for i in range(len(seq)-1)):
            order=[c for c,_ in sorted({mapping[j]:on[f"S{j}"].mean() for j in range(n) if f"S{j}" in df}.items(),key=lambda x:-x[1])]
            for j in range(min(n,len(order))): df[f"S{j}"]=df[order[j]];mapping[j]=order[j]
            mode="auto-reorder by tekanan (kolom tidak urut)"
            warn="⚠️ Urutan kolom terdeteksi tidak hulu→hilir — otomatis diurutkan by tekanan pompa-ON."
    return df,mapping,scols,mode,warn

# ---- deteksi anomali ----
def detect(df,cfg,kp_p,z_p,thresh=0.08,min_dur=15,exclude=None):
    sg=cfg["sg"];k=psi2m(sg);skp=cfg["sensor_kp"];n=len(skp)
    Z={i:zof(kp_p,z_p,skp[i]) for i in range(n)}
    scols=[f"S{i}" for i in range(n) if f"S{i}" in df.columns]
    if not scols: return {"error":"Tidak ada kolom sensor."}
    d=df.set_index("_ts").sort_index()
    r_all=d[scols].resample(RESAMPLE).mean()
    on=(d[scols].max(axis=1)>PUMP_ON_PSI).resample(RESAMPLE).mean()>0.5
    r=r_all[on.reindex(r_all.index,fill_value=False)]
    if len(r)<6: return {"error":"Periode pompa-ON tidak cukup."}
    idx=list(r.index);stp=pd.Timedelta(RESAMPLE);keep=np.ones(len(idx),dtype=bool)
    starts=[i for i in range(len(idx)) if i==0 or (idx[i]-idx[i-1])>stp*1.5]
    for s in starts:                       # skip 3 window startup ramp
        for o in range(3):
            if s+o<len(idx): keep[s+o]=False
    for i in range(len(idx)-1):
        if (idx[i+1]-idx[i])>stp*1.5: keep[i]=False
    keep[-1]=False
    diff=r.diff().abs().max(axis=1)         # skip lonjakan besar (startup/step ekstrim), switch dips lolos
    keep&=(diff.fillna(99).values<25.0)
    if exclude:
        for (a,b) in exclude:
            for i,t in enumerate(idx):
                if a<=t.time()<=b: keep[i]=False
    r=r[keep]
    if len(r)<4: return {"error":"Data steady-state tidak cukup."}
    H=pd.DataFrame(index=r.index)
    for i in range(n):
        if f"S{i}" in r: H[i]=Z[i]+r[f"S{i}"]*k
    seg=[(i,i+1) for i in range(n-1) if i in H and (i+1) in H]
    if not seg: return {"error":"Tak ada pasangan sensor berurutan hidup."}
    slope={p:(H[p[0]]-H[p[1]])/(skp[p[1]]-skp[p[0]]) for p in seg}
    def base(s):
        v=s.dropna().values
        if len(v)<3: return np.nan
        top=v[v>=np.nanpercentile(v,80)];return float(np.mean(top)) if len(top) else np.nan
    B={p:base(slope[p]) for p in seg}
    single=(len(seg)==1)
    def reli(p):
        i,j=p
        if np.isnan(B[p]) or B[p]<1.5: return False
        if i==0: return False
        if j==n-1 and cfg.get("terminal_backpressure"): return False
        return True
    rel=[p for p in seg if reli(p)]
    fd={p:(1-(slope[p]/B[p])**(1/FRICTION_EXP)).clip(lower=0) for p in seg}
    rows=[]
    for t in r.index:
        pool=rel if rel else ([seg[-1]] if single else [])
        best=None
        for p in pool:
            v=fd[p].get(t,np.nan)
            if np.isnan(v): continue
            if best is None or v>best[1]: best=(p,v)
        if best and best[1]>=thresh: rows.append((t,best[1],best[0]))
        else: rows.append((t,0.0,None))
    scored=pd.DataFrame(rows,columns=["t","score","seg"]).set_index("t")
    flags=scored[scored["score"]>0]
    intervals=[]
    if len(flags):
        cs=ce=None;segs=[];prev=None
        for t,row in flags.iterrows():
            if prev is None or (t-prev)>pd.Timedelta(minutes=MERGE_GAP_MIN):
                if cs is not None: intervals.append((cs,ce,segs))
                cs=t;segs=[]
            ce=t+pd.Timedelta(RESAMPLE);segs.append(row["seg"]);prev=t
        intervals.append((cs,ce,segs))
    findings=[]
    for (t0,t1,segs) in intervals:
        dur=(t1-t0).total_seconds()/60
        if dur<min_dur: continue
        sg_best=max(set(segs),key=segs.count);i,j=sg_best
        lo,hi=skp[i],skp[j];pt=lo+(hi-lo)*0.30
        sub=fd[sg_best].loc[t0:t1];avg=float(np.nanmean(sub)) if len(sub) else 0.0
        findings.append(dict(start=t0,end=t1,dur_min=dur,seg=sg_best,kp_lo=lo,kp_hi=hi,
                             point=pt,flow_drop=avg,rate=avg*cfg.get("flow_rate",0),single=single))
    return dict(findings=findings,scored=scored,r=r,single=single)

# ---- klasifikasi risiko titik ----
def valleys(kp_p,z_p,L):
    xx=np.arange(0,L,0.2);zz=np.array([zof(kp_p,z_p,x) for x in xx]);mn=[]
    for i in range(2,len(zz)-2):
        if zz[i]<zz[i-1] and zz[i]<zz[i+1] and zz[i]<zz[i-2] and zz[i]<zz[i+2]:
            if not mn or abs(xx[i]-mn[-1][0])>1.2: mn.append((xx[i],zz[i]))
    return mn

def classify(kp,cfg,kp_p,z_p):
    tags=[]
    for h in cfg.get("history",[]):
        if abs(kp-h)<2.0: tags.append(f"rawan BOCOR (dekat titik histori KP{h:g})");break
    vs=valleys(kp_p,z_p,cfg["length_km"])
    if vs:
        nv=min(vs,key=lambda v:abs(v[0]-kp))
        if abs(nv[0]-kp)<1.5:
            tags.append(f"rawan KOROSI (lembah KP{nv[0]:.1f} ~{nv[1]:.0f}m, air/sedimen bottom-of-line)")
    if kp<cfg["length_km"]*0.15: tags.append("zona near-pump (cek fitting/valve/discharge)")
    if not tags: tags.append("cek umum (sambungan/mekanikal)")
    return "; ".join(tags)

# ---- chart ----
def make_chart(df,cfg,kp_p,z_p,kept,route):
    skp=cfg["sensor_kp"];n=len(skp)
    fig,(a1,a2)=plt.subplots(2,1,figsize=(13,8),dpi=110,gridspec_kw={"height_ratios":[2,1]})
    col=["#29b6f6","#ffa726","#26c281","#ec407a","#7e57c2"]
    for i in range(n):
        if f"S{i}" in df:
            s=df[["_ts",f"S{i}"]].dropna();a1.plot(s["_ts"],s[f"S{i}"],".",ms=1.5,color=col[i%5],label=f"KP{skp[i]:g}")
    for idx,f in enumerate(kept,1):
        a1.axvspan(f["start"],f["end"],color="#e53935",alpha=.18)
        a1.annotate(f"#{idx} {f['start'].strftime('%H:%M')}–{f['end'].strftime('%H:%M')}",
                    (f["start"],a1.get_ylim()[1]*0.97),fontsize=7,color="#c62828")
    a1.set_ylabel("Pressure (psi)")
    v="AMAN — tidak ada anomali" if not kept else f"{len(kept)} anomali"
    a1.set_title(f"{route} — {v}",fontweight="bold");a1.legend(fontsize=7,ncol=n,loc="upper left")
    a1.grid(alpha=.25);a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a2.fill_between(kp_p,z_p,0,color="#d7cbb6",alpha=.5);a2.plot(kp_p,z_p,color="#5C4033",lw=1.1,label="Elevasi")
    for i in range(n):
        a2.plot(skp[i],zof(kp_p,z_p,skp[i]),"o",ms=6,color="#1565C0")
        a2.annotate(f"KP{skp[i]:g}",(skp[i],zof(kp_p,z_p,skp[i])),textcoords="offset points",xytext=(0,6),fontsize=7,ha="center",color="#1565C0")
    for f in kept:
        a2.axvspan(f["kp_lo"],f["kp_hi"],color="#e53935",alpha=.20)
        a2.plot(f["point"],zof(kp_p,z_p,f["point"]),"*",ms=16,color="#c62828",zorder=6)
    km="—" if not kept else ", ".join(f"KP{f['kp_lo']:g}–{f['kp_hi']:g}" for f in kept)
    a2.set_xlabel(f"KP (km) — panjang {cfg['length_km']} km   {route}");a2.set_ylabel("Elevasi (m)")
    a2.set_title(f"KM perlu dipatroli: {km}",fontsize=10);a2.set_xlim(0,cfg["length_km"]);a2.grid(alpha=.2);a2.legend(fontsize=8)
    fig.tight_layout();return fig

# ---- narasi broadcast ----
def broadcast(route,cfg,kept,t0,t1,kp_p,z_p,lat,lon):
    d0=t0.strftime("%d/%m/%Y %H:%M");d1=t1.strftime("%d/%m/%Y %H:%M")
    L=[f"📢 LAPORAN MONITORING PIPA — {route}",f"Periode data: {d0} s/d {d1}",""]
    if not kept:
        L+=["Hasil: ✅ KONDISI AMAN — tidak terdeteksi pola tekanan abnormal.",
            "Rekomendasi: monitoring rutin dilanjutkan.",
            "Status: berbasis analisa tekanan (HGL)."]
        return "\n".join(L)
    L.append(f"Hasil: ⚠️ TERDETEKSI {len(kept)} INDIKASI ANOMALI (potensi bocor):")
    for i,f in enumerate(kept,1):
        risk=classify(f["point"],cfg,kp_p,z_p)
        loc=(f"segmen KP{f['kp_lo']:g}–{f['kp_hi']:g} (jalur 2-sensor: deteksi saja)"
             if f["single"] else f"KP{f['kp_lo']:g}–{f['kp_hi']:g} (estimasi ~KP{f['point']:.1f})")
        L+=["",f"  {i}. Jam {f['start'].strftime('%H:%M')}–{f['end'].strftime('%H:%M')} "
              f"({f['dur_min']:.0f} menit)",
            f"     • Lokasi: {loc}",
            f"     • Indikasi: flow-drop ~{f['flow_drop']*100:.0f}% (~{f['rate']:.0f} bbl/jam)",
            f"     • Sifat titik: {risk}",
            f"     • Patroli KM: {f['kp_lo']:g}–{f['kp_hi']:g}",
            f"     • Maps titik: {maps_link(lat,lon,kp_p,f['point'])}"]
    zones=", ".join(f"KP{f['kp_lo']:g}–{f['kp_hi']:g}" for f in kept)
    L+=["",f"REKOMENDASI: patroli & verifikasi lapangan di {zones}.",
        "Metode verifikasi: UT thickness, visual, ILI, cross-check mass-balance.",
        "Status: INDIKASI berbasis tekanan — belum confirmed sampai temuan lapangan."]
    return "\n".join(L)

# =====================================================================
#  UI
# =====================================================================
st.set_page_config(page_title="Pipeline Anomaly Detector",layout="wide")
st.title("🛢️ Pipeline Anomaly / Leak Detector — Jambi Field")
st.caption("Deteksi otomatis range JAM anomali + KM patroli + maps + narasi broadcast. HGL elevation-corrected.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    route=st.selectbox("Pilih jalur trunkline",list(ROUTES.keys()))
    cfg=dict(ROUTES[route])
    st.markdown(f"**Panjang:** {cfg['length_km']} km · **Ø:** {cfg['diameter_in']}\"")
    st.markdown(f"**Sensor ({len(cfg['sensor_kp'])}):** KP "+", ".join(f"{x:g}" for x in cfg["sensor_kp"]))
    cfg["sg"]=st.number_input("SG crude",0.70,1.00,float(cfg["sg"]),0.01)
    cfg["flow_rate"]=st.number_input("Rate pompa (bbl/jam)",0.0,5000.0,float(cfg["flow_rate"]),1.0)
    st.markdown("---")
    sensitif=st.toggle("🔬 Mode sensitif",value=True,help="Tangkap lebih banyak drop (termasuk yg kecil). Filter operasi manual di bawah.")
    thr=st.slider("Ambang flow-drop (%)",4,30,6 if sensitif else 12)/100
    mindur=st.slider("Durasi minimal anomali (menit)",10,60,15 if sensitif else 30,5)

elev_path=os.path.join(BASE_DIR,cfg["elev_file"])
if not os.path.exists(elev_path): st.error(f"File elevasi '{cfg['elev_file']}' tak ada."); st.stop()
kp_p,z_p,lat,lon=load_elev(elev_path,cfg["length_km"])

st.markdown(f"### Jalur: {route}")
st.info("Format Excel: **kol1=tanggal, kol2=waktu, kol3..=sensor hulu→hilir** (header bebas, OFF=kosong).")
up=st.file_uploader("Upload Excel tekanan",type=["xlsx","xls"])
if up is None: st.stop()

df,mapping,scols,mode,warn=parse_pressure(up,cfg["sensor_kp"])
days=df["_ts"].dt.date.dropna().unique()
st.success(f"Terbaca {len(df):,} baris · {len(days)} hari ({df['_ts'].min()} → {df['_ts'].max()})")
st.caption(f"Pemetaan sensor: **{mode}**")
if warn: st.warning(warn)

# exclude windows manual (opsional)
exc_raw=st.text_input("Exclude jam operasi (opsional, format HH:MM-HH:MM, pisah koma)","")
exclude=[]
for part in exc_raw.split(","):
    part=part.strip()
    if "-" in part:
        try:
            a,b=part.split("-");exclude.append((pd.to_datetime(a.strip()).time(),pd.to_datetime(b.strip()).time()))
        except: pass

res=detect(df,cfg,kp_p,z_p,thresh=thr,min_dur=mindur,exclude=exclude)
if "error" in res: st.error(res["error"]); st.stop()
allf=res["findings"]

st.markdown("## Hasil deteksi")
if not allf:
    st.success("✅ **AMAN** — tidak ada anomali terdeteksi.")
    kept=[]
else:
    st.error(f"🔴 {len(allf)} kandidat anomali. **Hilangkan centang** kalau item itu ternyata pola operasi.")
    kept=[]
    for i,f in enumerate(allf):
        risk=classify(f["point"],cfg,kp_p,z_p)
        c1,c2=st.columns([1,11])
        with c1:
            k=st.checkbox("",value=True,key=f"keep_{i}")
        with c2:
            loc=(f"segmen KP{f['kp_lo']:g}–{f['kp_hi']:g}" if f["single"]
                 else f"KP{f['kp_lo']:g}–{f['kp_hi']:g} (~KP{f['point']:.1f})")
            st.markdown(f"**#{i+1} · {f['start'].strftime('%d %b %H:%M')}–{f['end'].strftime('%H:%M')}** "
                        f"({f['dur_min']:.0f} m) — {loc} · flow-drop ~{f['flow_drop']*100:.0f}% "
                        f"(~{f['rate']:.0f} bph)")
            st.caption(f"🧭 {risk}  ·  📍 [maps titik]({maps_link(lat,lon,kp_p,f['point'])}) · "
                       f"[hulu]({maps_link(lat,lon,kp_p,f['kp_lo'])}) · [hilir]({maps_link(lat,lon,kp_p,f['kp_hi'])})")
        if k: kept.append(f)
    st.warning("Status = **INDIKASI** berbasis tekanan. Verifikasi lapangan (UT/visual/ILI) + mass-balance wajib.")

st.pyplot(make_chart(df,cfg,kp_p,z_p,kept,route))

st.markdown("## 📢 Narasi broadcast (siap salin)")
msg=broadcast(route,cfg,kept,df["_ts"].min(),df["_ts"].max(),kp_p,z_p,lat,lon)
st.code(msg,language=None)

with st.expander("ℹ️ Catatan metode & elevasi"):
    st.markdown(
        f"Head = elevasi + P×{psi2m(cfg['sg']):.3f} → penurunan akibat elevasi dinormalisasi; "
        "yang dideteksi hanya gradien friksi (indikasi flow/bocor). Baseline = rata-rata window "
        "tersehat (top-20%). Near-pump & terminal back-pressure low-confidence. "
        "Mode sensitif menurunkan ambang & durasi minimal — bisa nangkap event operasi (switch tangki), "
        "makanya disediakan filter manual (hapus centang) + input exclude jam operasi.")
st.divider()
st.caption("Semua hasil = indikasi berbasis tekanan; verifikasi lapangan wajib. | Tambah jalur: edit ROUTES.")
