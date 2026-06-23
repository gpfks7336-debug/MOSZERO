import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
import streamlit as st
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="MOS-ZERO Error Analyzer", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', 'Noto Sans KR', sans-serif; 
        background-color: #F1F5F9 !important;
    }
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 94% !important; }

    .main-title { font-size: 2.4rem; font-weight: 800; color: #0f172a; letter-spacing: -1.5px; margin-bottom: 4px; text-align: center; }
    .main-title span { color: #0284C7; }
    .subtitle { font-size: 1rem; color: #64748b; font-weight: 400; margin-top: 4px; padding-bottom: 10px; text-align: center; }
    
    .section-header { font-size: 1rem; font-weight: 600; color: #334155; margin-top: 1rem; margin-bottom: 0.6rem; border-left: 3px solid #0284C7; padding-left: 10px; }
    
    .feature-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .feature-table th { background: #f1f5f9; color: #334155; font-weight: 600; padding: 6px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }
    .feature-table td { padding: 5px 10px; color: #475569; border-bottom: 1px solid #f1f5f9; }
    .feature-table tr:last-child td { border-bottom: none; }

    .report-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .rank-label { font-size: 0.82rem; font-weight: 700; color: #64748b; margin-bottom: 8px; }
    .defect-name-1 { font-size: 1.6rem; font-weight: 800; color: #0284C7; margin-bottom: 8px; line-height: 1.3; }
    .defect-name-2 { font-size: 1.2rem; font-weight: 700; color: #0f172a; margin-bottom: 8px; line-height: 1.3; }
    .defect-desc { font-size: 0.88rem; color: #64748b; line-height: 1.6; }

    .ref-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px 20px; margin-top: 8px;
    }
    .ref-meta { font-size: 0.75rem; color: #94a3b8; margin-bottom: 8px; }
    .ref-link { font-size: 0.82rem; color: #0284C7; font-weight: 600; text-decoration: none; }
    .ref-link:hover { text-decoration: underline; }
    .disclaimer-text { margin-top: 8px; font-size: 0.72rem; color: #94a3b8; }

    .stButton>button { 
        background: linear-gradient(90deg, #0284C7, #0369a1) !important; 
        color: #ffffff !important; font-weight: 600 !important; border-radius: 8px !important; 
        border: none !important; padding: 0.5rem 1rem !important; 
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
    }
    .stButton>button:hover { opacity: 0.92; }
    
    .stFileUploader { background: #ffffff !important; border-radius: 12px !important; }
    div[data-testid="stFileUploader"] { background: #ffffff; border-radius: 12px; padding: 8px; }
    header[data-testid="stHeader"] { background: transparent !important; }

    @media screen and (max-width: 768px) {
        .main-title { font-size: 1.6rem !important; }
        .subtitle { font-size: 0.85rem !important; }
        .defect-name-1 { font-size: 1.2rem !important; }
        .defect-name-2 { font-size: 1rem !important; }
        .block-container { padding-top: 0.5rem !important; max-width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "hybrid_models.pkl")

def load_hybrid_engine():
    with open(MODEL_PATH, "rb") as f: return pickle.load(f)

try:
    engine = load_hybrid_engine()
    models_ok = True
except FileNotFoundError:
    models_ok = False

PAPER_KNOWLEDGE_BASE = {
    "Mobile Ion Contamination (Hysteresis)": {
        "title": "Mobile Ion Contamination (Hysteresis)",
        "desc": "알칼리 가동성 이온(Na⁺, K⁺)이 산화막 전계에 따라 유동적으로 이동하여 플랫밴드 전압(Vfb) 히스테리시스를 유발합니다. 고온 고전압 스트레스 환경에서 기생 누설 전류를 증폭시키는 핵심 요인입니다.",
        "papers": [
            {"title": "Effects of mobile ion contamination on the electrical characteristics of MOS devices", "journal": "IEEE Transactions on Electron Devices", "link": "https://scholar.google.com/scholar?q=MOSCAP+mobile+ion+hysteresis", "summary": "알칼리 가동성 이온(Na+, K+)이 산화막 전계에 따라 유동적으로 이동하여 플랫밴드 전압(Vfb) 히스테리시스를 유발함. 해결을 위해 전공정 세정 단계 최적화 및 HCl 게터링 열처리 기술 적용이 요구됨."},
            {"title": "Mobile ion drift in MOS structures during high-temperature stress", "journal": "Journal of Applied Physics", "link": "https://scholar.google.com/scholar?q=mobile+ion+drift+MOS+high+temperature", "summary": "고온 고전압 스트레스 환경 하에서 소자의 기생 누설 전류를 증폭시키는 핵심 요인. PDA 공정 온도 최적화 및 프리클리닝 이온 세정 강화가 필요함."}
        ]
    },
    "Interface Trap Density Increase (Stretch-out)": {
        "title": "Interface Trap Density Increase (Stretch-out)",
        "desc": "실리콘 기판과 게이트 절연막 간 계면 불포화 결합(Dangling bonds)에 의해 트랩 전하 밀도(Dit)가 상승합니다. 게이트 바이어스 변화 시 트랩 전하의 충방전 지연으로 C-V 커브의 기울기가 완만해지는 Stretch-out 현상이 나타납니다.",
        "papers": [
            {"title": "Characterization of interface traps in MOS structures by capacitance-voltage methods", "journal": "Solid-State Electronics", "link": "https://scholar.google.com/scholar?q=MOSCAP+interface+trap+stretch+out", "summary": "실리콘 기판과 게이트 절연막 간 계면 불포화 결합에 의해 트랩 전하 밀도(Dit)가 상승함. 인가 게이트 바이어스 변화 시 트랩 전하의 충방전 지연으로 C-V 커브 기울기가 완만해짐."},
            {"title": "Hydrogen passivation of interface traps in MOS devices", "journal": "Applied Physics Letters", "link": "https://scholar.google.com/scholar?q=hydrogen+passivation+interface+traps+MOS", "summary": "포밍 가스 어닐링(FGA) 공정을 이용한 계면 수소 패시베이션 처리가 Dit 저감에 필수적임. 수소 분위기 열처리로 댕글링 본드를 효과적으로 제거 가능함."}
        ]
    },
    "Oxide Thickness Out-of-Spec (Cmax Decrease)": {
        "title": "Oxide Thickness Out-of-Spec (Cmax Decrease)",
        "desc": "물리적 산화막 증착 두께(Tox)가 관리 공정 한계를 초과하거나 유전율이 감소하면 타겟 Cmax 확보가 불가능합니다. ALD/CVD 장비 내 소스 분사 불균일 혹은 공정 온도 제어성 손실이 핵심 원인입니다.",
        "papers": [
            {"title": "Impact of ultra-thin gate oxide thickness variations on MOS capacitance", "journal": "Journal of Applied Physics", "link": "https://scholar.google.com/scholar?q=MOSCAP+oxide+thickness+variation", "summary": "물리적 산화막 증착 두께가 관리 공정 한계를 초과하면 타겟 Cmax 확보가 불가능함. ALD/CVD 장비 내 소스 분사 불균일 혹은 공정 온도 제어성 손실이 핵심 인자."},
            {"title": "ALD gate oxide uniformity control in advanced CMOS", "journal": "IEEE Electron Device Letters", "link": "https://scholar.google.com/scholar?q=ALD+gate+oxide+uniformity+CMOS", "summary": "박막 증착 속도 프로파일 재검증 및 유전막 증착 전후 직렬 저항 성분 필터링 수반 권장. ALD 공정 균일도 개선이 Cmax 편차 감소에 직결됨."}
        ]
    },
    "Measurement Setup Noise (Noise Spike)": {
        "title": "Measurement Setup Noise (Noise Spike)",
        "desc": "계측 장비 프로브 스테이션 팁 오염으로 인한 고접촉 저항 성분 형성 및 외부 전자기 노이즈가 유입되어 불규칙한 고주파 스파이크가 발생합니다. 전송 선로 간 임피던스 불일치 및 기생 인덕턴스가 고주파 분석 정확도를 저해합니다.",
        "papers": [
            {"title": "Noise reduction techniques in high-frequency C-V characterization", "journal": "Review of Scientific Instruments", "link": "https://scholar.google.com/scholar?q=C-V+measurement+noise+reduction", "summary": "계측 장비 프로브 팁 오염으로 인한 고접촉 저항 성분 형성 및 외부 전자기 노이즈 유입으로 고주파 스파이크 생성. Open/Short/Load 정밀 캘리브레이션 재실행 필요."},
            {"title": "Probe contact resistance effects in semiconductor C-V measurements", "journal": "Measurement Science and Technology", "link": "https://scholar.google.com/scholar?q=probe+contact+resistance+CV+measurement", "summary": "전송 선로 간 임피던스 불일치 및 미세 기생 인덕턴스가 고주파 분석 정확도를 저해함. 동축 차폐회로(Shielding) 강화 조치 필요."}
        ]
    }
}

FEATURE_INFO = {
    "Cmax": "최대 정전용량 (Maximum Capacitance)\n축적(Accumulation) 영역에서의 최대 정전용량값. 산화막 두께와 반비례.",
    "Cmin": "최소 정전용량 (Minimum Capacitance)\n공핍(Depletion) 영역에서의 최소 정전용량값.",
    "Ratio": "Cmax/Cmin 비율\n정상 커브 판별 기준값. 너무 낮으면 산화막 이상 의심.",
    "Slope Max": "최대 기울기 (Maximum Slope)\n커브에서 가장 급격하게 변하는 전이 영역의 기울기.",
    "Slope Std": "기울기 표준편차 (Slope Standard Deviation)\n커브가 얼마나 고르게 변하는지 나타내는 균일성 지표.",
    "Noise Std": "노이즈 표준편차 (Noise Standard Deviation)\n원본 데이터와 LPF 필터 적용 후 차이. 클수록 노이즈 많음.",
    "Hysteresis": "히스테리시스 (Hysteresis)\n양방향 스윕 시 커브 평행이동 정도. 0에 가까울수록 정상."
}

def denoise_cv_data(v, c_raw, cutoff=0.15, order=3):
    c_raw = np.array(c_raw, dtype=float).flatten()
    if len(c_raw) < 15: return c_raw
    try:
        b, a = butter(order, cutoff, btype='low', analog=False)
        c_smooth = filtfilt(b, a, c_raw)
    except Exception:
        c_smooth = c_raw
    return np.clip(c_smooth, 0.01, np.percentile(c_raw, 95))

def extract_features(v, c, c_clean):
    cmax, cmin = np.max(c_clean), np.min(c_clean)
    cmax_cmin_ratio = cmax / (cmin + 1e-15)
    dv = np.gradient(v)
    dv[dv == 0] = 1e-9  
    dc_dv = np.gradient(c_clean) / dv
    slope_max, slope_std, noise_std = np.max(np.abs(dc_dv)), np.std(dc_dv), np.std(c - c_clean)
    half = len(c_clean) // 2
    if half > 0:
        p1, p2 = np.argmax(np.gradient(c_clean[:half])), np.argmax(np.gradient(c_clean[half:]))
        hysteresis_proxy = float(abs(v[p1] - v[half + p2]))
    else: hysteresis_proxy = 0.0
    return np.nan_to_num([float(cmax), float(cmin), float(cmax_cmin_ratio), float(slope_max), float(slope_std), float(noise_std), hysteresis_proxy], nan=0.0, posinf=0.0, neginf=0.0).tolist()

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown('<div class="main-title">🔬 <span>MOS-ZERO</span> Error Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">반도체 소자 C-V 오류 데이터 정밀 분석 및 학술 솔루션 매칭 시스템</div>', unsafe_allow_html=True)

if not models_ok: 
    st.error("모델 파일을 찾을 수 없습니다. hybrid_models.pkl을 생성해 주세요.")
    st.stop()

st.markdown("<div class='section-header'>📥 데이터 업로드</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("C-V 측정 데이터 업로드 (CSV/TXT)", type=["csv", "txt"], label_visibility="collapsed")

if uploaded:
    try:
        df_data = pd.read_csv(uploaded, sep=None, engine="python")
        cols = list(dict.fromkeys(df_data.columns.tolist()))
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: v_col = st.selectbox("Gate Voltage Axis (X)", cols, index=0)
        with c2: c_col = st.selectbox("Capacitance Axis (Y)", cols, index=1 if len(cols)>1 else 0)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("분석 실행", use_container_width=True)
        if run:
            v_raw = df_data[v_col].iloc[:, 0] if isinstance(df_data[v_col], pd.DataFrame) else df_data[v_col]
            c_raw = df_data[c_col].iloc[:, 0] if isinstance(df_data[c_col], pd.DataFrame) else df_data[c_col]
            temp_df = pd.DataFrame({'v': v_raw, 'c': c_raw}).apply(pd.to_numeric, errors='coerce').dropna()
            voltage, capacitance = temp_df['v'].values, temp_df['c'].values
            if len(voltage) > 0 and len(capacitance) > 0:
                capacitance_clean = denoise_cv_data(voltage, capacitance)
                features = extract_features(voltage, capacitance, capacitance_clean)
                x_scaled = engine["scaler"].transform(np.array([features], dtype=float))
                prob_ml = engine["model_ml"].predict_proba(x_scaled)[0]
                prob_dl = engine["model_dl"].predict_proba(x_scaled)[0]
                final_probabilities = (prob_ml + prob_dl) / 2.0
                classes = engine["classes"]
                top2_indices = np.argsort(final_probabilities)[::-1][:2]
                st.session_state.results = {
                    'voltage': voltage,
                    'capacitance': capacitance,
                    'capacitance_clean': capacitance_clean,
                    'features': features,
                    'top2': [(classes[i], final_probabilities[i]) for i in top2_indices]
                }
                st.session_state.analysis_done = True
    except Exception as e:
        st.error(f"파일 처리 오류: {e}")

st.divider()

if st.session_state.analysis_done and st.session_state.results:
    r = st.session_state.results

    col_graph, col_feat = st.columns([1.3, 1])

    with col_graph:
        st.markdown("<div class='section-header'>📊 신호 분석 그래프</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r['voltage'], y=r['capacitance'], mode="markers", name="원본 데이터", marker=dict(color="#cbd5e1", size=4, opacity=0.5)))
        fig.add_trace(go.Scatter(x=r['voltage'], y=r['capacitance_clean'], mode="lines", name="LPF 필터 적용", line=dict(color="#0284C7", width=3)))
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(title="전압 (V)"),
            yaxis=dict(title="정전용량 (F)"),
            height=320,
            margin=dict(l=40, r=20, t=10, b=30),
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_feat:
        st.markdown("<div class='section-header'>⚙️ 특징값 분석</div>", unsafe_allow_html=True)
        f = r['features']
        feature_data = [
            ("Cmax", round(f[0], 6), "최대 정전용량"),
            ("Cmin", round(f[1], 6), "최소 정전용량"),
            ("Ratio", round(f[2], 6), "Cmax/Cmin 비율"),
            ("Slope Max", round(f[3], 6), "전이 영역 최대 기울기"),
            ("Slope Std", round(f[4], 6), "기울기 표준편차"),
            ("Noise Std", round(f[5], 6), "노이즈 크기"),
            ("Hysteresis", round(f[6], 6), "양방향 스윕 평행이동"),
        ]
        rows = "".join([f"<tr><td><b>{name}</b></td><td style='font-family:Courier New,monospace;'>{val}</td><td style='color:#94a3b8;font-size:0.78rem;'>{desc}</td></tr>" for name, val, desc in feature_data])
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:10px 16px; box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <table class="feature-table">
                <tr><th>항목</th><th>값</th><th>의미</th></tr>
                {rows}
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📋 소자 불량 진단 리포트</div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    rank_labels = ["🥇 1위 최유력 오류 요인", "🥈 2위 복합 가능성 요인"]
    cols = [col_left, col_right]

    for i, (assigned_category, prob) in enumerate(r['top2']):
        with cols[i]:
            info = PAPER_KNOWLEDGE_BASE.get(assigned_category, None)
            desc = info["desc"] if info else ""
            name_class = "defect-name-1" if i == 0 else "defect-name-2"

            st.markdown(f"""
            <div class="report-card">
                <div class="rank-label">{rank_labels[i]}</div>
                <div class="{name_class}">{assigned_category}</div>
                <div class="defect-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            if info and info.get("papers"):
                with st.expander("📚 관련 논문 보기"):
                    for j, paper in enumerate(info["papers"]):
                        with st.expander(f"📄 {paper['title']}"):
                            st.markdown(f"""
                            <div class="ref-card">
                                <div class="ref-meta">{paper['journal']}</div>
                                <div style="font-size:0.82rem; color:#475569; line-height:1.6; margin:8px 0;">{paper['summary']}</div>
                                <a href="{paper['link']}" target="_blank" class="ref-link">원문 보기 ↗</a>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown('<div class="disclaimer-text">※ 본 논문 정보는 AI가 매칭한 것으로 정확하지 않을 수 있습니다.</div>', unsafe_allow_html=True)