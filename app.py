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

    .main-title { 
        font-size: 2.4rem; font-weight: 800; color: #0f172a; letter-spacing: -1.5px; margin-bottom: 4px; text-align: center;
    }
    .main-title span { color: #0284C7; }
    .subtitle { font-size: 1rem; color: #64748b; font-weight: 400; margin-top: 4px; padding-bottom: 10px; text-align: center; }
    
    .section-header { 
        font-size: 1rem; font-weight: 600; color: #334155; margin-top: 1rem; margin-bottom: 0.6rem; 
        border-left: 3px solid #0284C7; padding-left: 10px;
    }
    
    .debug-card { 
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; 
        padding: 10px 16px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .feature-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .feature-table th { background: #f1f5f9; color: #334155; font-weight: 600; padding: 6px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }
    .feature-table td { padding: 5px 10px; color: #475569; border-bottom: 1px solid #f1f5f9; font-family: 'Courier New', monospace; }
    .feature-table tr:last-child td { border-bottom: none; }

    .dashboard-card { 
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; 
        padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .card-rank { font-size: 0.82rem; font-weight: 700; color: #0284C7; margin-bottom: 2px; letter-spacing: 0.3px; }
    .card-title { font-size: 1rem; font-weight: 700; margin-bottom: 2px; color: #0f172a; }
    
    .paper-card { 
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; 
        padding: 14px 16px; margin-top: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .paper-tag { 
        font-size: 0.72rem; font-weight: 700; color: #0284C7; 
        background: #e0f2fe; padding: 2px 8px; border-radius: 20px; 
        display: inline-block; margin-bottom: 8px; 
    }
    .paper-title { font-size: 0.92rem; font-weight: 700; color: #0f172a; margin-bottom: 4px; line-height: 1.4; }
    .paper-journal { font-size: 0.78rem; color: #94a3b8; font-style: italic; margin-bottom: 10px; }
    .paper-summary { font-size: 0.82rem; color: #475569; line-height: 1.6; }
    .summary-point { display: flex; align-items: start; margin-bottom: 6px; }
    .summary-icon { margin-right: 8px; color: #0284C7; font-weight: bold; font-size: 0.85rem; margin-top: 1px; }
    .paper-action-link { 
        display: inline-block; margin-top: 10px; color: #0284C7; text-decoration: none; 
        font-size: 0.82rem; font-weight: 600; transition: color 0.2s; 
    }
    .paper-action-link:hover { color: #0369a1; text-decoration: underline; }

    .disclaimer-text { 
        margin-top: 6px; margin-bottom: 10px; font-size: 0.72rem; 
        color: #94a3b8; font-weight: 400; letter-spacing: -0.2px; 
    }

    .stButton>button { 
        background: linear-gradient(90deg, #0284C7, #0369a1) !important; 
        color: #ffffff !important; font-weight: 600 !important; border-radius: 8px !important; 
        border: none !important; padding: 0.5rem 1.5rem !important; 
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
    }
    .stButton>button:hover { opacity: 0.92; }
    
    .stFileUploader { background: #ffffff !important; border-radius: 12px !important; }
    div[data-testid="stFileUploader"] { background: #ffffff; border-radius: 12px; padding: 8px; }
    header[data-testid="stHeader"] { background: transparent !important; }
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
        "title": "Effects of mobile ion contamination on the electrical characteristics of MOS devices",
        "journal": "IEEE Transactions on Electron Devices",
        "link": "https://scholar.google.com/scholar?q=MOSCAP+mobile+ion+hysteresis",
        "summaries": ["알칼리 가동성 이온(Na+, K+)이 산화막 전계에 따라 유동적으로 이동하여 플랫밴드 전압(Vfb) 히스테리시스를 유발함.", "고온 고전압 스트레스 환경 하에서 소자의 기생 누설 전류를 증폭시키는 핵심 요인.", "해결을 위해 전공정 세정(Pre-cleaning) 단계 최적화 및 HCl 게터링 열처리 기술 적용이 요구됨."]
    },
    "Interface Trap Density Increase (Stretch-out)": {
        "title": "Characterization of interface traps in MOS structures by capacitance-voltage methods",
        "journal": "Solid-State Electronics",
        "link": "https://scholar.google.com/scholar?q=MOSCAP+interface+trap+stretch+out",
        "summaries": ["실리콘 기판과 게이트 절연막 간 계면 불포화 결합(Dangling bonds)에 의해 트랩 전하 밀도(Dit)가 상승함.", "인가 게이트 바이어스 변화 시 트랩 전하의 충방전 지연으로 인하여 전체 C-V 커브의 기울기를 납작하게(Stretch-out) 늘어뜨림.", "포밍 가스 어닐링(Forming Gas Annealing, FGA) 공정을 이용한 계면 수소 패시베이션 처리가 필수적임."]
    },
    "Oxide Thickness Out-of-Spec (Cmax Decrease)": {
        "title": "Impact of ultra-thin gate oxide thickness variations on MOS capacitance",
        "journal": "Journal of Applied Physics",
        "link": "https://scholar.google.com/scholar?q=MOSCAP+oxide+thickness+variation",
        "summaries": ["물리적 산화막 증착 두께(Tox)가 관리 공정 한계를 초과하여 두꺼워지거나 유전율이 감소하면 타겟 Cmax 확보가 불가능함.", "원인으로 ALD/CVD 장비 내 소스 분사 불균일 혹은 공정 온도 제어성 손실이 핵심 인자로 파악됨.", "박막 증착 속도 프로파일 재검증 및 유전막 증착 전후 직렬 저항(Series Resistance) 성분 필터링 수반 권장."]
    },
    "Measurement Setup Noise (Noise Spike)": {
        "title": "Noise reduction techniques in high-frequency C-V characterization",
        "journal": "Review of Scientific Instruments",
        "link": "https://scholar.google.com/scholar?q=C-V+measurement+noise+reduction",
        "summaries": ["계측 장비 프로브 스테이션 팁 오염으로 인한 고접촉 저항 성분 형성 및 외부 전자기 노이즈가 유입되어 불규칙한 고주파 스파이크 생성.", "전송 선로 간 임피던스 불일치 및 미세 기생 인덕턴스가 고주파 분석 정확도를 전반적으로 저해함.", "장비 본체의 Open/Short/Load 정밀 캘리브레이션을 재실행하고 내부 도선 동축 차폐회로(Shielding) 강화 조치 필요."]
    }
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

# ── Session State ──
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None

# ── Main UI ──
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
        c1, c2, _ = st.columns([2, 2, 1])
        with c1: v_col = st.selectbox("Gate Voltage Axis (X)", cols, index=0)
        with c2: c_col = st.selectbox("Capacitance Axis (Y)", cols, index=1 if len(cols)>1 else 0)
        if st.button("분석 실행", use_container_width=True):
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

# ── 결과 표시 ──
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
        feature_names = ["Cmax", "Cmin", "Ratio", "Slope Max", "Slope Std", "Noise Std", "Hysteresis"]
        rows = "".join([f"<tr><td><b>{feature_names[i]}</b></td><td>{round(f[i], 6)}</td></tr>" for i in range(len(f))])
        st.markdown(f"""
        <div class="debug-card">
            <table class="feature-table">
                <tr><th>항목</th><th>값</th></tr>
                {rows}
            </table>
        </div>
        """, unsafe_allow_html=True)

    # 1위 2위 나란히
    st.markdown("<div class='section-header'>📋 진단 리포트</div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    rank_labels = ["🥇 1위 최유력 오류 요인", "🥈 2위 복합 가능성 요인"]
    cols = [col_left, col_right]

    for i, (assigned_category, prob) in enumerate(r['top2']):
        with cols[i]:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-rank">{rank_labels[i]}</div>
                <div class="card-title">{assigned_category}</div>
            </div>
            """, unsafe_allow_html=True)

            if assigned_category in PAPER_KNOWLEDGE_BASE:
                paper = PAPER_KNOWLEDGE_BASE[assigned_category]
                summary_html = "".join([f"<div class='summary-point'><span class='summary-icon'>•</span><div>{s}</div></div>" for s in paper["summaries"]])

                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-tag">{i+1}위 요인 연계 SCI Journal Insights</div>
                    <div class="paper-title">{paper['title']}</div>
                    <div class="paper-journal">{paper['journal']}</div>
                    <div class="paper-summary">{summary_html}</div>
                    <div><a href="{paper['link']}" target="_blank" class="paper-action-link">Google Scholar에서 원문 확인하기 →</a></div>
                </div>
                <div class="disclaimer-text">※ 본 논문의 요약은 AI가 수행한 것으로 정확하지 않을 수도 있습니다.</div>
                """, unsafe_allow_html=True)