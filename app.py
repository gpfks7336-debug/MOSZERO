from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
import io
from scipy.signal import butter, filtfilt

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "hybrid_models.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        engine = pickle.load(f)
    models_ok = True
except FileNotFoundError:
    engine = None
    models_ok = False

PAPER_KNOWLEDGE_BASE = {
    "Mobile Ion Contamination (Hysteresis)": {
        "desc": "알칼리 가동성 이온(Na⁺, K⁺)이 산화막 전계에 따라 유동적으로 이동하여 플랫밴드 전압(Vfb) 히스테리시스를 유발합니다. 고온 고전압 스트레스 환경에서 기생 누설 전류를 증폭시키는 핵심 요인입니다.",
        "papers": [
            {
                "title": "Effects of mobile ion contamination on the electrical characteristics of MOS devices",
                "journal": "IEEE Transactions on Electron Devices",
                "link": "https://scholar.google.com/scholar?q=MOSCAP+mobile+ion+hysteresis",
                "summary": "알칼리 가동성 이온(Na+, K+)이 산화막 전계에 따라 유동적으로 이동하여 플랫밴드 전압(Vfb) 히스테리시스를 유발함. 해결을 위해 전공정 세정 단계 최적화 및 HCl 게터링 열처리 기술 적용이 요구됨."
            },
            {
                "title": "Mobile ion drift in MOS structures during high-temperature stress",
                "journal": "Journal of Applied Physics",
                "link": "https://scholar.google.com/scholar?q=mobile+ion+drift+MOS+high+temperature",
                "summary": "고온 고전압 스트레스 환경 하에서 소자의 기생 누설 전류를 증폭시키는 핵심 요인. PDA 공정 온도 최적화 및 프리클리닝 이온 세정 강화가 필요함."
            }
        ]
    },
    "Interface Trap Density Increase (Stretch-out)": {
        "desc": "실리콘 기판과 게이트 절연막 간 계면 불포화 결합(Dangling bonds)에 의해 트랩 전하 밀도(Dit)가 상승합니다. 게이트 바이어스 변화 시 트랩 전하의 충방전 지연으로 C-V 커브의 기울기가 완만해지는 Stretch-out 현상이 나타납니다.",
        "papers": [
            {
                "title": "Characterization of interface traps in MOS structures by capacitance-voltage methods",
                "journal": "Solid-State Electronics",
                "link": "https://scholar.google.com/scholar?q=MOSCAP+interface+trap+stretch+out",
                "summary": "실리콘 기판과 게이트 절연막 간 계면 불포화 결합에 의해 트랩 전하 밀도(Dit)가 상승함. 인가 게이트 바이어스 변화 시 트랩 전하의 충방전 지연으로 C-V 커브 기울기가 완만해짐."
            },
            {
                "title": "Hydrogen passivation of interface traps in MOS devices",
                "journal": "Applied Physics Letters",
                "link": "https://scholar.google.com/scholar?q=hydrogen+passivation+interface+traps+MOS",
                "summary": "포밍 가스 어닐링(FGA) 공정을 이용한 계면 수소 패시베이션 처리가 Dit 저감에 필수적임. 수소 분위기 열처리로 댕글링 본드를 효과적으로 제거 가능함."
            }
        ]
    },
    "Oxide Thickness Out-of-Spec (Cmax Decrease)": {
        "desc": "물리적 산화막 증착 두께(Tox)가 관리 공정 한계를 초과하거나 유전율이 감소하면 타겟 Cmax 확보가 불가능합니다. ALD/CVD 장비 내 소스 분사 불균일 혹은 공정 온도 제어성 손실이 핵심 원인입니다.",
        "papers": [
            {
                "title": "Impact of ultra-thin gate oxide thickness variations on MOS capacitance",
                "journal": "Journal of Applied Physics",
                "link": "https://scholar.google.com/scholar?q=MOSCAP+oxide+thickness+variation",
                "summary": "물리적 산화막 증착 두께가 관리 공정 한계를 초과하면 타겟 Cmax 확보가 불가능함. ALD/CVD 장비 내 소스 분사 불균일 혹은 공정 온도 제어성 손실이 핵심 인자."
            },
            {
                "title": "ALD gate oxide uniformity control in advanced CMOS",
                "journal": "IEEE Electron Device Letters",
                "link": "https://scholar.google.com/scholar?q=ALD+gate+oxide+uniformity+CMOS",
                "summary": "박막 증착 속도 프로파일 재검증 및 유전막 증착 전후 직렬 저항 성분 필터링 수반 권장. ALD 공정 균일도 개선이 Cmax 편차 감소에 직결됨."
            }
        ]
    },
    "Measurement Setup Noise (Noise Spike)": {
        "desc": "계측 장비 프로브 스테이션 팁 오염으로 인한 고접촉 저항 성분 형성 및 외부 전자기 노이즈가 유입되어 불규칙한 고주파 스파이크가 발생합니다. 전송 선로 간 임피던스 불일치 및 기생 인덕턴스가 고주파 분석 정확도를 저해합니다.",
        "papers": [
            {
                "title": "Noise reduction techniques in high-frequency C-V characterization",
                "journal": "Review of Scientific Instruments",
                "link": "https://scholar.google.com/scholar?q=C-V+measurement+noise+reduction",
                "summary": "계측 장비 프로브 팁 오염으로 인한 고접촉 저항 성분 형성 및 외부 전자기 노이즈 유입으로 고주파 스파이크 생성. Open/Short/Load 정밀 캘리브레이션 재실행 필요."
            },
            {
                "title": "Probe contact resistance effects in semiconductor C-V measurements",
                "journal": "Measurement Science and Technology",
                "link": "https://scholar.google.com/scholar?q=probe+contact+resistance+CV+measurement",
                "summary": "전송 선로 간 임피던스 불일치 및 미세 기생 인덕턴스가 고주파 분석 정확도를 저해함. 동축 차폐회로(Shielding) 강화 조치 필요."
            }
        ]
    }
}


def denoise_cv_data(v, c_raw, cutoff=0.15, order=3):
    c_raw = np.array(c_raw, dtype=float).flatten()
    if len(c_raw) < 15:
        return c_raw
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
    slope_max = np.max(np.abs(dc_dv))
    slope_std = np.std(dc_dv)
    noise_std = np.std(c - c_clean)
    half = len(c_clean) // 2
    if half > 0:
        p1 = np.argmax(np.gradient(c_clean[:half]))
        p2 = np.argmax(np.gradient(c_clean[half:]))
        hysteresis_proxy = float(abs(v[p1] - v[half + p2]))
    else:
        hysteresis_proxy = 0.0
    return np.nan_to_num(
        [float(cmax), float(cmin), float(cmax_cmin_ratio), float(slope_max),
         float(slope_std), float(noise_std), hysteresis_proxy],
        nan=0.0, posinf=0.0, neginf=0.0
    ).tolist()


def decode_file(raw):
    for enc in ('utf-8', 'cp949', 'euc-kr', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


@app.route('/')
def index():
    return render_template('index.html', models_ok=models_ok)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': '파일이 없습니다.'}), 400
    try:
        content = decode_file(file.read())
        df = pd.read_csv(io.StringIO(content), sep=None, engine='python')
        cols = list(dict.fromkeys(df.columns.tolist()))
        return jsonify({'columns': cols})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    if not models_ok:
        return jsonify({'error': '모델 파일을 찾을 수 없습니다.'}), 500
    file = request.files.get('file')
    v_col = request.form.get('v_col')
    c_col = request.form.get('c_col')
    if not file or not v_col or not c_col:
        return jsonify({'error': '파일 또는 컬럼 정보가 없습니다.'}), 400
    try:
        content = decode_file(file.read())
        df = pd.read_csv(io.StringIO(content), sep=None, engine='python')

        v_series = df[v_col]
        c_series = df[c_col]
        if isinstance(v_series, pd.DataFrame):
            v_series = v_series.iloc[:, 0]
        if isinstance(c_series, pd.DataFrame):
            c_series = c_series.iloc[:, 0]

        temp_df = pd.DataFrame({'v': v_series, 'c': c_series}).apply(pd.to_numeric, errors='coerce').dropna()
        voltage = temp_df['v'].values
        capacitance = temp_df['c'].values

        if len(voltage) == 0:
            return jsonify({'error': '유효한 데이터가 없습니다.'}), 400

        capacitance_clean = denoise_cv_data(voltage, capacitance)
        features = extract_features(voltage, capacitance, capacitance_clean)

        x_scaled = engine["scaler"].transform(np.array([features], dtype=float))
        prob_ml = engine["model_ml"].predict_proba(x_scaled)[0]
        prob_dl = engine["model_dl"].predict_proba(x_scaled)[0]
        final_probabilities = (prob_ml + prob_dl) / 2.0
        classes = engine["classes"]
        top2_indices = np.argsort(final_probabilities)[::-1][:2]
        top2_raw = [(classes[i], float(final_probabilities[i])) for i in top2_indices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=voltage.tolist(), y=capacitance.tolist(),
            mode="markers", name="원본 데이터",
            marker=dict(color="#e5e5e5", size=4, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=voltage.tolist(), y=capacitance_clean.tolist(),
            mode="lines", name="LPF 필터 적용",
            line=dict(color="#2563eb", width=2.5)
        ))
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(title="전압 (V)", showgrid=True, gridcolor="#f5f5f4"),
            yaxis=dict(title="정전용량 (F)", showgrid=True, gridcolor="#f5f5f4"),
            height=300,
            margin=dict(l=40, r=20, t=10, b=30),
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff"
        )

        feature_names = ["Cmax", "Cmin", "Ratio", "Slope Max", "Slope Std", "Noise Std", "Hysteresis"]
        feature_tips = [
            "최대 정전용량. 축적 영역에서의 최대값.",
            "최소 정전용량. 공핍 영역에서의 최소값.",
            "Cmax/Cmin 비율. 낮으면 산화막 이상 의심.",
            "전이 영역 최대 기울기.",
            "기울기 표준편차. 커브 균일성 지표.",
            "노이즈 크기. 클수록 노이즈 많음.",
            "양방향 스윕 평행이동. 0에 가까울수록 정상."
        ]
        feature_data = [
            {"name": feature_names[i], "value": round(features[i], 6), "tip": feature_tips[i]}
            for i in range(len(features))
        ]

        top2_info = []
        for cat, prob in top2_raw:
            info = PAPER_KNOWLEDGE_BASE.get(cat, {})
            top2_info.append({
                "category": cat,
                "prob": round(prob * 100, 1),
                "desc": info.get("desc", ""),
                "papers": info.get("papers", [])
            })

        return jsonify({
            "chart": fig.to_json(),
            "features": feature_data,
            "top2": top2_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
