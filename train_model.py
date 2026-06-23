import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs("output", exist_ok=True)
print("🚀 [나노 스케일 최적화] 촘촘한 나노(nF) 단위 기반 하이브리드 엔진 학습 시작...\n")

def generate_hybrid_data():
    X_list, y_multi = [], []
    
    # [오류 1: hysteresis (이온 오염)] 150개
    for _ in range(150):
        X_list.append([
            np.random.uniform(3.0e-9, 5.0e-9),   # 1. Cmax (3~5 nF)
            np.random.uniform(0.5e-9, 1.2e-9),  # 2. Cmin (0.5~1.2 nF)
            np.random.uniform(3.0, 5.0),         # 3. Ratio
            np.random.uniform(2.0e-9, 5.0e-9),   # 4. Slope_max
            np.random.uniform(5.0e-10, 1.5e-9),  # 5. Slope_std
            np.random.uniform(0, 5.0e-12),       # 6. Noise (매우 깨끗함)
            np.random.uniform(0.5, 2.0)          # 7. Hysteresis 전압 전이(V)
        ])
        y_multi.append("Mobile Ion Contamination (Hysteresis)")
        
    # [오류 2: stretch_out (찌그러짐)] 150개
    for _ in range(150):
        X_list.append([
            np.random.uniform(3.0e-9, 5.0e-9), np.random.uniform(0.5e-9, 1.2e-9), 
            np.random.uniform(1.5, 2.8),         # Ratio 감소
            np.random.uniform(2.0e-10, 9.0e-10), # 💡 기울기가 나노 이하로 푹 찌그러짐
            np.random.uniform(5.0e-11, 3.0e-10), 
            np.random.uniform(0, 5.0e-12), np.random.uniform(0, 0.1)
        ])
        y_multi.append("Interface Trap Density Increase (Stretch-out)")
        
    # [오류 3: cmax_decrease (두께 불량)] 150개
    for _ in range(150):
        X_list.append([
            np.random.uniform(0.8e-9, 1.8e-9),   # 💡 Cmax가 1nF 대로 뚝 떨어짐 (박막 두께 이상)
            np.random.uniform(0.3e-9, 0.7e-9), 
            np.random.uniform(1.5, 2.5), np.random.uniform(2.0e-9, 5.0e-9), np.random.uniform(5.0e-10, 1.5e-9), 
            np.random.uniform(0, 5.0e-12), np.random.uniform(0, 0.1)
        ])
        y_multi.append("Oxide Thickness Out-of-Spec (Cmax Decrease)")
        
    # [오류 4: noise_spike (장비 노이즈)] 150개
    for _ in range(150):
        X_list.append([
            np.random.uniform(3.0e-9, 5.0e-9), np.random.uniform(0.5e-9, 1.2e-9), 
            np.random.uniform(3.0, 5.0), np.random.uniform(2.0e-9, 5.0e-9), np.random.uniform(5.0e-10, 1.5e-9), 
            np.random.uniform(3.0e-10, 1.2e-9), # 💡 노이즈 성분이 나노 수준으로 폭발함
            np.random.uniform(0, 0.1)
        ])
        y_multi.append("Measurement Setup Noise (Noise Spike)")
        
    return np.array(X_list), np.array(y_multi)

X, y = generate_hybrid_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("▶️ [머신러닝 레이어] SVC 학습 중...")
model_ml = SVC(probability=True, class_weight='balanced', random_state=42)
model_ml.fit(X_scaled, y_encoded)

print("▶️ [딥러닝 레이어] 다층 인공신경망(MLP) 훈련 중...")
model_dl = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42, early_stopping=True)
model_dl.fit(X_scaled, y_encoded)

with open("output/hybrid_models.pkl", "wb") as f:
    pickle.dump({
        "model_ml": model_ml, "model_dl": model_dl, "scaler": scaler, "classes": label_encoder.classes_
    }, f)

print("\n🎉 완료! 촘촘한 나노(nF) 스케일에 최적화된 하이브리드 오류 분석 뇌가 장착되었습니다!")