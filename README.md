# MOS-ZERO
> 반도체 C-V 커브 불량 원인 자동 분류 시스템

MOS 커패시터 C-V 측정 데이터를 업로드하면 SVC + MLP 하이브리드 AI 엔진이 불량 원인을 자동으로 분류하고, 관련 논문을 제안합니다.

## 주요 기능
- CSV / TXT 파일 업로드 (인코딩 자동 감지)
- Butterworth LPF 기반 노이즈 제거
- 7가지 특징값 자동 추출 (Cmax, Cmin, Ratio, Slope Max, Slope Std, Noise Std, Hysteresis)
- SVC + MLP 앙상블 모델로 4가지 불량 유형 분류
- 불량 유형별 관련 논문 자동 제안

## 불량 유형
| 유형 | 특징 |
|------|------|
| Mobile Ion Contamination | Hysteresis 전압 증가 |
| Interface Trap Density Increase | Slope Max 감소 |
| Oxide Thickness Out-of-Spec | Cmax 급감 |
| Measurement Setup Noise | Noise Std 급증 |

## 기술 스택
- **Backend**: Python, Flask
- **ML**: scikit-learn (SVC, MLP)
- **시각화**: Plotly
- **배포**: Render.com

## 데모
(https://moszero.onrender.com/)

## 팀원
김다정 · 김예빈 · 심혜란 · 오한솔 · 이정민
