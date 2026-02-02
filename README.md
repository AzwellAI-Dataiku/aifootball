# AIFootball - AI 기반 축구 공 추적 시스템

이 프로젝트는 YOLO 모델을 활용하여 축구 경기 영상에서 공과 사람을 추적하는 AI 시스템입니다.

## 디렉토리 구조

```
aifootball/
├── ball_tracking_model.py          # 공 추적 메인 스크립트
├── README.md                       # 프로젝트 설명 파일
├── inputs/s
│   ├── models/                     # 학습된 모델 파일들
│   │   ├── best_ball.pt            # 공 전용 YOLO 모델
│   │   └── best_person.pt          # 사람 전용 YOLO 모델
│   └── videos/                     # 입력 비디오 파일들
└── tracking_results/
    └── ball_tracking/              # 추적 결과 출력 디렉토리
        └── {run_id}/               # 실행별 결과 폴더 (날짜_시간 형식)
            ├── ball_track.csv      # 공 추적 데이터 (CSV)
            ├── run_meta.json       # 실행 메타데이터 (JSON)
            └── (ball_overlay.mp4)  # 옵션: 공 추적 오버레이 비디오
```

## 입력 (Input)

### 1. Videos
- **경로**: `inputs/videos/`
- **형식**: MP4 형식의 축구 경기 영상
- **용도**: 추적할 대상 비디오 파일

### 2. Models
- **경로**: `inputs/models/`
- **모델**: YOLOv8s를 fine-tuning한 전용 모델 2개
  - `best_ball.pt`: 공 추적용 모델
  - `best_person.pt`: 사람 추적용 모델
- **용도**: 사전 학습된 AI 모델 파일

## 출력 (Output)

### Tracking Results
- **경로**: `tracking_results/`
- **형식**:
  - **CSV**: 공 추적 좌표 및 메타데이터
  - **JSON**: 실행 정보 및 설정
  - **MP4**: 옵션으로 생성되는 추적 오버레이 비디오
- **내용**: 각 모델의 추론 결과를 영상 분석을 통해 출력

## ball_tracking_model.py

공 추적을 위한 메인 Python 스크립트입니다.

### 기능
- YOLO 모델을 사용하여 비디오에서 공을 실시간 추적
- 카메라 모션 보정 및 광학 흐름 기반 후보 추가
- 글로벌 최적화 알고리즘으로 정확한 추적 경로 생성
- RTS 칼만 스무딩으로 후처리

### 사용법
1. `inputs/videos/`에 MP4 비디오 파일을 업로드
2. `inputs/models/`에 `best_ball.pt` 모델 파일이 있는지 확인
3. 스크립트 실행: `python ball_tracking_model.py`
4. 결과는 `tracking_results/ball_tracking/{run_id}/`에 저장

### 주요 파라미터
- `CONF_THRES`: 탐지 신뢰도 임계값 (기본: 0.02)
- `MAX_GAP`: 최대 추적 간격 (기본: 8 프레임)
- `SAVE_OVERLAY`: 오버레이 비디오 생성 여부 (기본: True)

## 요구사항

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy, Pandas

## 설치 및 실행

```bash
# 의존성 설치
pip install ultralytics opencv-python numpy pandas

# 실행
python ball_tracking_model.py
```