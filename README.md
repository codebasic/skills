# Codebasic Skills (`codebasic-skills`)

딥러닝 및 시계열 분석 실무를 위한 고품질 에이전트 스킬(Agent Skills) 모음 저장소입니다.  
Anthropic의 [Agent Skills 오픈 규격](https://agentskills.io) 및 Claude Code, Google Antigravity 프레임워크와 완벽하게 호환됩니다.

---

## 📦 스킬 카탈로그 (Skill Catalog)

| 스킬명 | 설명 | 주요 기술 스택 | 링크 |
| :--- | :--- | :--- | :--- |
| **`keras`** | Keras 3를 PyTorch 백엔드로 활용하는 워크스페이스 컨벤션, 모델 구축, DataLoader 연동 및 MNIST 실습 | Keras 3, PyTorch Backend, torchvision, DataLoader | [keras/SKILL.md](./keras/SKILL.md) |
| **`pydeep`** | 실습 시스템 내 conda 가상환경(`base`, `pyml`, `pytorch`, `tensorflow`) 분기 및 환경 진단 가이드 | Conda, PyTorch (CUDA/MPS), TensorFlow | [pydeep/SKILL.md](./pydeep/SKILL.md) |
| **`timeseries`** | 시계열 회귀·분류·이상탐지 전 과정(전처리, 윈도잉, 베이스라인, 1D CNN/TCN/트랜스포머, TimesFM/TTM/MOMENT 파운데이션 모델, LM 직렬화, XAI) | Pandas, Keras 3, TimesFM, TTM, MOMENT, Transformers | [timeseries/SKILL.md](./timeseries/SKILL.md) |

---

## 🚀 에이전트 환경 연동 가이드

이 저장소의 스킬들은 개별 디렉터리 단위로 독립 동작하도록 설계되어 있습니다. 사용 중인 AI 코딩 에이전트에 맞게 연결하여 즉시 사용할 수 있습니다.

### 1. Claude Code (`.claude/skills`)

Claude Code 환경에서는 사용자 홈 디렉터리(`~/.claude/skills`) 또는 프로젝트 루트의 `.claude/skills`에 심볼릭 링크를 생성합니다:

```bash
# 전체 스킬을 Claude Code 글로벌 스킬로 등록
mkdir -p ~/.claude/skills
ln -s "$(pwd)/keras" ~/.claude/skills/keras
ln -s "$(pwd)/pydeep" ~/.claude/skills/pydeep
ln -s "$(pwd)/timeseries" ~/.claude/skills/timeseries

# 또는 특정 프로젝트에만 등록
cd /path/to/my-project
mkdir -p .claude/skills
ln -s /path/to/codebasic-skills/keras .claude/skills/keras
```

### 2. Google Antigravity (`.agents/skills`)

Antigravity 환경에서는 워크스페이스 루트의 `.agents/skills`에 배치하거나 글로벌 설정에 마운트합니다:

```bash
# 워크스페이스 로컬 스킬로 등록
cd /path/to/my-project
mkdir -p .agents/skills
ln -s /path/to/codebasic-skills/keras .agents/skills/keras
ln -s /path/to/codebasic-skills/timeseries .agents/skills/timeseries

# 방법 A: ~/.gemini/config/skills.json에 ~/.agents/skills 경로 등록 (권장)
# (이미 ~/.agents/skills에 심볼릭 링크되어 있는 경우 자동으로 모든 스킬 탐색)
cat << 'EOF' > ~/.gemini/config/skills.json
{
  "entries": [
    {
      "path": "~/.agents/skills"
    }
  ]
}
EOF

# 방법 B: 개별 스킬을 직접 Antigravity 글로벌 스킬로 링크
mkdir -p ~/.gemini/config/skills
ln -s "$(pwd)/keras" ~/.gemini/config/skills/keras
```

---

## 📂 디렉터리 구조 및 표준 규격

모든 스킬은 점진적 정보 노출(Progressive Disclosure) 원칙에 따라 구성되어 컨텍스트 토큰을 최소화합니다:

```text
codebasic-skills/
├── README.md               # 저장소 개요 및 연동 가이드
├── .gitignore              # 파이썬 캐시, 체크포인트, OS 메타데이터 제외
├── keras/
│   ├── SKILL.md            # [핵심] Keras 3 워크스페이스 컨벤션 및 퀵 가이드
│   ├── references/         # [참조] 설치, 세부 모델 구축, 훈련 가이드
│   └── scripts/            # [스크립트] MNIST 참조 구현 스크립트
├── pydeep/
│   ├── SKILL.md            # [핵심] Conda 환경 전환 및 진단 절차
│   └── references/         # [참조] 환경별 세부 패키지 명세
└── timeseries/
    ├── SKILL.md            # [핵심] 시계열 모델링 요약, 벤치마크, 의사결정 매트릭스
    ├── references/         # [참조] 전처리, 시퀀스 생성, 모델별 실측 교훈, 파운데이션 모델
    └── scripts/            # [스크립트] 기후 예측 전체 파이프라인 참조 구현
```

---

## 🧪 검증 및 평가 (Evals)

각 스킬의 `SKILL.md` 하단에는 **Evals (발동 검증 질문)**이 포함되어 있습니다.  
에이전트가 어떤 맥락에서 해당 스킬을 활성화해야 하는지(Positive Trigger)와 활성화하지 않아야 하는지(Negative Trigger)를 명확히 정의하여 의도치 않은 간섭 없이 높은 정확도로 실행됩니다.
