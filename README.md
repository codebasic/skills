# Codebasic Skills (`codebasic-skills`)

AI 모델 동향 조사, 딥러닝 및 시계열 분석 실무를 위한 고품질 에이전트 스킬(Agent Skills) 모음 저장소입니다.
Anthropic의 [Agent Skills 오픈 규격](https://agentskills.io) 및 Claude Code, Google Antigravity 프레임워크와 완벽하게 호환됩니다.

> 💖 **Codebasic Skills는 무료·오픈소스입니다.** 유용하게 쓰고 계신다면 후원으로 개발을 응원해 주세요
> → [**GitHub Sponsors로 후원하기**](https://github.com/sponsors/codebasic) (금액·주기 자유)

---

## 📦 스킬 카탈로그 (Skill Catalog)

| 스킬명 | 설명 | 주요 기술 스택 | 링크 |
| :--- | :--- | :--- | :--- |
| **`ai-report`** | 신모델 출시·성능·가격·라이선스를 조사해 출처를 갖춘 한국어 비교 보고서 작성 | 웹 검색, Artificial Analysis, 공식 발표·가격 문서 | [ai-report/SKILL.md](./ai-report/SKILL.md) |
| **`keras`** | Keras 3를 PyTorch 백엔드로 활용하는 워크스페이스 컨벤션, 모델 구축, DataLoader 연동 및 MNIST 실습 | Keras 3, PyTorch Backend, torchvision, DataLoader | [keras/SKILL.md](./keras/SKILL.md) |
| **`ml-data-preparation`** | 예측 시점·독립 단위에 맞춘 분할과 누수 없는 전처리 | 프레임워크 공통 | [ml-data-preparation/SKILL.md](./ml-data-preparation/SKILL.md) |
| **`ml-model-evaluation`** | 지표·검증 설계·임계값 선택과 공정한 모델 비교 | 프레임워크 공통 | [ml-model-evaluation/SKILL.md](./ml-model-evaluation/SKILL.md) |
| **`ml-training-diagnostics`** | 학습곡선·측정 조건·통제 실험에 근거한 학습 진단 | 프레임워크 공통 | [ml-training-diagnostics/SKILL.md](./ml-training-diagnostics/SKILL.md) |
| **`pydeep`** | 실습 시스템 내 conda 가상환경(`base`, `pyml`, `pytorch`, `tensorflow`) 분기 및 환경 진단 가이드 | Conda, PyTorch (CUDA/MPS), TensorFlow | [pydeep/SKILL.md](./pydeep/SKILL.md) |
| **`timeseries`** | 시계열 회귀·분류·이상탐지 전 과정(전처리, 윈도잉, 베이스라인, 1D CNN/TCN/트랜스포머, TimesFM/TTM/MOMENT 파운데이션 모델, LM 직렬화, XAI) | Pandas, Keras 3, TimesFM, TTM, MOMENT, Transformers | [timeseries/SKILL.md](./timeseries/SKILL.md) |

---

## 🚀 에이전트 환경 연동 가이드 (Agent Skills Sync)

이 저장소의 스킬들은 **Agent Skills 오픈 규격**을 준수하므로, 주요 AI 코딩 에이전트(**Claude Code, Google Antigravity, GitHub Copilot, OpenCode**)에서 공통으로 사용할 수 있습니다.

### ⚡ 원클릭 자동 등록 스크립트 (권장)

시스템에 설치된 에이전트들을 자동으로 감지하고 심볼릭 링크 및 설정을 일괄 구성하는 [scripts/register-skills.sh](./scripts/register-skills.sh) 스크립트를 제공합니다:

```bash
# 1. 시스템에 설치된 모든 에이전트에 스킬 자동 등록 (글로벌)
./scripts/register-skills.sh

# 2. 각 에이전트별 등록 상태 확인
./scripts/register-skills.sh --status

# 3. 특정 프로젝트 워크스페이스에 스킬 동기화
./scripts/register-skills.sh --project /path/to/my-project

# 4. 등록된 심볼릭 링크 제거
./scripts/register-skills.sh --uninstall
```

#### 스크립트가 자동 감지 및 구성하는 에이전트:
* **Agent Skills Hub**: `~/.agents/skills` (오픈 표준 공통 허브)
* **Claude Code**: `~/.claude/skills`
* **Google Antigravity (`agy`)**: `~/.gemini/config/skills.json` (전역 스캔 등록) 및 `~/.gemini/config/skills`
* **GitHub Copilot**: `~/.copilot/skills`
* **OpenCode**: `~/.config/opencode/skills`

---

### 🛠️ 에이전트별 표준 스킬 경로 (수동 등록 시)

| 에이전트 | 글로벌 등록 위치 | 프로젝트 로컬 등록 위치 |
| :--- | :--- | :--- |
| **Claude Code** | `~/.claude/skills/<skill>` | `<project>/.claude/skills/<skill>` |
| **Google Antigravity** | `~/.gemini/config/skills.json` 등록 | `<project>/.agents/skills/<skill>` |
| **GitHub Copilot** | `~/.copilot/skills/<skill>` | `<project>/.github/skills/<skill>` |
| **OpenCode** | `~/.config/opencode/skills/<skill>` | `<project>/.opencode/skills/<skill>` |

---

## 📂 디렉터리 구조 및 표준 규격

모든 스킬은 점진적 정보 노출(Progressive Disclosure) 원칙에 따라 구성되어 컨텍스트 토큰을 최소화합니다:

```text
codebasic-skills/
├── README.md               # 저장소 개요 및 연동 가이드
├── .gitignore              # 파이썬 캐시, 체크포인트, OS 메타데이터 제외
├── scripts/
│   └── register-skills.sh  # [자동화] 시스템 에이전트 자동 탐색 및 스킬 동기화 스크립트
├── ai-report/
│   └── SKILL.md            # AI 모델 동향 조사 및 비교 보고서
├── keras/
│   ├── SKILL.md            # [핵심] Keras 3 워크스페이스 컨벤션 및 퀵 가이드
│   ├── references/         # [참조] 설치, 세부 모델 구축, 훈련 가이드
│   └── scripts/            # [스크립트] MNIST 참조 구현 스크립트
├── ml-data-preparation/
│   └── SKILL.md            # 분할과 누수 없는 전처리
├── ml-model-evaluation/
│   └── SKILL.md            # 지표·검증·시험과 모델 비교
├── ml-training-diagnostics/
│   └── SKILL.md            # 학습곡선과 통제 실험 기반 진단
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

스킬의 적용 범위는 `description`과 본문의 작업 경계에서 확인합니다. 일부 스킬은 **Evals (발동 검증 질문)**에 적용·비적용 예시를 추가로 제공합니다. 실제 요청과 원자료로 적용 여부와 결과를 검증해야 하며, 발동 질문이나 형식 검사만으로 실행 정확성을 보장하지 않습니다.
