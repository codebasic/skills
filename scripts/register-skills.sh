#!/usr/bin/env bash
# ==============================================================================
# register-skills.sh
# 
# 잘 알려진 AI 에이전트(Claude Code, Antigravity, GitHub Copilot, OpenCode 등)를
# 시스템에서 자동 탐색하여, 이 저장소의 스킬들을 각 에이전트가 탐색할 수 있도록
# 심볼릭 링크 및 설정을 등록/관리합니다.
#
# 지원 에이전트:
#   - Claude Code       (~/.claude/skills)
#   - Google Antigravity (~/.gemini/config/skills.json & ~/.gemini/config/skills)
#   - GitHub Copilot    (~/.copilot/skills)
#   - OpenCode          (~/.config/opencode/skills)
#   - Agent Skills Hub  (~/.agents/skills)
# ==============================================================================

set -euo pipefail

# 터미널 색상 정의
if [ -t 1 ]; then
  BOLD="\033[1m"
  GREEN="\033[32m"
  YELLOW="\033[33m"
  BLUE="\033[34m"
  CYAN="\033[36m"
  RED="\033[31m"
  RESET="\033[0m"
else
  BOLD=""
  GREEN=""
  YELLOW=""
  BLUE=""
  CYAN=""
  RED=""
  RESET=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 모드 플래그
MODE="install"
TARGET_PROJECT=""
DRY_RUN=0

print_usage() {
  cat << EOF
${BOLD}사용법:${RESET}
  $0 [옵션]

${BOLD}옵션:${RESET}
  -h, --help               도움말 출력
  -s, --status             각 에이전트별 스킬 등록 현황 점검 (변경 없음)
  -p, --project <경로>     글로벌 대신 특정 프로젝트 디렉터리에 스킬 등록
  -u, --uninstall          등록된 스킬 심볼릭 링크 제거
  -n, --dry-run            실제 변경 없이 수행될 작업만 출력

${BOLD}예시:${RESET}
  $0                       # 시스템의 모든 에이전트에 스킬 자동 등록 (글로벌)
  $0 --status              # 현재 스킬 등록 상태 확인
  $0 -p /path/to/project   # 특정 프로젝트 워크스페이스에 스킬 등록
EOF
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    -s|--status)
      MODE="status"
      shift
      ;;
    -p|--project)
      if [[ -z "${2:-}" ]]; then
        echo -e "${RED}오류: --project 옵션에 디렉터리 경로를 지정하세요.${RESET}" >&2
        exit 1
      fi
      TARGET_PROJECT="$2"
      shift 2
      ;;
    -u|--uninstall)
      MODE="uninstall"
      shift
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo -e "${RED}알 수 없는 옵션: $1${RESET}" >&2
      print_usage
      exit 1
      ;;
  esac
done

# 저장소 내 스킬 자동 탐색 (SKILL.md가 존재하는 하위 디렉터리)
SKILLS=()
for skill_md in "$REPO_ROOT"/*/SKILL.md; do
  if [ -f "$skill_md" ]; then
    skill_dir="$(dirname "$skill_md")"
    SKILLS+=("$(basename "$skill_dir")")
  fi
done

if [ ${#SKILLS[@]} -eq 0 ]; then
  echo -e "${RED}오류: 저장소에서 스킬(SKILL.md)을 찾을 수 없습니다.${RESET}" >&2
  exit 1
fi

echo -e "${BOLD}======================================================${RESET}"
echo -e "${BOLD}🤖 AI 에이전트 스킬 통합 등록기 (Agent Skills Sync)${RESET}"
echo -e "${BOLD}======================================================${RESET}"
echo -e "저장소 루트: ${CYAN}${REPO_ROOT}${RESET}"
echo -e "발견된 스킬: ${GREEN}${SKILLS[*]}${RESET}"
echo ""

# 안전한 심볼릭 링크 생성 헬퍼
create_link() {
  local src="$1"
  local dest_dir="$2"
  local skill_name="$3"
  local dest="$dest_dir/$skill_name"

  if [ $DRY_RUN -eq 1 ]; then
    echo -e "  ${YELLOW}[DRY-RUN]${RESET} ln -sfn \"$src\" \"$dest\""
    return 0
  fi

  mkdir -p "$dest_dir"
  ln -sfn "$src" "$dest"
  echo -e "  ${GREEN}✔${RESET} $skill_name -> $dest"
}

# 심볼릭 링크 제거 헬퍼
remove_link() {
  local dest_dir="$1"
  local skill_name="$2"
  local dest="$dest_dir/$skill_name"

  if [ -L "$dest" ]; then
    if [ $DRY_RUN -eq 1 ]; then
      echo -e "  ${YELLOW}[DRY-RUN]${RESET} rm \"$dest\""
    else
      rm "$dest"
      echo -e "  ${RED}✖${RESET} $skill_name 제거됨 ($dest)"
    fi
  fi
}

# Antigravity skills.json 구성 헬퍼
configure_antigravity_json() {
  local config_dir="$HOME/.gemini/config"
  local config_file="$config_dir/skills.json"
  local target_path="$HOME/.agents/skills"

  if [ $DRY_RUN -eq 1 ]; then
    echo -e "  ${YELLOW}[DRY-RUN]${RESET} $config_file 에 $target_path 등록"
    return 0
  fi

  mkdir -p "$config_dir"
  if [ -f "$config_file" ]; then
    if grep -q "$target_path" "$config_file" 2>/dev/null; then
      echo -e "  ${GREEN}✔${RESET} Antigravity skills.json 이미 등록됨"
      return 0
    fi
  fi

  cat << EOF > "$config_file"
{
  "entries": [
    {
      "path": "$target_path"
    }
  ]
}
EOF
  echo -e "  ${GREEN}✔${RESET} Antigravity skills.json 생성/갱신 완료"
}

# ------------------------------------------------------------------------------
# 1. 특정 프로젝트 대상 모드 (--project)
# ------------------------------------------------------------------------------
if [ -n "$TARGET_PROJECT" ]; then
  proj_path="$(cd "$TARGET_PROJECT" 2>/dev/null && pwd || echo "$TARGET_PROJECT")"
  echo -e "${BOLD}[프로젝트 모드]${RESET} 대상: ${CYAN}${proj_path}${RESET}"
  echo ""

  # 각 에이전트의 프로젝트 표준 위치 정의
  TARGET_DIRS=(
    "$proj_path/.agents/skills"     # Agent Skills / Antigravity / Copilot
    "$proj_path/.claude/skills"     # Claude Code / OpenCode 호환
    "$proj_path/.github/skills"     # GitHub Copilot
    "$proj_path/.opencode/skills"   # OpenCode
  )

  for dir in "${TARGET_DIRS[@]}"; do
    rel_dir="${dir#$proj_path/}"
    echo -e "${BLUE}▶ ${rel_dir}${RESET}"
    if [ "$MODE" == "install" ]; then
      for skill in "${SKILLS[@]}"; do
        create_link "$REPO_ROOT/$skill" "$dir" "$skill"
      done
    elif [ "$MODE" == "uninstall" ]; then
      for skill in "${SKILLS[@]}"; do
        remove_link "$dir" "$skill"
      done
    elif [ "$MODE" == "status" ]; then
      for skill in "${SKILLS[@]}"; do
        if [ -L "$dir/$skill" ]; then
          echo -e "  ${GREEN}✔${RESET} $skill (연결됨 -> $(readlink "$dir/$skill"))"
        else
          echo -e "  ${YELLOW}○${RESET} $skill (미연결)"
        fi
      done
    fi
  done

  echo ""
  echo -e "${GREEN}${BOLD}프로젝트 동기화 작업이 완료되었습니다.${RESET}"
  exit 0
fi

# ------------------------------------------------------------------------------
# 2. 글로벌 시스템 에이전트 탐색 및 등록 모드 (기본)
# ------------------------------------------------------------------------------

# 에이전트 정의: 식별자 | 이름 | 글로벌 스킬 디렉터리 | 감지 조건 (폴더/CLI)
AGENTS=(
  "hub|Agent Skills Hub (표준)|$HOME/.agents/skills|$HOME/.agents"
  "claude|Claude Code|$HOME/.claude/skills|$HOME/.claude"
  "agy|Google Antigravity|$HOME/.gemini/config/skills|$HOME/.gemini"
  "copilot|GitHub Copilot|$HOME/.copilot/skills|$HOME/.copilot"
  "opencode|OpenCode|$HOME/.config/opencode/skills|$HOME/.config/opencode"
)

echo -e "${BOLD}[시스템 에이전트 탐색 및 스킬 동기화]${RESET}"
echo ""

for agent_def in "${AGENTS[@]}"; do
  IFS="|" read -r agent_id agent_name agent_skill_dir detect_path <<< "$agent_def"

  is_installed=0
  if [ -d "$detect_path" ] || command -v "$agent_id" >/dev/null 2>&1; then
    is_installed=1
  fi

  if [ $is_installed -eq 1 ]; then
    echo -e "${BLUE}▶ ${agent_name}${RESET} (${GREEN}설치 감지됨${RESET}: ${agent_skill_dir})"
  else
    echo -e "${BLUE}▶ ${agent_name}${RESET} (${YELLOW}미설치 또는 설정 폴더 없음${RESET})"
  fi

  if [ "$MODE" == "install" ]; then
    # 설치 감지된 에이전트에는 심볼릭 링크 생성
    if [ $is_installed -eq 1 ]; then
      for skill in "${SKILLS[@]}"; do
        create_link "$REPO_ROOT/$skill" "$agent_skill_dir" "$skill"
      done

      # Antigravity 전용 추가 처리 (skills.json 자동 구성)
      if [ "$agent_id" == "agy" ]; then
        configure_antigravity_json
      fi
    else
      echo -e "  ${YELLOW}건너뜀 (에이전트가 설치되어 있지 않음)${RESET}"
    fi

  elif [ "$MODE" == "uninstall" ]; then
    if [ -d "$agent_skill_dir" ]; then
      for skill in "${SKILLS[@]}"; do
        remove_link "$agent_skill_dir" "$skill"
      done
    fi

  elif [ "$MODE" == "status" ]; then
    for skill in "${SKILLS[@]}"; do
      dest="$agent_skill_dir/$skill"
      if [ -L "$dest" ]; then
        target="$(readlink "$dest")"
        echo -e "  ${GREEN}✔${RESET} $skill -> $target"
      elif [ -d "$dest" ]; then
        echo -e "  ${CYAN}[DIR]${RESET} $skill (실제 디렉터리)"
      else
        echo -e "  ${YELLOW}○${RESET} $skill (미연결)"
      fi
    done
    if [ "$agent_id" == "agy" ] && [ -f "$HOME/.gemini/config/skills.json" ]; then
      echo -e "  ${GREEN}✔${RESET} skills.json 구성됨"
    fi
  fi
  echo ""
done

if [ "$MODE" == "install" ]; then
  echo -e "${GREEN}${BOLD}모든 에이전트에 스킬 등록 및 동기화가 완료되었습니다!${RESET}"
  echo -e "상태를 점검하려면 ${CYAN}$0 --status${RESET} 를 실행하세요."
fi
