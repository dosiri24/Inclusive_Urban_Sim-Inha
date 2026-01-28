# CLAUDE.md

## Development Rules (Top Priority)

| # | Rule |
|---|------|
| 1 | **Implement only requested features**. No unrequested features or extensibility-oriented code |
| 2 | **Thorough comments** on all functions/classes for future modifications |
| 3 | **Unlimited tokens guaranteed** by user. Maximize MCP tools usage, read entire files when analyzing code |
| 4 | **Working code only**. No mocking, test prints, or placeholder outputs |
| 5 | **Research first** via MCP tools when coding AI/trending tech topics |
| 6 | **Thorough logging**. Reference logs when modifying code |
| 7 | **Test before adding** any feature to production code |
| 8 | **Concise + meaningful** names for functions, classes, files |
| 9 | **Expert-level code**, beginner-level explanations |
| 10 | **No code hardcopy** in plans. Focus on what/how/why |
| 11 | **Plan first, then develop**. Sync all changes back to plan document |
| 12 | **No unrequested error handling**. Return errors only for exceptions |
| 13 | **Documents in English** (except proper nouns). **Code comments in simple English**. **Conversations strictly in Korean** (English terms must have Korean translation in parentheses, or use Korean only) |
| 14 | **Follow DEVELOPMENT_PLAN.md**. All development must adhere to the plan document |
| 15 | **Detailed implementation plan required** before starting any new Phase. Write specific implementation details, expected outputs, and acceptance criteria |
| 16 | **LLM code requires verification**. When writing LLM-related code: (1) Research latest models/libraries via MCP tools, (2) Test with actual API calls, (3) Only then integrate into production code |

---

## Project Overview

| Item | Content |
|------|---------|
| Title | LLM Multi-Agent Simulation for Inclusive Citizen Participation: Comparing System Architectures |
| Duration | 6 months (1-3: Implementation / 4: Simulation / 5-6: Analysis & Paper) |
| Goal | Analyze how **technical implementation of discussion methods** affects **inclusion of socially vulnerable groups' opinions** |

### Research Hypotheses
- **H1**: Context separation → Independent expression of vulnerable groups' opinions → Higher inclusivity
- **H2**: Multiple models → Diverse perspectives due to different bias patterns → Higher inclusivity
- **H3**: Moderator agent → Balanced speaking opportunities → Higher inclusivity

---

## Experiment Design

### Main Experiment (Hypothesis Testing)

| Level | Method | Model | Hypothesis | Purpose |
|-------|--------|-------|------------|---------|
| Lv.1 | Single session, single model | Gemini 3 Flash | - | Reproduce Lee (2025), confirm limitations |
| Lv.2 | Separate context, single model | Gemini 3 Flash | H1 | Test context separation effect |
| Lv.3 | Separate context, multiple models | GPT, Claude, Gemini, Exaone, Kimi | H2 | Test model diversification effect |
| Lv.4 | Lv.3 + Moderator agent | Multiple models | H3 | Test moderator role effect |

**Scale**: 80 discussions (20 sets × 4 levels), same agent group proceeds Lv.1→4 sequentially

### Supplementary Experiment

| Level | Method | Model | Purpose |
|-------|--------|-------|---------|
| Lv.2-S | Separate context, single model | Exaone 4.0 | Compare Korean LLM vs Gemini (Lv.2) |

**Scale**: 20 discussions (20 sets × 1 level), uses same agent groups as main experiment

**Note**: Lv.2-S is an auxiliary experiment to explore whether Korean-specialized LLM (Exaone) shows different inclusivity patterns compared to Lv.2. Results will be reported separately from main hypothesis testing.

### LLM Models

| Provider | Model |
|----------|-------|
| Google | gemini-3-flash-preview |
| OpenAI | GPT-5 mini |
| Anthropic | Claude Haiku 4.5 |
| Moonshot AI | kimi-k2-0905-preview |
| LG AI Research | LGAI-EXAONE/K-EXAONE-236B-A23B |

---

## Environment Setup

### Paths

| Item | Path |
|------|------|
| Git Repository | `D:\Users\INHA\Desktop\포용적 주민참여를 위한 LLM 다중 에이전트 시뮬레이션 기술적 구현에 따른 효과 비교\Inclusive_Urban_Sim-Inha` |
| Papers Storage | `D:\Users\INHA\Desktop\논문\` |
| Paper Reviews | `D:\Users\INHA\Desktop\논문 리뷰\` |

### Python Execution

```bash
# Anaconda path (python command unavailable)
"C:\ProgramData\Anaconda3\python.exe" script.py
```

### File Format Support

| Format | Method |
|--------|--------|
| PDF | Claude direct read |
| HWP/HWPX | `gethwp` library |
| LNK | PowerShell to extract original path |

---

## Folder Structure

```
Inclusive_Urban_Sim-Inha/
├── CLAUDE.md
├── PAPERS.md          # Paper reviews
├── utils/             # Utilities (api_calculator.py, etc.)
├── data/              # Demographics, regional data
├── prompts/           # Agent profiles, discussion topics
├── src/               # Simulation code
├── results/           # Experiment results
└── analysis/          # Analysis code
```

---

*Last Modified: 2026.01.28*
