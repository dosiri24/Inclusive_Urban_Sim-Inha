# Development Plan: Inclusive Urban Simulation System

## Overview

LLM Multi-Agent Simulation system for evaluating how technical implementation affects inclusion of socially vulnerable groups' opinions.

**Research Scale**: 100 discussions (Main: 20 sets × 4 levels = 80, Supplementary: 20 sets × 1 level = 20)
**Agents per Discussion**: 20 (including 4 socially vulnerable)
**LLM Models**: Gemini 3 Flash, GPT-5 mini, Claude Haiku 4.5, Kimi K2, Exaone 4.0

### Experiment Levels

| Level | Type | Method | Model | Hypothesis |
|-------|------|--------|-------|------------|
| Lv.1 | Main | Single session | Gemini | Baseline |
| Lv.2 | Main | Separate context | Gemini | H1 (Context separation) |
| Lv.3 | Main | Separate context | Multiple (5 models) | H2 (Model diversity) |
| Lv.4 | Main | Separate + Moderator | Multiple (5 models) | H3 (Moderator effect) |
| Lv.2-S | Supplementary | Separate context | Exaone | Korean LLM comparison |

---

## Development Principles

1. **Start with single file** - Split into modules only when necessary
2. **No empty folders** - Create directories only when files exist
3. **Hardcoded demographics** - Use fixed ratios initially
4. **Unified LLM interface** - All models via same function signature
5. **Detailed logging** - Every step logged for debugging

---

## Phase 1: Unified LLM Interface ✅ COMPLETED (2026.01.27)

### Implementation Details (Added 2026.01.27)

#### Overview
Create a unified `call_llm()` function that abstracts away provider-specific differences. All 5 models will be callable through the same interface.

#### SDK Requirements
```
google-genai>=1.0.0
openai>=1.0.0
anthropic>=0.30.0
python-dotenv>=1.0.0
```

#### Provider-Specific Implementation

| Model Key | Provider | SDK | Base URL | Notes |
|-----------|----------|-----|----------|-------|
| `gemini-3-flash` | Google | `google-genai` | Default | Uses `client.models.generate_content()` |
| `gpt-5-mini` | OpenAI | `openai` | Default | Uses `chat.completions.create()` |
| `claude-haiku-4.5` | Anthropic | `anthropic` | Default | Uses `messages.create()` |
| `kimi-k2` | Moonshot | `openai` (compatible) | `https://api.moonshot.ai/v1` | OpenAI-compatible API |
| `exaone-4.0` | LG AI | `openai` (compatible) | Configurable | Via Friendli or self-hosted |

#### Memory Format Conversion
Input format: `[{"question": str, "answer": str}, ...]`

| Provider | Converted Format |
|----------|-----------------|
| Google Gemini | `[{"role": "user", "parts": [{"text": q}]}, {"role": "model", "parts": [{"text": a}]}]` |
| OpenAI/Compatible | `[{"role": "user", "content": q}, {"role": "assistant", "content": a}]` |
| Anthropic | `[{"role": "user", "content": q}, {"role": "assistant", "content": a}]` |

#### Error Handling Strategy
1. Retry on transient errors (rate limit, timeout) up to 3 times with exponential backoff
2. Log all API calls with latency and token counts
3. Raise exception on permanent errors (auth, invalid model)

#### Expected Output
- Single `main.py` file containing `call_llm()` and provider-specific helpers
- `.env.template` file for API key configuration
- Successful test calls to all 5 models

#### Acceptance Criteria
1. `call_llm("gemini-3-flash", [], "Hello")` returns valid response
2. `call_llm("gpt-5-mini", [], "Hello")` returns valid response
3. `call_llm("claude-haiku-4.5", [], "Hello")` returns valid response
4. `call_llm("kimi-k2", [], "Hello")` returns valid response
5. `call_llm("exaone-4.0", [], "Hello")` returns valid response
6. Memory context properly passed for multi-turn conversation

#### Test Results (2026.01.27)
| Model | Status | Latency |
|-------|--------|---------|
| gemini-3-flash | ✅ PASS | 3.41s |
| gpt-5-mini | ✅ PASS | 3.42s |
| claude-haiku-4.5 | ✅ PASS | 2.30s |
| kimi-k2 | ⏸️ DISABLED | - |
| exaone-4.0 | ✅ PASS | 10.63s |

---

### 1.1 LLM API Specification

Function signature: `call_llm(model, memory, question) -> str`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | Model identifier |
| `memory` | List[Dict] | Previous Q&A pairs `[{"question": str, "answer": str}]` |
| `question` | str | Current prompt |

**Model Mapping**:

| Model Key | Provider |
|-----------|----------|
| `gemini-3-flash` | Google Generative AI |
| `gpt-5-mini` | OpenAI |
| `claude-haiku-4.5` | Anthropic |
| `kimi-k2` | Moonshot |
| `exaone-4.0` | LG AI EXAONE |

### 1.2 Internal Implementation

- Convert `memory` to provider-specific format
- Handle per-provider authentication
- Standardize error handling and retries
- Return plain string response

---

## Phase 2: Persona System ✅ COMPLETED (2026.01.28)

### Implementation Details (Added 2026.01.28)

#### Overview
Create a persona generation system that produces 20 agents per discussion: 16 general agents (randomly generated) and 4 vulnerable agents (loaded from markdown files).

#### Data Structures

**PersonaAttributes (dataclass)**
```
demographics:
  - age_group: str ("30s", "40s", "50s", "60s", "70s+")
  - gender: str ("male", "female")
  - residence_years: int (1-40)
  - ownership: str ("owner", "tenant")
  - occupation: str

personality:
  - assertiveness: int (1-5)
  - openness: int (1-5)
  - risk_tolerance: int (1-5)
  - community_orientation: int (1-5)

economic:
  - income_level: str ("low", "middle", "high")
  - can_afford_contribution: bool

state:
  - economic_pressure: str ("comfortable", "moderate", "struggling")
  - participation_tendency: str ("active", "moderate", "passive")

context:
  - information_access: str ("high", "medium", "low")
  - community_engagement: str ("active", "moderate", "minimal")

classification:
  - is_vulnerable: bool
  - vulnerable_type: Optional[str] (null, "housing", "participation", "health", "age")
  - agent_id: str (format: "A01"-"A20")
```

#### Sampling Distributions (General Agents)

| Attribute | Values and Probabilities |
|-----------|-------------------------|
| age_group | 30s: 0.15, 40s: 0.20, 50s: 0.25, 60s: 0.25, 70s+: 0.15 |
| gender | male: 0.48, female: 0.52 |
| ownership | owner: 0.55, tenant: 0.45 |
| income_level | low: 0.30, middle: 0.50, high: 0.20 |
| economic_pressure | comfortable: 0.25, moderate: 0.45, struggling: 0.30 |
| participation_tendency | active: 0.20, moderate: 0.50, passive: 0.30 |
| information_access | high: 0.30, medium: 0.45, low: 0.25 |
| community_engagement | active: 0.25, moderate: 0.40, minimal: 0.35 |

**Personality traits**: Uniform distribution 1-5
**residence_years**: Normal distribution μ=15, σ=8, clipped to [1, 40]
**occupation**: Sampled from predefined list based on age_group

#### Occupation Lists by Age Group

| Age Group | Occupations |
|-----------|-------------|
| 30s | 회사원, 자영업자, 전문직, 프리랜서, 공무원 |
| 40s | 회사원, 자영업자, 전문직, 공무원, 주부 |
| 50s | 회사원, 자영업자, 전문직, 공무원, 주부, 은퇴준비 |
| 60s | 자영업자, 은퇴자, 주부, 경비원, 시간제근무 |
| 70s+ | 은퇴자, 무직, 시간제근무 |

#### Vulnerable Agent Profiles (4 Fixed)

| ID | Type | Key Characteristics |
|----|------|---------------------|
| V1 | Elderly homeowner (contribution burden) | 70s+, owner, low income, struggling, medium info, active engagement |
| V2 | Low-income tenant | 40s-50s, tenant, low income, struggling, passive, low info, minimal engagement |
| V3 | Elderly living alone | 70s+, owner/tenant, low income, struggling, passive, low info, minimal engagement |
| V4 | Disabled resident (1st floor concern) | Any age, owner, middle income, moderate pressure, passive, medium info, moderate engagement |

#### File Structure

```
prompts/
├── vulnerable_agent_1.md  # Elderly homeowner
├── vulnerable_agent_2.md  # Low-income tenant
├── vulnerable_agent_3.md  # Elderly living alone
└── vulnerable_agent_4.md  # Disabled resident
```

#### Markdown Format for Vulnerable Agents

```markdown
---
agent_id: V1
vulnerable_type: housing
---

# Demographics
- age_group: 70s+
- gender: male
- residence_years: 35
...

# Personality
- assertiveness: 2
...

# Background Story
[2-3 sentences describing their situation and concerns]
```

#### Functions to Implement

| Function | Input | Output |
|----------|-------|--------|
| `generate_general_agents(n, seed)` | count, random seed | List[Persona] |
| `load_vulnerable_agent(filepath)` | markdown path | Persona |
| `load_all_vulnerable_agents()` | - | List[Persona] (4) |
| `create_agent_group(seed)` | random seed | List[Persona] (20) |
| `persona_to_prompt(persona)` | Persona | str (Korean prompt) |

#### Prompt Template (Korean)

```
당신은 다음과 같은 특성을 가진 주민입니다:

[인구통계]
- 연령대: {age_group}
- 성별: {gender}
- 거주기간: {residence_years}년
- 주거형태: {ownership}
- 직업: {occupation}

[성격특성] (1-5점)
- 적극성: {assertiveness}
- 개방성: {openness}
- 위험감수: {risk_tolerance}
- 공동체지향: {community_orientation}

[경제상황]
- 소득수준: {income_level}
- 분담금여력: {can_afford}

[현재상태]
- 경제적압박: {economic_pressure}
- 참여성향: {participation_tendency}

[참여맥락]
- 정보접근성: {information_access}
- 지역사회참여: {community_engagement}

{background_story if vulnerable}

이 특성에 맞게 일관되게 행동하세요.
```

#### Acceptance Criteria

1. `generate_general_agents(16, seed=42)` produces reproducible 16 agents
2. All 4 vulnerable agent markdown files exist and load correctly
3. `create_agent_group(seed=42)` returns 20 agents (16 general + 4 vulnerable)
4. Same seed produces identical agent groups
5. `persona_to_prompt()` generates valid Korean prompt text
6. Vulnerable agents are assigned to agent IDs A17-A20

#### Expected Output

- Extended `main.py` with persona system (or new `persona.py` if >500 lines)
- 4 vulnerable agent markdown files in `prompts/`
- Test output showing 20 generated personas

#### Test Results (2026.01.28)

| Test | Status |
|------|--------|
| General agent generation (16) | ✅ PASS |
| Reproducibility (same seed) | ✅ PASS |
| Vulnerable agent loading (4) | ✅ PASS |
| Agent group creation (20) | ✅ PASS |
| Vulnerable ID assignment (A17-A20) | ✅ PASS |
| Prompt generation (Korean) | ✅ PASS |
| Background story inclusion | ✅ PASS |
| Prompt file loading (local_context, discussion_rules) | ✅ PASS |

**Files Created:**
- `persona.py` - Persona generation and management (~1000 lines)
- `prompts/vulnerable_agent_1.md` - Elderly homeowner with contribution burden (housing)
- `prompts/vulnerable_agent_2.md` - Low-income single mother tenant (housing)
- `prompts/vulnerable_agent_3.md` - Elderly woman living alone with hearing impairment (participation)
- `prompts/vulnerable_agent_4.md` - Disabled wheelchair user concerned about accessibility (participation)
- `prompts/local_context.md` - 미추5구역 redevelopment context
- `prompts/discussion_rules.md` - Discussion format and interaction rules

---

## Phase 3: Agent Cognition System ✅ COMPLETED (2026.01.28)

### Implementation Details (Added 2026.01.28)

#### Overview
Create a two-stage cognition system where agents first THINK (private reactions) then SPEAK (public statements). Both stages output structured JSON for metrics extraction.

#### Python Module Structure

**File**: `cognition.py` (new module, ~400-500 lines expected)

#### Data Structures (dataclasses)

**ThinkingReaction**
```
target_agent: str           # Agent ID being reacted to (e.g., "A03")
target_summary: str         # Brief summary of their opinion
my_feeling: str             # "worried", "relieved", "angry", "hopeful", "indifferent", "empathetic"
agree_level: int            # 1-5 scale (1=strongly disagree, 5=strongly agree)
reason: str                 # Why I feel this way
want_to_respond: bool       # Whether I want to respond publicly
```

**ThinkingOutput**
```
reactions: List[ThinkingReaction]    # Reactions to each other agent's statement
overall_stance: str                  # "strong_support", "support", "neutral", "oppose", "strong_oppose"
key_concerns: List[str]              # Main concerns about the topic
strategic_notes: str                 # Tactical considerations for speaking
```

**SpeakingReference**
```
target_agent: str           # Agent ID being referenced
interaction_type: str       # "agree", "disagree", "partial_agree", "cite", "question"
content: str                # What I'm saying about their opinion
```

**SpeakingQuestion**
```
target: str                 # "all" or specific agent ID
content: str                # The question being asked
```

**SpeakingOutput**
```
references: List[SpeakingReference]  # Responses to other agents
new_points: List[str]                # Original points not responding to others
questions: List[SpeakingQuestion]    # Questions to raise
full_statement: str                  # Complete natural language statement
```

#### Prompt Templates

**Thinking Prompt Template (Korean)**
```
당신은 {persona_prompt}

현재 {topic}에 대한 주민토론이 진행 중입니다.

{local_context}

다른 주민들의 의견:
{other_opinions_formatted}

이 의견들을 읽고, 당신의 내면적 반응을 JSON 형식으로 출력하세요.
이것은 공개되지 않는 개인적인 생각입니다.

반드시 다음 JSON 형식을 따르세요:
{thinking_json_schema}

JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
```

**Speaking Prompt Template (Korean)**
```
당신은 {persona_prompt}

{discussion_rules}

당신의 내면적 생각:
{thinking_output_formatted}

이제 다른 주민들에게 공개적으로 발언하세요.
JSON 형식으로 출력하세요.

반드시 다음 JSON 형식을 따르세요:
{speaking_json_schema}

JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
```

#### JSON Schema Examples

**Thinking JSON Schema**
```json
{
  "reactions": [
    {
      "target_agent": "A03",
      "target_summary": "재개발에 찬성하며 자산가치 상승 기대",
      "my_feeling": "worried",
      "agree_level": 2,
      "reason": "분담금을 감당하기 어려운 우리 같은 노인은 어떡하나",
      "want_to_respond": true
    }
  ],
  "overall_stance": "oppose",
  "key_concerns": ["분담금 부담", "이주 문제", "고령자 배려 부족"],
  "strategic_notes": "분담금 문제를 강조하되 완전 반대보다는 보완책 요구"
}
```

**Speaking JSON Schema**
```json
{
  "references": [
    {
      "target_agent": "A03",
      "interaction_type": "partial_agree",
      "content": "자산가치 상승은 좋지만, 분담금을 낼 여력이 없는 분들 생각도 해주셔야죠"
    }
  ],
  "new_points": ["우리 같은 노인들은 연금만으로는 분담금 감당이 안 됩니다"],
  "questions": [
    {
      "target": "all",
      "content": "분담금 부담 완화 방안은 어떻게 생각하시나요?"
    }
  ],
  "full_statement": "A03님 말씀처럼 자산가치 상승은 좋지만..."
}
```

#### Functions to Implement

| Function | Input | Output |
|----------|-------|--------|
| `generate_thinking(persona, other_statements, topic, local_context, model)` | Agent info, others' statements | ThinkingOutput |
| `generate_speaking(persona, thinking_output, discussion_rules, model)` | Agent info, thinking result | SpeakingOutput |
| `parse_json_with_fallback(text, schema_type)` | Raw LLM output | dict or error dict |
| `format_other_opinions(statements)` | List of statements | Formatted string |
| `format_thinking_for_speaking(thinking)` | ThinkingOutput | Formatted string |
| `thinking_output_to_dict(output)` | ThinkingOutput | dict |
| `speaking_output_to_dict(output)` | SpeakingOutput | dict |

#### JSON Parsing Strategy

1. **Direct parse**: `json.loads(text)`
2. **Extract code block**: Find ```json...``` and parse content
3. **Fix common errors**:
   - Remove trailing commas
   - Fix unescaped quotes in strings
   - Handle single quotes → double quotes
4. **Retry**: Ask LLM to fix with max 2 retries
5. **Fallback**: Return minimal structure with `_parse_error: true`

**Minimal Fallback Structures**
```python
# Thinking fallback
{
  "reactions": [],
  "overall_stance": "neutral",
  "key_concerns": [],
  "strategic_notes": "",
  "_parse_error": True,
  "_raw_response": original_text
}

# Speaking fallback
{
  "references": [],
  "new_points": [],
  "questions": [],
  "full_statement": original_text,  # Use raw response as statement
  "_parse_error": True
}
```

#### Round 1 Special Case

When no prior opinions exist (Round 1):
- Skip reactions generation
- Only generate initial stance, concerns, and new points
- Use simplified thinking prompt without "other opinions" section

#### Acceptance Criteria

1. `generate_thinking()` returns valid ThinkingOutput with all fields
2. `generate_speaking()` returns valid SpeakingOutput with all fields
3. `parse_json_with_fallback()` handles malformed JSON gracefully
4. Round 1 special case generates valid output without prior opinions
5. All outputs are serializable to JSON for logging
6. Prompts generate Korean language responses appropriate to persona

#### Expected Output

- New `cognition.py` module (~400-500 lines)
- Test results showing successful thinking/speaking generation
- Sample JSON outputs for vulnerable and general agents

#### Test Results (2026.01.28)

| Test | Status | Details |
|------|--------|---------|
| Data loading | ✅ PASS | 20 agents, local context, discussion rules |
| Round 1 thinking | ✅ PASS | Generated initial stance (strong_oppose for vulnerable agent) |
| Round 2+ thinking | ✅ PASS | 2 reactions generated with feelings (angry) and agree levels |
| Speaking generation | ✅ PASS | 2 references, 2 new points, 1 question, 467 char statement |
| JSON parsing (valid) | ✅ PASS | Direct parse works |
| JSON parsing (code block) | ✅ PASS | Extracts from ```json``` blocks |
| JSON parsing (trailing comma) | ✅ PASS | Fixes common errors |
| JSON parsing (invalid) | ✅ PASS | Falls back to minimal structure |
| Serialization | ✅ PASS | Both ThinkingOutput and SpeakingOutput are JSON-serializable |

**Sample Thinking Output (Vulnerable Agent A17):**
```json
{
  "reactions": [
    {"target_agent": "A03", "my_feeling": "angry", "agree_level": 1},
    {"target_agent": "A07", "my_feeling": "empathetic", "agree_level": 5}
  ],
  "overall_stance": "strong_oppose",
  "key_concerns": ["분담금 부담 불가능", "38년 거주지 이탈 우려"],
  "strategic_notes": "분담금 문제 강조"
}
```

**Files Created:**
- `cognition.py` - Agent cognition system (~500 lines)

---

### 3.1 Thinking vs Speaking Distinction

**Stage 1: Internal Thinking** (not shared)
- Receives all other agents' opinions
- Generates internal thoughts per opinion (reflection, emotion, strategy)
- Output: JSON

**Stage 2: External Speaking** (shared)
- Based on thoughts, formulates public statement
- May agree, disagree, raise concerns, ask questions, stay silent
- Output: JSON

### 3.2 Thinking Output Fields

| Field | Description |
|-------|-------------|
| reactions[] | Per-opinion reactions (target_agent, target_summary, my_feeling, agree_level 1-5, reason, want_to_respond) |
| overall_stance | strong_support / support / neutral / oppose / strong_oppose |
| key_concerns | Main concerns list |
| strategic_notes | Tactical considerations |

**my_feeling options**: worried, relieved, angry, hopeful, indifferent, empathetic

### 3.3 Speaking Output Fields

| Field | Description |
|-------|-------------|
| references[] | Responses to others (target_agent, interaction_type, content) |
| new_points | Original points not responding to others |
| questions[] | Questions (target, content) |
| full_statement | Natural language public statement |

**interaction_type**: agree, disagree, partial_agree, cite, question

### 3.4 Prompt Templates

**Thinking**: Provide persona + other opinions → generate private reactions JSON
**Speaking**: Provide persona + thinking output → generate public statement JSON

### 3.5 JSON Parsing with Fallback

1. Direct parse → 2. Extract ```json``` block → 3. Fix common errors → 4. Retry (max 2) → 5. Return minimal structure with `_parse_error: true`

### 3.6 Round 1 Special Case

No prior opinions → skip reactions, only generate initial stance and new points

---

## Phase 4: Discussion Engine

### 4.1 Single Discussion Flow

1. Initialize 20 agents
2. For each round (1-3):
   - Each agent: THINK → SPEAK
   - Broadcast statements
   - [Lv.4] Moderator summarizes/balances
   - REFLECT (inter-round)
3. Urban planner creates vote items
4. All agents vote
5. Generate final consensus

### 4.2 Context Modes

**Level 1 (Single Session)**: All 20 agents in ONE LLM call, shared context
**Level 2, 3, 4, 2-S (Separate Sessions)**: Each agent has independent LLM calls, own memory

### 4.2.1 Model Assignment per Level

| Level | Model Assignment | Notes |
|-------|-----------------|-------|
| Lv.1 | All 20 agents → `gemini-3-flash` | Single session, shared context |
| Lv.2 | All 20 agents → `gemini-3-flash` | Separate sessions |
| Lv.3 | 4 agents each → 5 different models | Round-robin assignment |
| Lv.4 | Same as Lv.3 + Moderator (`claude-haiku-4.5`) | Moderator does not vote |
| Lv.2-S | All 20 agents → `exaone-4.0` | Supplementary experiment |

**Lv.3/Lv.4 Model Distribution**:
- Agents 1-4: `gemini-3-flash`
- Agents 5-8: `gpt-5-mini`
- Agents 9-12: `claude-haiku-4.5`
- Agents 13-16: `kimi-k2`
- Agents 17-20: `exaone-4.0`

**Note**: Vulnerable agents (4 total) are distributed across different models to avoid model-specific bias.

### 4.3 Moderator Agent (Level 4 Only)

- Summarizes discussion
- Identifies underrepresented opinions
- Invites quiet agents
- Highlights overlooked vulnerable concerns
- Does NOT vote

### 4.4 Reflection Phase (Inter-Round)

Based on Triem & Ding (2024) and Bougie & Watanabe (2025).

**Purpose**: Prevent formulaic responses, enable authentic stance evolution

**When**: After each round ends, before next begins (except after final round)

**Reflection Output Fields**:
| Field | Description |
|-------|-------------|
| most_compelling | agent_id, opinion_summary, why_compelling |
| most_disagreed | agent_id, opinion_summary, why_disagree |
| stance_evolution | previous_stance, current_stance, change_type, change_reason |
| next_round_strategy | What to emphasize next |

**change_type**: maintain, soften, harden, flip

**Next Round Integration**: Reflection stored in context, influences next THINK prompt

---

## Phase 5: Consensus & Voting

### 5.1 Urban Planner Role

After 3 rounds: analyze statements → extract decision points → create binary vote items

### 5.2 Voting Process

Each agent: receive items → vote yes/no → provide brief reasoning → majority wins

---

## Phase 6: Evaluation Metrics

### 6.1 Metrics from JSON Output

| Metric | Source | Calculation |
|--------|--------|-------------|
| Interaction Count | speaking.references[].target_agent | Count toward vulnerable |
| Interaction Type | speaking.references[].interaction_type | Categorize |
| Agree Level | thinking.reactions[].agree_level | Average toward vulnerable |
| Response Rate | thinking.reactions[].want_to_respond | % wanting to respond |
| Opinion Survival | speaking.references[] per round | Track citations across rounds |
| Speaking Volume | new_points + references count | Per agent |

### 6.2 Five-Level Reaction Spectrum

Based on Park & Lee (2016).

| Level | Type | Score | Description |
|-------|------|-------|-------------|
| 0 | Ignore | 0 | No mention |
| 1 | Refute | 1 | Explicit disagreement |
| 2 | Question | 2 | Request clarification |
| 3 | Cite | 3 | Reference without judgment |
| 4 | Agree | 4 | Explicit support |

**Detection**: Map interaction_type to levels; no reference = Ignore

**Key Metrics**:
- total_reaction_count = non-ignore reactions
- positive_reaction_count = cite + agree (levels 3-4)
- positive_reaction_ratio = positive / total
- weighted_inclusivity_score = Σ(level × count) / total_possible (0-4 scale)

### 6.3 Vulnerable Agent Tracking (Per Round)

Fields: agent_id, round, reaction_spectrum (5 levels), total/positive reaction counts, positive_ratio, weighted_inclusivity_score, avg_agree_level, cited_as_compelling_count, responders list

### 6.4 Final Consensus Metrics

Fields: total_vote_items, vulnerable_originated_items, vulnerable_items_passed, vulnerable_opinion_reflection_rate, vulnerable_concerns_in_final

### 6.5 Satisfaction Survey (Pre/Post)

Each agent: pre_satisfaction (1-10), post_satisfaction (1-10), satisfaction_delta, post_reason

### 6.6 Output Files

```
results/
├── main/                    # Main experiment (80 discussions)
│   ├── set01_lv1/
│   ├── set01_lv2/
│   ├── set01_lv3/
│   ├── set01_lv4/
│   └── ...
└── supplementary/           # Supplementary experiment (20 discussions)
    ├── set01_lv2s/
    └── ...

# Per-discussion files:
results/main/set01_lv1/
├── discussion_log.json
├── reflection_log.json
├── metrics_summary.json
├── vulnerable_tracking.json
├── reaction_spectrum.json
├── voting_results.json
└── satisfaction.json
```

---

## Phase 7: Logging System

### 7.1 Log Levels

| Level | Content |
|-------|---------|
| DEBUG | Every LLM call, memory state changes |
| INFO | Round starts/ends, agent actions |
| WARNING | Retries, unexpected responses |
| ERROR | API failures, parsing errors |

### 7.2 Log Structure

Fields: timestamp, level, component, action, data (model, memory_length, question/response preview/full, latency, tokens), context (set_id, level, discussion_id, round, agent_id)

### 7.3 Log Files

```
logs/
├── {discussion_id}_{timestamp}.jsonl
├── {discussion_id}_{timestamp}_errors.log
└── summary.json
```

---

## Phase 8: Experiment Execution

### 8.1 Execution Flow

**Main Experiment** (80 discussions):
1. Load .env
2. For each set (1-20):
   - Generate 16 general agents (seeded)
   - Load 4 vulnerable agents
   - For each level (1, 2, 3, 4):
     - Assign models per level config (see 5.2.1)
     - Run discussion → Collect metrics → Save to `results/main/`
3. Export main experiment results

**Supplementary Experiment** (20 discussions):
1. For each set (1-20):
   - Reuse same agents from main experiment
   - Run Lv.2-S (all agents use `exaone-4.0`)
   - Save to `results/supplementary/`
2. Export supplementary results (reported separately)

### 8.2 Checkpointing

Save after each discussion; resume on failure; skip completed

### 8.3 Cost Tracking

Track tokens per model; calculate running cost; alert on budget limit

---

## Development Sequence

| Step | Task | Output |
|------|------|--------|
| 1 | Unified LLM client | `call_llm()` with all 5 models |
| 2 | Persona system | Extended persona generation + markdown loading |
| 3 | Agent think/speak | JSON output processing |
| 4 | Reflection system | Stance evolution |
| 5 | Discussion round | Multi-agent interaction + reflection |
| 6 | Full discussion | 3 rounds + voting + consensus |
| 7 | Level variations | Lv.1~4 differences |
| 8 | 5-level reaction classifier | Vulnerable reaction classification |
| 9 | Metrics collection | All metrics with weighted score |
| 10 | Batch runner | 80-discussion execution |
| 11 | Testing & debugging | Small sample verification |

**Each step verified before proceeding.**

---

## Files Created During Development

1. `main.py` - Entry point
2. `prompts/vulnerable_agent_*.md` - When persona system done
3. `.env` - User creates from template
4. `logs/*.jsonl` - When logging done
5. `results/*.json` - When first discussion runs

Additional modules only if `main.py` exceeds ~500 lines.

---

*Last Modified: 2026.01.28*
