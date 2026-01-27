# Development Plan: Inclusive Urban Simulation System

## Overview

LLM Multi-Agent Simulation system for evaluating how technical implementation affects inclusion of socially vulnerable groups' opinions.

**Research Scale**: 80 discussions (20 sets × 4 levels)
**Agents per Discussion**: 20 (including 4 socially vulnerable)
**LLM Models**: Gemini 3 Flash, GPT-5 mini, Claude Haiku 4.5, Kimi K2, Exaone 4.0

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

## Phase 2: Environment Configuration

### 2.1 Environment Variables (.env)

**API Keys**: GOOGLE, OPENAI, ANTHROPIC, MOONSHOT, EXAONE

**Experiment Parameters**:
- NUM_AGENTS=20, NUM_VULNERABLE=4, NUM_ROUNDS=3, NUM_SETS=20
- DEFAULT_MODEL, MODERATOR_MODEL, LOG_LEVEL, LOG_DIR

### 2.2 Runtime Configuration

- Current experiment level (1, 2-1, 2-2, 3, 4)
- Current set number
- Random seed for reproducibility

---

## Phase 3: Persona System

### 3.1 Standardized Persona Format

Based on Lee et al. (2024) and Bougie & Watanabe (2025).

**Demographics (Basic)**: age_group, gender, residence_years, ownership, occupation

**Personality (Basic)**: assertiveness, openness, risk_tolerance, community_orientation (1-5 scale)

**Economic (Basic)**: income_level (low/middle/high), can_afford_contribution

**State (Current disposition)**:
- economic_pressure: comfortable / moderate / struggling
- participation_tendency: active / moderate / passive

**Context (Participation factors)**:
- information_access: high / medium / low
- community_engagement: active / moderate / minimal

**Classification**: is_vulnerable, vulnerable_type (null / housing / participation)

### 3.2 State/Context Attribute Impacts

| Attribute | Impact |
|-----------|--------|
| economic_pressure | Priority concerns (cost vs. improvement) |
| participation_tendency | Speaking frequency and assertiveness |
| information_access | Knowledge depth and argument sophistication |
| community_engagement | Local awareness and social influence |

**Note**: `attitude_toward_redevelopment` excluded - agents form opinions through discussion.

### 3.3 General Agent Generation (16 agents)

Hardcoded ratios for sampling:
- age_group: 30s(0.15), 40s(0.20), 50s(0.25), 60s(0.25), 70s+(0.15)
- gender: male(0.48), female(0.52)
- ownership: owner(0.55), tenant(0.45)
- income_level: low(0.30), middle(0.50), high(0.20)
- economic_pressure: comfortable(0.25), moderate(0.45), struggling(0.30)
- participation_tendency: active(0.20), moderate(0.50), passive(0.30)
- information_access: high(0.30), medium(0.45), low(0.25)
- community_engagement: active(0.25), moderate(0.40), minimal(0.35)

### 3.4 Vulnerable Agent Profiles (4 agents, markdown files)

| Type | State Defaults | Context Defaults |
|------|---------------|------------------|
| Elderly homeowner (contribution burden) | struggling, moderate | medium, active |
| Low-income tenant | struggling, passive | low, minimal |
| Elderly living alone | struggling, passive | low, minimal |
| Disabled resident (1st floor) | moderate, passive | medium, moderate |

Files: `prompts/vulnerable_agent_1~4.md`

---

## Phase 4: Agent Cognition System

### 4.1 Thinking vs Speaking Distinction

**Stage 1: Internal Thinking** (not shared)
- Receives all other agents' opinions
- Generates internal thoughts per opinion (reflection, emotion, strategy)
- Output: JSON

**Stage 2: External Speaking** (shared)
- Based on thoughts, formulates public statement
- May agree, disagree, raise concerns, ask questions, stay silent
- Output: JSON

### 4.2 Thinking Output Fields

| Field | Description |
|-------|-------------|
| reactions[] | Per-opinion reactions (target_agent, target_summary, my_feeling, agree_level 1-5, reason, want_to_respond) |
| overall_stance | strong_support / support / neutral / oppose / strong_oppose |
| key_concerns | Main concerns list |
| strategic_notes | Tactical considerations |

**my_feeling options**: worried, relieved, angry, hopeful, indifferent, empathetic

### 4.3 Speaking Output Fields

| Field | Description |
|-------|-------------|
| references[] | Responses to others (target_agent, interaction_type, content) |
| new_points | Original points not responding to others |
| questions[] | Questions (target, content) |
| full_statement | Natural language public statement |

**interaction_type**: agree, disagree, partial_agree, cite, question

### 4.4 Prompt Templates

**Thinking**: Provide persona + other opinions → generate private reactions JSON
**Speaking**: Provide persona + thinking output → generate public statement JSON

### 4.5 JSON Parsing with Fallback

1. Direct parse → 2. Extract ```json``` block → 3. Fix common errors → 4. Retry (max 2) → 5. Return minimal structure with `_parse_error: true`

### 4.6 Round 1 Special Case

No prior opinions → skip reactions, only generate initial stance and new points

---

## Phase 5: Discussion Engine

### 5.1 Single Discussion Flow

1. Initialize 20 agents
2. For each round (1-3):
   - Each agent: THINK → SPEAK
   - Broadcast statements
   - [Lv.4] Moderator summarizes/balances
   - REFLECT (inter-round)
3. Urban planner creates vote items
4. All agents vote
5. Generate final consensus

### 5.2 Context Modes

**Level 1 (Single Session)**: All 20 agents in ONE LLM call, shared context
**Level 2+ (Separate Sessions)**: Each agent has independent LLM calls, own memory

### 5.3 Moderator Agent (Level 4 Only)

- Summarizes discussion
- Identifies underrepresented opinions
- Invites quiet agents
- Highlights overlooked vulnerable concerns
- Does NOT vote

### 5.4 Reflection Phase (Inter-Round)

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

## Phase 6: Consensus & Voting

### 6.1 Urban Planner Role

After 3 rounds: analyze statements → extract decision points → create binary vote items

### 6.2 Voting Process

Each agent: receive items → vote yes/no → provide brief reasoning → majority wins

---

## Phase 7: Evaluation Metrics

### 7.1 Metrics from JSON Output

| Metric | Source | Calculation |
|--------|--------|-------------|
| Interaction Count | speaking.references[].target_agent | Count toward vulnerable |
| Interaction Type | speaking.references[].interaction_type | Categorize |
| Agree Level | thinking.reactions[].agree_level | Average toward vulnerable |
| Response Rate | thinking.reactions[].want_to_respond | % wanting to respond |
| Opinion Survival | speaking.references[] per round | Track citations across rounds |
| Speaking Volume | new_points + references count | Per agent |

### 7.2 Five-Level Reaction Spectrum

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

### 7.3 Vulnerable Agent Tracking (Per Round)

Fields: agent_id, round, reaction_spectrum (5 levels), total/positive reaction counts, positive_ratio, weighted_inclusivity_score, avg_agree_level, cited_as_compelling_count, responders list

### 7.4 Final Consensus Metrics

Fields: total_vote_items, vulnerable_originated_items, vulnerable_items_passed, vulnerable_opinion_reflection_rate, vulnerable_concerns_in_final

### 7.5 Satisfaction Survey (Pre/Post)

Each agent: pre_satisfaction (1-10), post_satisfaction (1-10), satisfaction_delta, post_reason

### 7.6 Output Files

```
results/set01_lv1/
├── discussion_log.json
├── reflection_log.json
├── metrics_summary.json
├── vulnerable_tracking.json
├── reaction_spectrum.json
├── voting_results.json
└── satisfaction.json
```

---

## Phase 8: Logging System

### 8.1 Log Levels

| Level | Content |
|-------|---------|
| DEBUG | Every LLM call, memory state changes |
| INFO | Round starts/ends, agent actions |
| WARNING | Retries, unexpected responses |
| ERROR | API failures, parsing errors |

### 8.2 Log Structure

Fields: timestamp, level, component, action, data (model, memory_length, question/response preview/full, latency, tokens), context (set_id, level, discussion_id, round, agent_id)

### 8.3 Log Files

```
logs/
├── {discussion_id}_{timestamp}.jsonl
├── {discussion_id}_{timestamp}_errors.log
└── summary.json
```

---

## Phase 9: Experiment Execution

### 9.1 Execution Flow

1. Load .env
2. For each set (1-20):
   - Generate 16 general agents (seeded)
   - Load 4 vulnerable agents
   - For each level (1, 2-1, 2-2, 3, 4): run → metrics → save → log
3. Export results

### 9.2 Checkpointing

Save after each discussion; resume on failure; skip completed

### 9.3 Cost Tracking

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

*Last Modified: 2026.01.27*
