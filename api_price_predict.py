"""
포용적 주민참여를 위한 LLM 다중 에이전트 시뮬레이션 API 비용 예측 모듈

연구 조건:
Lv.1. 단일 세션, gemini 단일 모델 (향후 구현 예정)
Lv.2. 개별 세션, gemini 단일 모델
Lv.3. 개별 세션, 다중 모델 (무작위 배정)
Lv.4. 개별 세션, 다중 모델, 무작위 모델 사회자 (향후 구현 예정)

시뮬레이션 흐름:
- Round 0: narrative(병렬) → initial(병렬)
- Round 1~3: speaking(순차) → reaction(병렬) → reflection(병렬)
- Round 4: final(병렬)

캐시 적용 (두 가지 시나리오):
- Conservative (보수적): 병렬=0%, 순차(speaking)=100%
- Optimistic (낙관적): 모든 후속 호출에서 이전 prefix 캐시
- 실측 캐시 비율: ~56% (두 시나리오 사이)
"""

import random

# =============================================================================
# Simulation Settings
# =============================================================================

NUM_AGENTS = 10
SAMPLES_PER_LV = 30
ROUNDS_PER_DEBATE = 3

# =============================================================================
# Token Constants (based on actual test measurements)
# =============================================================================

# Static prompts (Memory slots 1~4)
TOKEN_SYSTEM_GUIDE = 150      # slot 1: system_guide.md (~370 chars)
TOKEN_DEBATE_RULE = 600       # slot 2: debate_rule.md (~1500 chars)
TOKEN_LOCAL_CONTEXT = 280     # slot 3: local_context.md (~700 chars)
TOKEN_PERSONA_AVG = 150       # slot 4: persona (~375 chars avg, including vulnerable)
TOKEN_STATIC = TOKEN_SYSTEM_GUIDE + TOKEN_DEBATE_RULE + TOKEN_LOCAL_CONTEXT + TOKEN_PERSONA_AVG  # ~1180

# Task prompts (input)
TOKEN_TASK_NARRATIVE = 250
TOKEN_TASK_INITIAL = 150
TOKEN_TASK_SPEAKING = 100
TOKEN_TASK_THINK = 150
TOKEN_TASK_REFLECTION = 120
TOKEN_TASK_FINAL = 130

# Output tokens (based on test measurements)
TOKEN_OUT_NARRATIVE = 320
TOKEN_OUT_INITIAL = 90
TOKEN_OUT_SPEAKING = 140
TOKEN_OUT_REACTION = 80
TOKEN_OUT_REFLECTION = 80
TOKEN_OUT_FINAL = 100

# Moderator (Lv4 only)
TOKEN_TASK_MODERATOR = 200
TOKEN_OUT_MODERATOR = 300

# =============================================================================
# Model Pricing (USD per 1M tokens) - Currently Implemented Models Only
# =============================================================================

MODELS = {
    "gemini-3-flash-preview": {
        "name": "gemini-3-flash-preview",
        "input": 0.50,
        "cached_input": 0.05,
        "output": 3.00
    },
    "LGAI-EXAONE": {
        "name": "LGAI-EXAONE/K-EXAONE-236B-A23B",
        "input": 0.60,
        "cached_input": 0.06,
        "output": 1.00
    },
    "kimi-k2-0905-preview": {
        "name": "kimi-k2-0905-preview",
        "input": 0.60,
        "cached_input": 0.15,
        "output": 2.50
    },
    "Claude-Haiku-4.5": {
        "name": "Claude Haiku 4.5",
        "input": 1.00,
        "cached_input": 0.10,
        "output": 5.00
    },
    "GPT-5-mini": {
        "name": "GPT-5-mini",
        "input": 0.25,
        "cached_input": 0.025,
        "output": 2.00
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_individual_session_tokens_with_cache(n_agents: int = NUM_AGENTS, n_rounds: int = ROUNDS_PER_DEBATE, scenario: str = "conservative"):
    """
    Calculate total input/output tokens for one simulation with cache tracking.

    Args:
        scenario: "conservative" or "optimistic"
            - conservative: parallel=0% cache, sequential(speaking)=100% cache
            - optimistic: all subsequent calls use previous prefix cache

    Returns:
        (total_input, total_output, cached_input, non_cached_input)
    """
    total_input = 0
    total_output = 0
    total_cached = 0
    total_non_cached = 0

    agent_conv_history = {i: 0 for i in range(n_agents)}
    agent_think = {i: 0 for i in range(n_agents)}
    agent_prev_prefix = {i: 0 for i in range(n_agents)}

    # === Phase 0-1: Narrative (parallel) - first call, no cache ===
    for agent_id in range(n_agents):
        input_tokens = TOKEN_STATIC + TOKEN_TASK_NARRATIVE
        total_input += input_tokens
        total_output += TOKEN_OUT_NARRATIVE
        total_non_cached += input_tokens

        agent_think[agent_id] += TOKEN_OUT_NARRATIVE
        agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

    # === Phase 0-2: Initial opinion (parallel) ===
    for agent_id in range(n_agents):
        input_tokens = TOKEN_STATIC + agent_think[agent_id] + TOKEN_TASK_INITIAL
        total_input += input_tokens
        total_output += TOKEN_OUT_INITIAL

        if scenario == "optimistic":
            cached = agent_prev_prefix[agent_id]
            non_cached = input_tokens - cached
            total_cached += cached
            total_non_cached += non_cached
        else:  # conservative: parallel = no cache
            total_non_cached += input_tokens

        agent_think[agent_id] += TOKEN_OUT_INITIAL
        agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

    # === Phase 1~3: Debate rounds ===
    for _ in range(n_rounds):
        for speaker_id in range(n_agents):
            # === Speaker's turn (sequential) - cache in both scenarios ===
            speaker_input = (TOKEN_STATIC +
                           agent_think[speaker_id] +
                           agent_conv_history[speaker_id] +
                           TOKEN_TASK_SPEAKING)
            total_input += speaker_input
            total_output += TOKEN_OUT_SPEAKING

            cached = agent_prev_prefix[speaker_id]
            current_prefix = TOKEN_STATIC + agent_conv_history[speaker_id] + agent_think[speaker_id]
            delta = current_prefix - cached
            non_cached = delta + TOKEN_TASK_SPEAKING
            total_cached += cached
            total_non_cached += non_cached
            agent_prev_prefix[speaker_id] = current_prefix

            for other_id in range(n_agents):
                if other_id != speaker_id:
                    agent_conv_history[other_id] += TOKEN_OUT_SPEAKING

            # === Other agents react (parallel) ===
            for reactor_id in range(n_agents):
                if reactor_id != speaker_id:
                    reactor_input = (TOKEN_STATIC +
                                   agent_think[reactor_id] +
                                   agent_conv_history[reactor_id] +
                                   TOKEN_TASK_THINK)
                    total_input += reactor_input
                    total_output += TOKEN_OUT_REACTION

                    if scenario == "optimistic":
                        cached = agent_prev_prefix[reactor_id]
                        current_prefix = TOKEN_STATIC + agent_conv_history[reactor_id] + agent_think[reactor_id]
                        delta = current_prefix - cached
                        non_cached = delta + TOKEN_TASK_THINK
                        total_cached += cached
                        total_non_cached += non_cached
                    else:  # conservative: parallel = no cache
                        total_non_cached += reactor_input

                    agent_think[reactor_id] += TOKEN_OUT_REACTION
                    agent_prev_prefix[reactor_id] = TOKEN_STATIC + agent_conv_history[reactor_id] + agent_think[reactor_id]

        # === End of round reflection (parallel) ===
        for agent_id in range(n_agents):
            reflect_input = (TOKEN_STATIC +
                           agent_think[agent_id] +
                           agent_conv_history[agent_id] +
                           TOKEN_TASK_REFLECTION)
            total_input += reflect_input
            total_output += TOKEN_OUT_REFLECTION

            if scenario == "optimistic":
                cached = agent_prev_prefix[agent_id]
                current_prefix = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]
                delta = current_prefix - cached
                non_cached = delta + TOKEN_TASK_REFLECTION
                total_cached += cached
                total_non_cached += non_cached
            else:  # conservative: parallel = no cache
                total_non_cached += reflect_input

            agent_think[agent_id] += TOKEN_OUT_REFLECTION
            agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

    # === Phase 4: Final opinion (parallel) ===
    for agent_id in range(n_agents):
        final_input = (TOKEN_STATIC +
                      agent_think[agent_id] +
                      agent_conv_history[agent_id] +
                      TOKEN_TASK_FINAL)
        total_input += final_input
        total_output += TOKEN_OUT_FINAL

        if scenario == "optimistic":
            cached = agent_prev_prefix[agent_id]
            current_prefix = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]
            delta = current_prefix - cached
            non_cached = delta + TOKEN_TASK_FINAL
            total_cached += cached
            total_non_cached += non_cached
        else:  # conservative: parallel = no cache
            total_non_cached += final_input

    return total_input, total_output, total_cached, total_non_cached


def calculate_shared_session_tokens(n_agents: int = NUM_AGENTS, n_rounds: int = ROUNDS_PER_DEBATE):
    """
    Calculate tokens for shared session (Lv1) - all agents share one context.
    Returns: (total_input, total_output, cached_input, non_cached_input)
    """
    total_input = 0
    total_output = 0
    total_cached = 0
    total_non_cached = 0

    shared_static = TOKEN_SYSTEM_GUIDE + TOKEN_DEBATE_RULE + TOKEN_LOCAL_CONTEXT + (TOKEN_PERSONA_AVG * n_agents)
    accumulated_output = 0
    prev_prefix = 0

    # === Phase 0-1: Narrative (first call, no cache) ===
    input_tokens = shared_static + TOKEN_TASK_NARRATIVE
    total_input += input_tokens
    total_output += TOKEN_OUT_NARRATIVE * n_agents
    total_non_cached += input_tokens
    accumulated_output += TOKEN_OUT_NARRATIVE * n_agents
    prev_prefix = shared_static + accumulated_output

    # === Phase 0-2: Initial opinion ===
    input_tokens = shared_static + accumulated_output + TOKEN_TASK_INITIAL
    total_input += input_tokens
    total_output += TOKEN_OUT_INITIAL * n_agents
    cached = prev_prefix
    delta = accumulated_output - (prev_prefix - shared_static)
    non_cached = delta + TOKEN_TASK_INITIAL
    total_cached += cached
    total_non_cached += non_cached
    accumulated_output += TOKEN_OUT_INITIAL * n_agents
    prev_prefix = shared_static + accumulated_output

    # === Phase 1~3: Debate rounds ===
    for _ in range(n_rounds):
        # Speaking
        round_speaking_output = TOKEN_OUT_SPEAKING * n_agents
        input_tokens = shared_static + accumulated_output + TOKEN_TASK_SPEAKING
        total_input += input_tokens
        total_output += round_speaking_output
        cached = prev_prefix
        delta = accumulated_output - (prev_prefix - shared_static)
        non_cached = delta + TOKEN_TASK_SPEAKING
        total_cached += cached
        total_non_cached += non_cached
        accumulated_output += round_speaking_output
        prev_prefix = shared_static + accumulated_output

        # Think
        round_think_output = TOKEN_OUT_REACTION * n_agents
        input_tokens = shared_static + accumulated_output + TOKEN_TASK_THINK
        total_input += input_tokens
        total_output += round_think_output
        cached = prev_prefix
        delta = accumulated_output - (prev_prefix - shared_static)
        non_cached = delta + TOKEN_TASK_THINK
        total_cached += cached
        total_non_cached += non_cached
        accumulated_output += round_think_output
        prev_prefix = shared_static + accumulated_output

        # Reflection
        round_reflect_output = TOKEN_OUT_REFLECTION * n_agents
        input_tokens = shared_static + accumulated_output + TOKEN_TASK_REFLECTION
        total_input += input_tokens
        total_output += round_reflect_output
        cached = prev_prefix
        delta = accumulated_output - (prev_prefix - shared_static)
        non_cached = delta + TOKEN_TASK_REFLECTION
        total_cached += cached
        total_non_cached += non_cached
        accumulated_output += round_reflect_output
        prev_prefix = shared_static + accumulated_output

    # === Phase 4: Final opinion ===
    input_tokens = shared_static + accumulated_output + TOKEN_TASK_FINAL
    total_input += input_tokens
    total_output += TOKEN_OUT_FINAL * n_agents
    cached = prev_prefix
    delta = accumulated_output - (prev_prefix - shared_static)
    non_cached = delta + TOKEN_TASK_FINAL
    total_cached += cached
    total_non_cached += non_cached

    return total_input, total_output, total_cached, total_non_cached


# =============================================================================
# Level Functions
# =============================================================================

def lv1():
    """Lv.1. 단일 세션, 단일 모델 (향후 구현 예정)"""
    model = MODELS["gemini-3-flash-preview"]

    total_input = 0
    total_output = 0
    total_cached = 0
    total_non_cached = 0

    for _ in range(SAMPLES_PER_LV):
        inp, out, cached, non_cached = calculate_shared_session_tokens()
        total_input += inp
        total_output += out
        total_cached += cached
        total_non_cached += non_cached

    cost_cached = (total_cached / 1_000_000) * model["cached_input"]
    cost_non_cached = (total_non_cached / 1_000_000) * model["input"]
    cost_input = cost_cached + cost_non_cached
    cost_output = (total_output / 1_000_000) * model["output"]
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.1 (planned)",
        "description": "단일 세션, 단일 모델",
        "model": model["name"],
        "total_input_tokens": total_input,
        "cached_input_tokens": total_cached,
        "non_cached_input_tokens": total_non_cached,
        "total_output_tokens": total_output,
        "cost_cached_input": round(cost_cached, 2),
        "cost_non_cached_input": round(cost_non_cached, 2),
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total / SAMPLES_PER_LV, 4)
    }


def lv2(scenario: str = "conservative"):
    """Lv.2. 개별 세션, gemini 단일 모델"""
    model = MODELS["gemini-3-flash-preview"]

    total_input = 0
    total_output = 0
    total_cached = 0
    total_non_cached = 0

    for _ in range(SAMPLES_PER_LV):
        inp, out, cached, non_cached = calculate_individual_session_tokens_with_cache(scenario=scenario)
        total_input += inp
        total_output += out
        total_cached += cached
        total_non_cached += non_cached

    cost_cached = (total_cached / 1_000_000) * model["cached_input"]
    cost_non_cached = (total_non_cached / 1_000_000) * model["input"]
    cost_input = cost_cached + cost_non_cached
    cost_output = (total_output / 1_000_000) * model["output"]
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.2",
        "scenario": scenario,
        "description": "개별 세션, 단일 모델",
        "model": model["name"],
        "total_input_tokens": total_input,
        "cached_input_tokens": total_cached,
        "non_cached_input_tokens": total_non_cached,
        "total_output_tokens": total_output,
        "cost_cached_input": round(cost_cached, 2),
        "cost_non_cached_input": round(cost_non_cached, 2),
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total / SAMPLES_PER_LV, 4)
    }


def lv3(scenario: str = "conservative"):
    """Lv.3. 개별 세션, 다중 모델 (무작위 배정)"""
    model_list = list(MODELS.values())

    total_input_by_model = {m["name"]: 0 for m in model_list}
    total_output_by_model = {m["name"]: 0 for m in model_list}
    total_cached_by_model = {m["name"]: 0 for m in model_list}
    total_non_cached_by_model = {m["name"]: 0 for m in model_list}

    for _ in range(SAMPLES_PER_LV):
        assigned_models = [random.choice(model_list) for _ in range(NUM_AGENTS)]

        agent_conv_history = {i: 0 for i in range(NUM_AGENTS)}
        agent_think = {i: 0 for i in range(NUM_AGENTS)}
        agent_prev_prefix = {i: 0 for i in range(NUM_AGENTS)}

        # Phase 0-1: Narrative (parallel) - first call, no cache
        for agent_id in range(NUM_AGENTS):
            model = assigned_models[agent_id]
            input_tokens = TOKEN_STATIC + TOKEN_TASK_NARRATIVE
            total_input_by_model[model["name"]] += input_tokens
            total_output_by_model[model["name"]] += TOKEN_OUT_NARRATIVE
            total_non_cached_by_model[model["name"]] += input_tokens

            agent_think[agent_id] += TOKEN_OUT_NARRATIVE
            agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

        # Phase 0-2: Initial (parallel)
        for agent_id in range(NUM_AGENTS):
            model = assigned_models[agent_id]
            input_tokens = TOKEN_STATIC + agent_think[agent_id] + TOKEN_TASK_INITIAL
            total_input_by_model[model["name"]] += input_tokens
            total_output_by_model[model["name"]] += TOKEN_OUT_INITIAL

            if scenario == "optimistic":
                cached = agent_prev_prefix[agent_id]
                non_cached = input_tokens - cached
                total_cached_by_model[model["name"]] += cached
                total_non_cached_by_model[model["name"]] += non_cached
            else:
                total_non_cached_by_model[model["name"]] += input_tokens

            agent_think[agent_id] += TOKEN_OUT_INITIAL
            agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

        # Phase 1~3: Rounds
        for _ in range(ROUNDS_PER_DEBATE):
            for speaker_id in range(NUM_AGENTS):
                # Speaking (sequential) - cache in both scenarios
                model = assigned_models[speaker_id]
                speaker_input = (TOKEN_STATIC + agent_think[speaker_id] +
                               agent_conv_history[speaker_id] + TOKEN_TASK_SPEAKING)
                total_input_by_model[model["name"]] += speaker_input
                total_output_by_model[model["name"]] += TOKEN_OUT_SPEAKING

                cached = agent_prev_prefix[speaker_id]
                current_prefix = TOKEN_STATIC + agent_conv_history[speaker_id] + agent_think[speaker_id]
                delta = current_prefix - cached
                non_cached = delta + TOKEN_TASK_SPEAKING
                total_cached_by_model[model["name"]] += cached
                total_non_cached_by_model[model["name"]] += non_cached
                agent_prev_prefix[speaker_id] = current_prefix

                for other_id in range(NUM_AGENTS):
                    if other_id != speaker_id:
                        agent_conv_history[other_id] += TOKEN_OUT_SPEAKING

                # Reaction (parallel)
                for reactor_id in range(NUM_AGENTS):
                    if reactor_id != speaker_id:
                        model = assigned_models[reactor_id]
                        reactor_input = (TOKEN_STATIC + agent_think[reactor_id] +
                                       agent_conv_history[reactor_id] + TOKEN_TASK_THINK)
                        total_input_by_model[model["name"]] += reactor_input
                        total_output_by_model[model["name"]] += TOKEN_OUT_REACTION

                        if scenario == "optimistic":
                            cached = agent_prev_prefix[reactor_id]
                            current_prefix = TOKEN_STATIC + agent_conv_history[reactor_id] + agent_think[reactor_id]
                            delta = current_prefix - cached
                            non_cached = delta + TOKEN_TASK_THINK
                            total_cached_by_model[model["name"]] += cached
                            total_non_cached_by_model[model["name"]] += non_cached
                        else:
                            total_non_cached_by_model[model["name"]] += reactor_input

                        agent_think[reactor_id] += TOKEN_OUT_REACTION
                        agent_prev_prefix[reactor_id] = TOKEN_STATIC + agent_conv_history[reactor_id] + agent_think[reactor_id]

            # Reflection (parallel)
            for agent_id in range(NUM_AGENTS):
                model = assigned_models[agent_id]
                reflect_input = (TOKEN_STATIC + agent_think[agent_id] +
                               agent_conv_history[agent_id] + TOKEN_TASK_REFLECTION)
                total_input_by_model[model["name"]] += reflect_input
                total_output_by_model[model["name"]] += TOKEN_OUT_REFLECTION

                if scenario == "optimistic":
                    cached = agent_prev_prefix[agent_id]
                    current_prefix = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]
                    delta = current_prefix - cached
                    non_cached = delta + TOKEN_TASK_REFLECTION
                    total_cached_by_model[model["name"]] += cached
                    total_non_cached_by_model[model["name"]] += non_cached
                else:
                    total_non_cached_by_model[model["name"]] += reflect_input

                agent_think[agent_id] += TOKEN_OUT_REFLECTION
                agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

        # Phase 4: Final (parallel)
        for agent_id in range(NUM_AGENTS):
            model = assigned_models[agent_id]
            final_input = (TOKEN_STATIC + agent_think[agent_id] +
                          agent_conv_history[agent_id] + TOKEN_TASK_FINAL)
            total_input_by_model[model["name"]] += final_input
            total_output_by_model[model["name"]] += TOKEN_OUT_FINAL

            if scenario == "optimistic":
                cached = agent_prev_prefix[agent_id]
                current_prefix = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]
                delta = current_prefix - cached
                non_cached = delta + TOKEN_TASK_FINAL
                total_cached_by_model[model["name"]] += cached
                total_non_cached_by_model[model["name"]] += non_cached
            else:
                total_non_cached_by_model[model["name"]] += final_input

    cost_cached = sum((total_cached_by_model[m["name"]] / 1_000_000) * m["cached_input"] for m in model_list)
    cost_non_cached = sum((total_non_cached_by_model[m["name"]] / 1_000_000) * m["input"] for m in model_list)
    cost_input = cost_cached + cost_non_cached
    cost_output = sum((total_output_by_model[m["name"]] / 1_000_000) * m["output"] for m in model_list)
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.3",
        "scenario": scenario,
        "description": "개별 세션, 다중 모델 (무작위)",
        "models": {m["name"]: {
            "input_tokens": total_input_by_model[m["name"]],
            "cached_tokens": total_cached_by_model[m["name"]],
            "non_cached_tokens": total_non_cached_by_model[m["name"]],
            "output_tokens": total_output_by_model[m["name"]]
        } for m in model_list},
        "total_input_tokens": sum(total_input_by_model.values()),
        "cached_input_tokens": sum(total_cached_by_model.values()),
        "non_cached_input_tokens": sum(total_non_cached_by_model.values()),
        "total_output_tokens": sum(total_output_by_model.values()),
        "cost_cached_input": round(cost_cached, 2),
        "cost_non_cached_input": round(cost_non_cached, 2),
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total / SAMPLES_PER_LV, 4)
    }


def lv4(scenario: str = "conservative"):
    """Lv.4. 개별 세션, 다중 모델, 무작위 모델 사회자 (향후 구현 예정)"""
    model_list = list(MODELS.values())

    total_input_by_model = {m["name"]: 0 for m in model_list}
    total_output_by_model = {m["name"]: 0 for m in model_list}
    total_cached_by_model = {m["name"]: 0 for m in model_list}
    total_non_cached_by_model = {m["name"]: 0 for m in model_list}

    for _ in range(SAMPLES_PER_LV):
        assigned_models = [random.choice(model_list) for _ in range(NUM_AGENTS)]
        moderator_model = random.choice(model_list)

        agent_conv_history = {i: 0 for i in range(NUM_AGENTS)}
        agent_think = {i: 0 for i in range(NUM_AGENTS)}
        agent_prev_prefix = {i: 0 for i in range(NUM_AGENTS)}
        moderator_summary_accumulated = 0
        moderator_prev_prefix = 0

        # Phase 0-1: Narrative (parallel) - first call, no cache
        for agent_id in range(NUM_AGENTS):
            model = assigned_models[agent_id]
            input_tokens = TOKEN_STATIC + TOKEN_TASK_NARRATIVE
            total_input_by_model[model["name"]] += input_tokens
            total_output_by_model[model["name"]] += TOKEN_OUT_NARRATIVE
            total_non_cached_by_model[model["name"]] += input_tokens

            agent_think[agent_id] += TOKEN_OUT_NARRATIVE
            agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

        # Phase 0-2: Initial (parallel)
        for agent_id in range(NUM_AGENTS):
            model = assigned_models[agent_id]
            input_tokens = TOKEN_STATIC + agent_think[agent_id] + TOKEN_TASK_INITIAL
            total_input_by_model[model["name"]] += input_tokens
            total_output_by_model[model["name"]] += TOKEN_OUT_INITIAL

            if scenario == "optimistic":
                cached = agent_prev_prefix[agent_id]
                non_cached = input_tokens - cached
                total_cached_by_model[model["name"]] += cached
                total_non_cached_by_model[model["name"]] += non_cached
            else:
                total_non_cached_by_model[model["name"]] += input_tokens

            agent_think[agent_id] += TOKEN_OUT_INITIAL
            agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id]

        # Phase 1~3: Rounds with moderator
        for _ in range(ROUNDS_PER_DEBATE):
            round_speeches_total = 0

            for speaker_id in range(NUM_AGENTS):
                # Speaking (sequential) - cache in both scenarios
                model = assigned_models[speaker_id]
                speaker_input = (TOKEN_STATIC + agent_think[speaker_id] +
                               agent_conv_history[speaker_id] +
                               moderator_summary_accumulated + TOKEN_TASK_SPEAKING)
                total_input_by_model[model["name"]] += speaker_input
                total_output_by_model[model["name"]] += TOKEN_OUT_SPEAKING
                round_speeches_total += TOKEN_OUT_SPEAKING

                cached = agent_prev_prefix[speaker_id]
                current_prefix = TOKEN_STATIC + agent_conv_history[speaker_id] + agent_think[speaker_id] + moderator_summary_accumulated
                delta = current_prefix - cached
                non_cached = delta + TOKEN_TASK_SPEAKING
                total_cached_by_model[model["name"]] += cached
                total_non_cached_by_model[model["name"]] += non_cached
                agent_prev_prefix[speaker_id] = current_prefix

                for other_id in range(NUM_AGENTS):
                    if other_id != speaker_id:
                        agent_conv_history[other_id] += TOKEN_OUT_SPEAKING

                # Reaction (parallel)
                for reactor_id in range(NUM_AGENTS):
                    if reactor_id != speaker_id:
                        model = assigned_models[reactor_id]
                        reactor_input = (TOKEN_STATIC + agent_think[reactor_id] +
                                       agent_conv_history[reactor_id] +
                                       moderator_summary_accumulated + TOKEN_TASK_THINK)
                        total_input_by_model[model["name"]] += reactor_input
                        total_output_by_model[model["name"]] += TOKEN_OUT_REACTION

                        if scenario == "optimistic":
                            cached = agent_prev_prefix[reactor_id]
                            current_prefix = TOKEN_STATIC + agent_conv_history[reactor_id] + agent_think[reactor_id] + moderator_summary_accumulated
                            delta = current_prefix - cached
                            non_cached = delta + TOKEN_TASK_THINK
                            total_cached_by_model[model["name"]] += cached
                            total_non_cached_by_model[model["name"]] += non_cached
                        else:
                            total_non_cached_by_model[model["name"]] += reactor_input

                        agent_think[reactor_id] += TOKEN_OUT_REACTION
                        agent_prev_prefix[reactor_id] = TOKEN_STATIC + agent_conv_history[reactor_id] + agent_think[reactor_id] + moderator_summary_accumulated

            # Moderator summarizes after round (sequential)
            moderator_input = (TOKEN_STATIC + round_speeches_total +
                             moderator_summary_accumulated + TOKEN_TASK_MODERATOR)
            total_input_by_model[moderator_model["name"]] += moderator_input
            total_output_by_model[moderator_model["name"]] += TOKEN_OUT_MODERATOR

            if moderator_prev_prefix == 0:
                total_non_cached_by_model[moderator_model["name"]] += moderator_input
            else:
                cached = moderator_prev_prefix
                current_prefix = TOKEN_STATIC + moderator_summary_accumulated
                delta = round_speeches_total + (moderator_summary_accumulated - (moderator_prev_prefix - TOKEN_STATIC))
                non_cached = delta + TOKEN_TASK_MODERATOR
                total_cached_by_model[moderator_model["name"]] += cached
                total_non_cached_by_model[moderator_model["name"]] += non_cached

            moderator_summary_accumulated += TOKEN_OUT_MODERATOR
            moderator_prev_prefix = TOKEN_STATIC + moderator_summary_accumulated

            # Reflection (parallel)
            for agent_id in range(NUM_AGENTS):
                model = assigned_models[agent_id]
                reflect_input = (TOKEN_STATIC + agent_think[agent_id] +
                               agent_conv_history[agent_id] +
                               moderator_summary_accumulated + TOKEN_TASK_REFLECTION)
                total_input_by_model[model["name"]] += reflect_input
                total_output_by_model[model["name"]] += TOKEN_OUT_REFLECTION

                if scenario == "optimistic":
                    cached = agent_prev_prefix[agent_id]
                    current_prefix = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id] + moderator_summary_accumulated
                    delta = current_prefix - cached
                    non_cached = delta + TOKEN_TASK_REFLECTION
                    total_cached_by_model[model["name"]] += cached
                    total_non_cached_by_model[model["name"]] += non_cached
                else:
                    total_non_cached_by_model[model["name"]] += reflect_input

                agent_think[agent_id] += TOKEN_OUT_REFLECTION
                agent_prev_prefix[agent_id] = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id] + moderator_summary_accumulated

        # Phase 4: Final (parallel)
        for agent_id in range(NUM_AGENTS):
            model = assigned_models[agent_id]
            final_input = (TOKEN_STATIC + agent_think[agent_id] +
                          agent_conv_history[agent_id] +
                          moderator_summary_accumulated + TOKEN_TASK_FINAL)
            total_input_by_model[model["name"]] += final_input
            total_output_by_model[model["name"]] += TOKEN_OUT_FINAL

            if scenario == "optimistic":
                cached = agent_prev_prefix[agent_id]
                current_prefix = TOKEN_STATIC + agent_conv_history[agent_id] + agent_think[agent_id] + moderator_summary_accumulated
                delta = current_prefix - cached
                non_cached = delta + TOKEN_TASK_FINAL
                total_cached_by_model[model["name"]] += cached
                total_non_cached_by_model[model["name"]] += non_cached
            else:
                total_non_cached_by_model[model["name"]] += final_input

    cost_cached = sum((total_cached_by_model[m["name"]] / 1_000_000) * m["cached_input"] for m in model_list)
    cost_non_cached = sum((total_non_cached_by_model[m["name"]] / 1_000_000) * m["input"] for m in model_list)
    cost_input = cost_cached + cost_non_cached
    cost_output = sum((total_output_by_model[m["name"]] / 1_000_000) * m["output"] for m in model_list)
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.4 (planned)",
        "scenario": scenario,
        "description": "개별 세션, 다중 모델, 사회자",
        "models": {m["name"]: {
            "input_tokens": total_input_by_model[m["name"]],
            "cached_tokens": total_cached_by_model[m["name"]],
            "non_cached_tokens": total_non_cached_by_model[m["name"]],
            "output_tokens": total_output_by_model[m["name"]]
        } for m in model_list},
        "total_input_tokens": sum(total_input_by_model.values()),
        "cached_input_tokens": sum(total_cached_by_model.values()),
        "non_cached_input_tokens": sum(total_non_cached_by_model.values()),
        "total_output_tokens": sum(total_output_by_model.values()),
        "cost_cached_input": round(cost_cached, 2),
        "cost_non_cached_input": round(cost_non_cached, 2),
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total / SAMPLES_PER_LV, 4)
    }


# =============================================================================
# Main
# =============================================================================

def print_scenario_results(scenario: str):
    """Print results for a single scenario."""
    print(f"\n{'=' * 80}")
    print(f"시나리오: {scenario.upper()}")
    if scenario == "conservative":
        print("  - 병렬 호출: 캐시 없음")
        print("  - 순차 호출 (speaking): 캐시 적용")
    else:
        print("  - 모든 후속 호출: 이전 prefix 캐시 적용")
    print("=" * 80)

    # Lv.1 (shared session, same for both scenarios)
    res1 = lv1()
    print(f"\n[{res1['level']}] {res1['description']}")
    print(f"  Total: ${res1['cost_total']} | Avg/sample: ${res1['avg_cost_per_sample']}")

    # Lv.2
    res2 = lv2(scenario=scenario)
    cache_ratio = res2['cached_input_tokens'] / res2['total_input_tokens'] * 100
    print(f"\n[{res2['level']}] {res2['description']}")
    print(f"  Cache ratio: {cache_ratio:.1f}%")
    print(f"  Total: ${res2['cost_total']} | Avg/sample: ${res2['avg_cost_per_sample']}")

    # Lv.3
    res3 = lv3(scenario=scenario)
    cache_ratio = res3['cached_input_tokens'] / res3['total_input_tokens'] * 100
    print(f"\n[{res3['level']}] {res3['description']}")
    print(f"  Cache ratio: {cache_ratio:.1f}%")
    print(f"  Total: ${res3['cost_total']} | Avg/sample: ${res3['avg_cost_per_sample']}")

    # Lv.4
    res4 = lv4(scenario=scenario)
    cache_ratio = res4['cached_input_tokens'] / res4['total_input_tokens'] * 100
    print(f"\n[{res4['level']}] {res4['description']}")
    print(f"  Cache ratio: {cache_ratio:.1f}%")
    print(f"  Total: ${res4['cost_total']} | Avg/sample: ${res4['avg_cost_per_sample']}")

    total = res1['cost_total'] + res2['cost_total'] + res3['cost_total'] + res4['cost_total']
    print(f"\n  전체 합계: ${round(total, 2)}")

    return res1, res2, res3, res4


if __name__ == "__main__":
    print("=" * 80)
    print("포용적 주민참여 LLM 다중 에이전트 시뮬레이션 - API 비용 예측")
    print(f"설정: {NUM_AGENTS}명 에이전트, {ROUNDS_PER_DEBATE}라운드, {SAMPLES_PER_LV}샘플/레벨")
    print("=" * 80)

    api_calls_per_sim = (
        NUM_AGENTS +  # narrative
        NUM_AGENTS +  # initial
        NUM_AGENTS * ROUNDS_PER_DEBATE +  # speaking
        (NUM_AGENTS - 1) * NUM_AGENTS * ROUNDS_PER_DEBATE +  # reaction
        NUM_AGENTS * ROUNDS_PER_DEBATE +  # reflection
        NUM_AGENTS  # final
    )
    print(f"API 호출 횟수/시뮬레이션: {api_calls_per_sim}회")

    # Run both scenarios
    cons_results = print_scenario_results("conservative")
    opt_results = print_scenario_results("optimistic")

    # Summary comparison
    print("\n" + "=" * 80)
    print("시나리오 비교 요약")
    print("=" * 80)

    cons_total = sum(r['cost_total'] for r in cons_results)
    opt_total = sum(r['cost_total'] for r in opt_results)

    print(f"  Conservative (보수적): ${round(cons_total, 2)}")
    print(f"  Optimistic (낙관적):   ${round(opt_total, 2)}")
    print(f"  실측 예상 범위:        ${round(opt_total, 2)} ~ ${round(cons_total, 2)}")
    print(f"  (실측 캐시 비율 ~56%는 두 시나리오 사이)")
