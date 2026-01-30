"""
포용적 주민참여를 위한 LLM 다중 에이전트 시뮬레이션 기술적 구현에 따른 효과 비교
연구비용(정확히는 api price) 예측 모듈

연구 조건
Lv.1. 단일 세션, 단일 모델 (gemini-3-flash-preview)
Lv.2-1. 개별 세션, 단일 모델 (gemini-3-flash-preview)
Lv.2-2. 개별 세션, 단일 모델 (LGAI-EXAONE/K-EXAONE-236B-A23B)
Lv.3. 개별 세션, 다중 모델 (gemini-3-flash-preview + LGAI-EXAONE/K-EXAONE-236B-A23B 
+ kimi-k2-0905-preview + Claude Haiku 4.5 + GPT-5 mini)
Lv.4. 개별세션, 다중 모델, 무작위 모델 사회자 (Lv.3과 동일 모델군)
"""

import random

### 실험 설계 변수들
NUM_AGENTS = 20 # 에이전트 수
SAMPLES_PER_LV = 30 # 각 연구 조건별 샘플 수
ROUNDS_PER_DEBATE = 3 # 토론 라운드 수

TOKEN_SYSTEM_PROMPT = 500 # 에이전트 배경, 토론 의제 및 규칙 등
TOKEN_PERSONA = 1500 # 에이전트 성격
TOKEN_LOCAL = 1500 # 지역 맥락 정보
TOKEN_FIXED = TOKEN_SYSTEM_PROMPT + TOKEN_PERSONA + TOKEN_LOCAL # 고정 토큰 수

TOKEN_AGENT_OUTPUT = 400 # 에이전트 의견 제시 토큰 수
TOKEN_AGENT_THINK = 200 # 상대의견별 에이전트 사고 과정 토큰 수
TOKEN_MODERATOR_OUTPUT = 600 # 사회자 의견 정리 및 요약 토큰 수
TOKEN_SHARED_OUTPUT = TOKEN_AGENT_OUTPUT # 다른 에이전트에게 공유되는 에이전트 의견 토큰 수

### 모델별 토큰당 가격 (USD per 1M tokens)
MODELS = {
    "gemini-3-flash-preview": {
        "name": "gemini-3-flash-preview",
        "input": 0.50, 
        "output": 3.00
    },
    "LGAI-EXAONE": {
        "name": "LGAI-EXAONE/K-EXAONE-236B-A23B",
        "input": 0.60,
        "output": 1.00
    },
    "kimi-k2-0905-preview": {
        "name": "kimi-k2-0905-preview",
        "input": 0.60,
        "output": 2.50
    },
    "Claude-Haiku-4.5": {
        "name": "Claude Haiku 4.5",
        "input": 1.00,
        "output": 5.00
    },
    "GPT-5-mini": {
        "name": "GPT-5-mini",
        "input": 0.25,
        "output": 2.00

    },
}

def lv1():
    ## Lv.1. 단일 세션, 단일 모델 (gemini-3-flash-preview)

    model = MODELS["gemini-3-flash-preview"]
    input_price = model["input"]
    output_price = model["output"]

    total_input_tokens = 0
    total_output_tokens = 0
    
    per_agent_output = TOKEN_AGENT_OUTPUT + TOKEN_AGENT_THINK*(NUM_AGENTS-1)
    
    for debate in range(SAMPLES_PER_LV):
        prev_outputs = 0

        for round_idx in range(ROUNDS_PER_DEBATE):
            # 단일 세션: 컨텍스트 1회만, 이전 라운드 출력 누적
            round_input = TOKEN_FIXED if round_idx == 0 else TOKEN_FIXED + prev_outputs
            round_output = per_agent_output*NUM_AGENTS

            total_input_tokens += round_input
            total_output_tokens += round_output

            prev_outputs += TOKEN_SHARED_OUTPUT*NUM_AGENTS

    cost_input = (total_input_tokens/1_000_000)*input_price
    cost_output = (total_output_tokens/1_000_000)*output_price
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.1",
        "model": model["name"],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total/SAMPLES_PER_LV, 2),
        "models": {model["name"]: {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}}
    }


def lv2_1():
    ## Lv.2-1. 개별 세션, 단일 모델 (gemini-3-flash-preview)

    model = MODELS["gemini-3-flash-preview"]
    input_price = model["input"]
    output_price = model["output"]

    total_input_tokens = 0
    total_output_tokens = 0

    per_agent_output = TOKEN_AGENT_OUTPUT + TOKEN_AGENT_THINK*(NUM_AGENTS-1)

    for debate in range(SAMPLES_PER_LV):
        prev_outputs = 0

        for round_idx in range(ROUNDS_PER_DEBATE):
            # 에이전트별 개별 세션: 이전 라운드 출력 누적을 입력으로 사용
            round_input_per_agent = TOKEN_FIXED + prev_outputs
            round_output_per_agent = per_agent_output

            total_input_tokens += round_input_per_agent*NUM_AGENTS
            total_output_tokens += round_output_per_agent*NUM_AGENTS

            prev_outputs += TOKEN_SHARED_OUTPUT*NUM_AGENTS

    cost_input = (total_input_tokens/1_000_000)*input_price
    cost_output = (total_output_tokens/1_000_000)*output_price
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.2-1",
        "model": model["name"],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total/SAMPLES_PER_LV, 2),
        "models": {model["name"]: {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}}
    }


def lv2_2():
    ## Lv.2-2. 개별 세션, 단일 모델 (LGAI-EXAONE/K-EXAONE-236B-A23B)

    model = MODELS["LGAI-EXAONE"]
    input_price = model["input"]
    output_price = model["output"]

    total_input_tokens = 0
    total_output_tokens = 0

    per_agent_output = TOKEN_AGENT_OUTPUT + TOKEN_AGENT_THINK*(NUM_AGENTS-1)

    for debate in range(SAMPLES_PER_LV):
        prev_outputs = 0

        for round_idx in range(ROUNDS_PER_DEBATE):
            round_input_per_agent = TOKEN_FIXED + prev_outputs
            round_output_per_agent = per_agent_output

            total_input_tokens += round_input_per_agent*NUM_AGENTS
            total_output_tokens += round_output_per_agent*NUM_AGENTS

            prev_outputs += TOKEN_SHARED_OUTPUT*NUM_AGENTS

    cost_input = (total_input_tokens/1_000_000)*input_price
    cost_output = (total_output_tokens/1_000_000)*output_price
    cost_total = cost_input + cost_output

    return {
        "level": "Lv.2-2",
        "model": model["name"],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total/SAMPLES_PER_LV, 2),
        "models": {model["name"]: {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}}
    }


def lv3():
    ## Lv.3. 개별 세션, 다중 모델 (무작위 배정)

    model_list = list(MODELS.values())

    total_input_by_model = {m["name"]: 0 for m in model_list}
    total_output_by_model = {m["name"]: 0 for m in model_list}

    per_agent_output = TOKEN_AGENT_OUTPUT + TOKEN_AGENT_THINK*(NUM_AGENTS-1)

    for debate in range(SAMPLES_PER_LV):
        # 에이전트별 모델 무작위 배정
        assigned = [random.choice(model_list) for _ in range(NUM_AGENTS)]

        prev_outputs = 0

        for round_idx in range(ROUNDS_PER_DEBATE):
            round_input_per_agent = TOKEN_FIXED + prev_outputs
            round_output_per_agent = per_agent_output

            for model in assigned:
                total_input_by_model[model["name"]] += round_input_per_agent
                total_output_by_model[model["name"]] += round_output_per_agent

            prev_outputs += TOKEN_SHARED_OUTPUT*NUM_AGENTS

    cost_input = 0
    cost_output = 0

    for m in model_list:
        cost_input += (total_input_by_model[m["name"]]/1_000_000)*m["input"]
        cost_output += (total_output_by_model[m["name"]]/1_000_000)*m["output"]

    cost_total = cost_input + cost_output

    return {
        "level": "Lv.3",
        "models": {m["name"]: {"input_tokens": total_input_by_model[m["name"]], "output_tokens": total_output_by_model[m["name"]]} for m in model_list},
        "total_input_tokens": sum(total_input_by_model.values()),
        "total_output_tokens": sum(total_output_by_model.values()),
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total/SAMPLES_PER_LV, 2)
    }


def lv4():
    ## Lv.4. 개별세션, 다중 모델, 무작위 모델 사회자

    model_list = list(MODELS.values())

    total_input_by_model = {m["name"]: 0 for m in model_list}
    total_output_by_model = {m["name"]: 0 for m in model_list}

    per_agent_output = TOKEN_AGENT_OUTPUT + TOKEN_AGENT_THINK*(NUM_AGENTS-1)

    for debate in range(SAMPLES_PER_LV):
        assigned = [random.choice(model_list) for _ in range(NUM_AGENTS)]

        prev_outputs = 0

        for round_idx in range(ROUNDS_PER_DEBATE):
            # 에이전트 입력/출력
            round_input_per_agent = TOKEN_FIXED + prev_outputs
            round_output_per_agent = per_agent_output

            for model in assigned:
                total_input_by_model[model["name"]] += round_input_per_agent
                total_output_by_model[model["name"]] += round_output_per_agent

            round_output_total = TOKEN_SHARED_OUTPUT*NUM_AGENTS

            # 사회자 입력/출력 (무작위 모델)
            moderator_model = random.choice(model_list)
            moderator_input = TOKEN_FIXED + prev_outputs + round_output_total
            total_input_by_model[moderator_model["name"]] += moderator_input
            total_output_by_model[moderator_model["name"]] += TOKEN_MODERATOR_OUTPUT

            prev_outputs += round_output_total + TOKEN_MODERATOR_OUTPUT

    cost_input = 0
    cost_output = 0

    for m in model_list:
        cost_input += (total_input_by_model[m["name"]]/1_000_000)*m["input"]
        cost_output += (total_output_by_model[m["name"]]/1_000_000)*m["output"]

    cost_total = cost_input + cost_output

    return {
        "level": "Lv.4",
        "models": {m["name"]: {"input_tokens": total_input_by_model[m["name"]], "output_tokens": total_output_by_model[m["name"]]} for m in model_list},
        "total_input_tokens": sum(total_input_by_model.values()),
        "total_output_tokens": sum(total_output_by_model.values()),
        "cost_input": round(cost_input, 2),
        "cost_output": round(cost_output, 2),
        "cost_total": round(cost_total, 2),
        "avg_cost_per_sample": round(cost_total/SAMPLES_PER_LV, 2)
    }


if __name__ == "__main__":
    results = [lv1(), lv2_1(), lv2_2(), lv3(), lv4()]

    for res in results:
        print(f"{res['level']} | model(s): {res.get('model', 'mixed')} | total: ${res['cost_total']} | avg/sample: ${res['avg_cost_per_sample']}")

    overall_total = round(sum(res["cost_total"] for res in results), 2)
    print(f"Overall total cost (all levels): ${overall_total}")

    # 모델별 총 비용 집계 (Lv.1~Lv.4 전부)
    model_costs = {}
    for model in MODELS.values():
        model_costs[model["name"]] = {"cost_input": 0, "cost_output": 0, "cost_total": 0}

    for res in results:
        if "models" not in res:
            continue
        for model_name, tokens in res["models"].items():
            price = None
            for m in MODELS.values():
                if m["name"] == model_name:
                    price = m
                    break
            if price is None:
                continue

            cost_input = (tokens["input_tokens"]/1_000_000)*price["input"]
            cost_output = (tokens["output_tokens"]/1_000_000)*price["output"]
            cost_total = cost_input + cost_output

            model_costs[model_name]["cost_input"] += cost_input
            model_costs[model_name]["cost_output"] += cost_output
            model_costs[model_name]["cost_total"] += cost_total

    print("\nModel-wise total costs:")
    for model_name, costs in model_costs.items():
        print(f"- {model_name}: input ${round(costs['cost_input'],2)}, output ${round(costs['cost_output'],2)}, total ${round(costs['cost_total'],2)}")

