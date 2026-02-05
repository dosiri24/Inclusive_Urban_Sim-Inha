# Task Templates
# All task instructions for debate simulation.
# Each function returns a complete task string with response format.


# Generate narrative background task.
def get_narrative_task() -> str:
    return """당신은 주안2 미추5구역에 사는 주민입니다. 주어진 페르소나를 바탕으로 다음 질문에 대해 자연스럽게 이야기하세요.

1. 이 동네에 어떻게 오게 되었나요?
2. 재개발에 대해 어디서, 누구에게, 어떻게 들었나요?
3. '비례율', '분담금', '조합', '감정평가' 같은 재개발 용어를 들어본 적 있나요? 어느 정도 이해하고 있나요?
4. 재개발이 되면 당신의 생활은 어떻게 바뀔 것 같나요?
5. 구체적으로 어떤 일을 하고, 대략적인 하루 일과가 어떻게 되나요?
6. 이 동네에서 자주 만나는 이웃이 있나요? 혹은 동네 모임이나 반상회에 참여하나요?
7. 여러 사람이 모여서 의견이 갈릴 때 당신은 보통 어떻게 행동하나요?
8. 재개발 외에 요즘 가장 신경 쓰이는 생활 문제가 있나요? (예: 자녀교육, 건강, 경제적 문제 등)

응답 형식:
{"생각": "당신의 이야기"}"""


# Generate initial opinion task (MODE 4).
def get_initial_task() -> str:
    return """토론이 시작되기 전입니다. 아직 다른 주민의 의견을 듣지 않은 상태에서, 당신의 페르소나와 상황을 바탕으로 미추5구역 촉진계획 세부안에 대한 당신의 입장과 조건, 우려 등을 이야기해주세요.

응답 형식:
{"입장": "찬성/조건부찬성/조건부반대/반대 중 택1", "생각": "당신의 이유와 생각"}"""


# Generate speaking turn task (MODE 1).
def get_speaking_task(round_num: int) -> str:
    return f"""현재 {round_num}라운드입니다. 당신의 발화 차례입니다.

응답 형식:
{{"발화": "당신의 의견", "지목": [{{"대상": "resident_XX", "입장": "공감/비판/인용/질문"}}] 또는 []}}

지목 규칙:
- 특정 주민의 의견을 언급할 때 해당 주민을 지목
- 자신의 답변에서 여러 주민을 언급하면 여러 번 지목 가능 (예: [{{"대상": "resident_01", "입장": "공감"}}, {{"대상": "resident_05", "입장": "비판"}}])
- 반드시 여러 명을 지목할 필요는 없음. 지목 없이 일반적인 의견만 말해도 됨 (지목: [])"""


# Generate thinking reaction task (MODE 2).
def get_think_task(listener_id: str, speaker_id: str, response_code: str, speech: str) -> str:
    speech_preview = speech[:100] if len(speech) > 100 else speech
    return f"""당신은 {listener_id}입니다. 대화기록에서 {listener_id}의 발언은 당신 자신의 발언입니다.

{speaker_id}의 발화(코드: {response_code}):
"{speech_preview}"

상대방이 왜 그렇게 말했는지, 그리고 당신({listener_id})은 어떻게 생각하는지 정리하세요.

응답 형식:
{{"상대의견": "{response_code}", "반응유형": "공감/비판/인용/질문/무시 중 택1", "생각": "당신의 생각"}}"""


# Generate round reflection task (MODE 3).
def get_reflection_task(resident_id: str, round_num: int) -> str:
    return f"""당신은 {resident_id}입니다. 대화기록에서 {resident_id}의 발언은 당신 자신의 발언입니다.

{round_num}라운드가 끝났습니다. 지금까지 들은 의견 중 가장 기억에 남는 것과 그 이유를 당신({resident_id})의 관점에서 정리하세요.

응답 형식:
{{"생각": "당신의 생각"}}"""


# Generate final opinion task (MODE 4).
def get_final_task() -> str:
    return """토론이 모두 끝났습니다. 다른 주민들의 의견을 모두 들은 지금, 미추5구역 촉진계획 세부안에 대한 당신의 최종 입장은 어떻습니까?

응답 형식:
{"입장": "찬성/조건부찬성/조건부반대/반대 중 택1", "생각": "당신의 최종 생각과 이유, 조건 등"}"""


# =============================================================================
# Lv.1 Batch Tasks (single session, batch output)
# =============================================================================


def get_lv1_narrative_task(participants: str) -> str:
    """Generate batch narrative task for all participants at once."""
    return f"""당신은 주안2동 미추5구역 재개발 토론 시뮬레이션을 진행하는 역할입니다.
아래 참여자 목록의 각 주민 입장에서, 그 페르소나에 맞게 다음 질문에 답하세요.

[참여자 목록]
{participants}

[질문]
1. 이 동네에 어떻게 오게 되었나요?
2. 재개발에 대해 어디서, 누구에게, 어떻게 들었나요?
3. '비례율', '분담금', '조합', '감정평가' 같은 재개발 용어를 들어본 적 있나요?
4. 재개발이 되면 당신의 생활은 어떻게 바뀔 것 같나요?
5. 이 동네에서 자주 만나는 이웃이 있나요?
6. 여러 사람이 모여서 의견이 갈릴 때 당신은 보통 어떻게 행동하나요?

각 주민의 페르소나(연령, 직업, 자가여부 등)에 맞게 자연스럽게 서술하세요.

응답 형식 (JSON 배열):
[
  {{"resident_id": "resident_01", "서사": "..."}},
  {{"resident_id": "resident_02", "서사": "..."}},
  ...
]"""


def get_lv1_initial_task(participants: str, narratives: str) -> str:
    """Generate batch initial opinion task."""
    return f"""토론이 시작되기 전입니다. 각 주민의 페르소나와 서사를 바탕으로,
미추5구역 촉진계획 세부안에 대한 초기 입장을 작성하세요.

[참여자 목록]
{participants}

[각 주민의 서사]
{narratives}

각 주민은 아직 다른 주민의 의견을 듣지 않은 상태입니다.
페르소나에 맞게 찬성/조건부찬성/조건부반대/반대 중 하나를 선택하고 이유를 작성하세요.

응답 형식 (JSON 배열):
[
  {{"resident_id": "resident_01", "입장": "찬성/조건부찬성/조건부반대/반대 중 택1", "생각": "..."}},
  {{"resident_id": "resident_02", "입장": "...", "생각": "..."}},
  ...
]"""


def get_lv1_speaking_task(
    round_num: int,
    participants: str,
    initial_opinions: str,
    previous_speeches: str = ""
) -> str:
    """Generate batch speaking task for one round."""
    prev_section = ""
    if previous_speeches:
        prev_section = f"""
[이전 라운드 발화]
{previous_speeches}
"""

    return f"""{round_num}라운드입니다. 각 주민이 순서대로 발언합니다.

[참여자 목록]
{participants}

[초기 입장]
{initial_opinions}
{prev_section}
[라운드 {round_num} 지침]
- 1라운드: 간단한 자기소개 후 계획안에 대한 첫 의견
- 2라운드: 다른 주민 발언에 대한 반응, 질문, 동의/반대
- 3라운드: 토론을 통해 바뀐 생각이나 유지되는 입장 정리

각 주민은 자신의 페르소나와 초기 입장을 바탕으로 발언하세요.
다른 주민의 의견을 언급할 경우 지목(공감/비판/인용/질문)하세요.

응답 형식 (JSON 배열):
[
  {{"resident_id": "resident_01", "발화": "...", "지목": [{{"대상": "resident_XX", "입장": "공감/비판/인용/질문"}}] 또는 []}},
  {{"resident_id": "resident_02", "발화": "...", "지목": [...]}},
  ...
]"""


def get_lv1_final_task(
    participants: str,
    initial_opinions: str,
    all_speeches: str
) -> str:
    """Generate batch final opinion task."""
    return f"""토론이 모두 끝났습니다. 각 주민의 최종 입장을 작성하세요.

[참여자 목록]
{participants}

[초기 입장]
{initial_opinions}

[토론 내용]
{all_speeches}

각 주민은 토론을 통해 들은 의견을 바탕으로 최종 입장을 정하세요.
처음과 달라졌다면 무엇 때문인지 포함하세요.

응답 형식 (JSON 배열):
[
  {{"resident_id": "resident_01", "입장": "찬성/조건부찬성/조건부반대/반대 중 택1", "생각": "..."}},
  {{"resident_id": "resident_02", "입장": "...", "생각": "..."}},
  ...
]"""
