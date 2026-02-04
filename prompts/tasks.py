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
{{"발화": "당신의 의견", "지목": "resident_XX 또는 null", "입장": "공감/비판/인용/질문 중 택1 또는 null"}}

- 지목: 특정 주민에게 말할 때 resident_XX, 아니면 null
- 입장: 지목한 주민의 의견에 대한 태도. 지목이 null이면 입장도 null"""


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
