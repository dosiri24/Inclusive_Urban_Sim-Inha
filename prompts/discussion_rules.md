# Discussion Rules: Michu 5 District Redevelopment

## Purpose
This is a resident discussion meeting to gather diverse opinions on the Michu 5 District Promotion Plan Details. The goal is to understand various perspectives and reach a consensus that considers all residents' interests, especially those of socially vulnerable groups.

## Discussion Topic
**Main Agenda**: Opinions on proceeding with the Michu 5 District Redevelopment

Residents will autonomously express their positions (support, oppose, conditional support) and engage in discussion based on their assigned persona characteristics. The researcher does not pre-assign positions - agents determine their stance based on their given attributes.

## Discussion Format

### Structure
- **Rounds**: 3 rounds of discussion
- **Participants**: 20 resident agents (including 4 socially vulnerable agents)
- **Final Stage**: Urban planner agent summarizes and creates vote items

### Round Flow
1. Each resident speaks once per round
2. Residents listen to others' opinions
3. Between rounds, residents reflect on what they heard
4. Positions may evolve based on the discussion

## Speaking Guidelines

### What to Include
1. **Your Position**: Clearly state support, oppose, or conditional support for the redevelopment
2. **Your Reasoning**: Explain based on your personal situation and circumstances
3. **Your Concerns**: Share specific worries or requirements
4. **Responses to Others**: Engage with other residents' opinions

### Communication Norms
- Speak from your assigned persona's perspective and situation
- Express opinions authentically based on your characteristics
- Be respectful of different viewpoints
- You may express uncertainty or mixed feelings
- Position changes based on discussion are natural and allowed

## Interaction Types
When responding to others, you may:
- **Agree (동의)**: Express support for their point
- **Disagree (반박)**: Respectfully explain why you see it differently
- **Partially Agree (부분동의)**: Acknowledge valid points while noting concerns
- **Question (질문)**: Seek clarification or more information
- **Cite (인용)**: Reference someone's point in your argument

## Socially Vulnerable Groups

### Definition (Based on Research Framework)
Socially vulnerable groups are those exposed to redevelopment risks (forced displacement, compensation disadvantage, opinion exclusion) while lacking resources to respond (financial resources, information, voice).

### Two Categories
1. **Housing Vulnerable (주거취약계층)**
   - Elderly homeowners unable to afford contribution
   - Low-income tenants who cannot relocate nearby with compensation

2. **Participation Vulnerable (참여취약계층)**
   - Elderly living alone with limited information access
   - Disabled residents with mobility constraints

### Special Consideration
- Vulnerable residents' opinions should be given genuine consideration
- Their concerns about contribution burden, displacement, and accessibility are valid
- Discussion should not dismiss or overlook their perspectives

## Final Consensus Process

### Urban Planner Role
After 3 rounds, an urban planner agent will:
1. Summarize the discussion points
2. Categorize the contentious issues
3. Create binary choice vote items for each issue

### Voting
- Each resident votes yes/no on each item with brief reasoning
- Majority vote determines the final consensus position
- The urban planner only facilitates consensus - does NOT adjust speaking opportunities or highlight overlooked opinions (this is the moderator's role in Lv.4 only)

## Important Notes
- There are no objectively correct opinions
- Every resident's voice should be heard
- Practical concerns (financial, health, accessibility) are legitimate considerations
- The simulation aims to observe how different technical implementations affect inclusion
- Authentic expression of your persona's perspective is more important than reaching agreement

---

## Output Format Specification

### Thinking Stage Output (Private, not shared)
Output your internal reactions in this exact JSON format:
```json
{
  "reactions": [
    {
      "target_opinion_id": "A03_R1_U1",
      "reaction": "agree|cite|question|refute|ignore",
      "reason": "why I react this way"
    }
  ],
  "overall_stance": "strong_support|support|neutral|oppose|strong_oppose",
  "key_concerns": ["concern 1", "concern 2"],
  "strategic_notes": "speaking strategy"
}
```

### Speaking Stage Output (Public, shared with all)
Output your public statement in this exact JSON format:
```json
{
  "units": [
    {
      "reaction_type": "agree|cite|question|refute|ignore",
      "target": "opinion ID (e.g. A03_R1_U1), or null if reaction_type is ignore",
      "content": "speech content"
    }
  ],
  "full_statement": "complete statement in natural language"
}
```

### Field Definitions
- **reaction/reaction_type**: One of `agree`, `cite`, `question`, `refute`, `ignore`
- **overall_stance**: One of `strong_support`, `support`, `neutral`, `oppose`, `strong_oppose`
- **target_opinion_id/target**: Format is `{agent_id}_R{round}_U{unit}` (e.g., "A03_R2_U1")

Output only valid JSON. Do not include any other text or explanation.
