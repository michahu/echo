# Task
- {identifier} serves as a digital assistant for {primary_user}. It can access different tools to send messages to other users. It interprets the user's intent and context from the messages received and uses the appropriate tools to fulfill the user's request.
- You will be given a dialogue between {identifier} and various human users. These are individual conversations, where each message will be labeled by sender and recipient in the format: `sender-to-recipient: message`. Your job is to rate {identifier}'s response on a scale from 0 to 2, based on the rubric provided below.

## Scoring Scale
0. The response does not satisfy the criterion.
1. The response partially meets the criterion.
2. The response fully meets the criterion.

# Guidelines
- Format your output as follows using YAML:
```
criteria: <Copy of the criterion>
rationale: <Brief explanation of why you gave this score>
score: <Score between 0-2 based on how well {identifier}'s response meets the criterion>
```
- Base your evaluation solely on the given criteria.
- If the criterion is clearly satisfied without any ambiguity, assign a full score of 2.
- Valid scores are 0, 1, or 2 only. 
- Formatting of the response shouldn't affect the score.
- Extra details that do not mislead or contradict the answer should not lower the score.

{examples}

# Conversation
Conversation Date: {conversation_date}
{conversation}

# Output
```
criteria: {criteria}