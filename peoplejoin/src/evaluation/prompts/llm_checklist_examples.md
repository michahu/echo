# Examples

## Conversation 1
Conversation Date: Thursday, Feb 8

Bot-to-Alice Anastasiou: Hello, I'm here to help you as your Agent.
Alice Anastasiou-to-Bot: How many aircrafts have distance between 1000 and 5000?
Bot-to-Alice Anastasiou: Hi Alice, there are 8 aircrafts with distance between 1000 and 5000. Let me know if you need any more information.
Alice Anastasiou-to-Bot: No, that would be all.

## Output 1
```
criteria: Bot informs the user that the answer to their original request is [[8]]. Exact format of the information does not matter. Bot does not provide any untrue or misleading extra information.
rationale: Bot told user that there are 8 aircrafts with distance between 1000 and 5000, which is same as the expected answer of 9.
score: 2
```

## Conversation 2
Conversation Date: January 02, Thursday

Bot-to-Alice Anastasiou: Hello, I'm here to help you as your Agent.
Alice Anastasiou-to-Bot: What are the names of musicals who have no actors?
Bot-to-Alice Anastasiou: Hi Alice, I couldn't find enough information in your documents to compile that directly. I can reach out to Ruwaidah and Hannah who may have more information. Sounds good?
Alice Anastasiou-to-Bot: Yes, please do.
Bot-to-Ruwaidah Fakhoury: Hi Ruwaidah, Can you please tell me the names of musicals who have no actors?
Ruwaidah Fakhoury-to-Bot: I do not have information about musicals with no actors. I have information about actors and their associated musicals.
Bot-to-Hannah Jarvis: Hi Hannah, Can you please tell me the names of musicals who have no actors?
Hannah Jarvis-to-Bot: I'm sorry, I don't have information on musicals without actors.
Bot-to-Alice Anastasiou: Hi Alice, I contacted Ruwaidah and Hannah, but unfortunately, they do not have information about musicals with no actors. Let me know if there's anything else I can assist you with!
Alice Anastasiou-to-Bot: No, that's all for now. Thank you.

## Output 2
```
criteria: Bot informs the user that the answer to their original request is [["Wicked"], ["Rent"], ["Chicago"]]. Exact format of the information does not matter. Bot does not provide any untrue or misleading extra information.
rationale: Bot told user that it could not find the requested information, which is different from the expected list of musicals -- "Wicked" , "Rent", "Chicago". 
score: 0
```

## Conversation 3
Conversation Date: January 02, Thursday

Bot-to-Alice Anastasiou: Hello, I'm here to help you as your Agent.
Alice Anastasiou-to-Bot: What are the names of musicals who have no actors?
Bot-to-Alice Anastasiou: Hi Alice, I couldn't find enough information in your documents to compile that directly. I can reach out to Ruwaidah and Hannah who may have more information. Sounds good?
Alice Anastasiou-to-Bot: Yes, please do.
Bot-to-Ruwaidah Fakhoury: Hi Ruwaidah, Can you please tell me the names of musicals who have no actors?
Ruwaidah Fakhoury-to-Bot: Wicked, Domino and Rent are such musicals
Bot-to-Hannah Jarvis: Hi Hannah, Can you please tell me the names of musicals who have no actors?
Hannah Jarvis-to-Bot: I'm sorry, I don't have information on musicals without actors.
Bot-to-Alice Anastasiou: Hi Alice, I contacted Ruwaidah and Hannah. Wicked, Domino and Rent are musicals who have no actors. Let me know if there's anything else I can assist you with!
Alice Anastasiou-to-Bot: No, that's all for now. Thank you.

## Output 3
```
criteria: Bot informs the user that the answer to their original request is [["Wicked"], ["Rent"], ["Chicago"]]. Exact format of the information does not matter. Bot does not provide any untrue or misleading extra information.
rationale: Bot told user that "Wicked", "Domino" and "Rent" is the answer. "Wicket" and "Rent" are correct but Bot did not mention "Chicago", so bot did not provide the entire correct answer. Also, "Domino" is incorrect as per the criteria, so bot gave untrue extra information.
score: 1
```