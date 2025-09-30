# Instructions
- Given the user description, the messages history, and user documents, generate the next user response.
- Respond with answer the question appropriately based on the the description, message history, and user documents.
- Respond with to-the-point statements. Format the text as needed.
- Respond with a single line message. Always end the message with a <eos>.
- Respond with 'skip' if bot utterance does not seem to need a response. *Do not* respond 'skip' after the bot greets you; instead, state your request.

# Examples

## Example
### User Description: Is interested in compiling a summary about the impact of climate change on the economy.
### Info: Today's date is Thu, September 5, 05:53 PM .
### User Documents:
#### Document:
Title: Swift set to play halftime show
Content: Taylor Swift is set to play the superbowl halftime show this year. She is expected to perform a medley of her greatest hits.
### Messages History:
Bot: Hello, I'm here to help you as your Agent
Alice: I'm interested in compiling a summary on the topic "The impact of climate change on the economy"
Bot: Hi Alice, I couldn't find enough information in your documents to compile that information directly. I can reach out to Bhushan and Cassie who may have more information. Sounds good?
### Next Response:
Alice: Sounds great! <eos>

## Example
### User Description: May have information relevant to the summary.
### Info: Today's date is Thu, September 5, 05:53 PM .
### User Documents:
#### Document:
Title: val/1174/21/4
Content: The automotive sector is expected to be heavily impacted by climate change. The sector is expected to see a shift towards electric vehicles and sustainable practices.
#### Document:
Title: val/1174/21/3
Content: Peter Piper picked a peck of pickled peppers. Indeed, it was a peck of pickled peppers that Peter Piper picked.
### Messages History:
Bot: Hi Bhushan, do you have any information about the impact of climate change on the economy?
### Next Response:
Bhushan: Yes, here's a relevant document I found: "The automotive sector is expected to be heavily impacted by climate change. The sector is expected to see a shift towards electric vehicles and sustainable practices." <eos>

## Example
### User Description: May have information relevant to the summary.
### Info: Today's date is Thu, September 5, 05:53 PM .
### User Documents:
#### Document:
Title: val/1174/21/4
Content: The automotive sector is expected to be heavily impacted by climate change. The sector is expected to see a shift towards electric vehicles and sustainable practices.
#### Document:
Title: val/1174/21/3
Content: Peter Piper picked a peck of pickled peppers. Indeed, it was a peck of pickled peppers that Peter Piper picked.
### Messages History:
Bot: Hi Bhushan, do you have any information about the fishing industry?
### Next Response:
Bhushan: Sorry, I'm afraid I don't have any information. <eos>

# Task
## Example