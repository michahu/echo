# Instructions
- Given the user description, the messages history, and user documents, generate the next user response.
- Respond with answer the question appropriately based on the the description, message history, and user documents.
- Respond with to-the-point statements. Format the text as needed.
- Respond by being a helpful colleague. If the exact information is not available, then respond with what type of related information you could provide.
- Respond with a single line message. Always end the message with a <eos>
- Respond with 'skip' only if bot utterance is not a question

# Examples:

## Example
### User Description: Alice Anastasiou is an employee at company. She is interested in knowing the names and ages of the students who have a dog but do not have a cat as pet which may be available in her documents or notes or Bot might have to reach out to other people. When Bot initates interaction, start with 'I needs names and ages of the students who have a dog but do not have a cat as pet'.
### Info: Today's date is Jan 1st, 2022, Monday.
### User Documents:
#### Document: 
Title: has pet
Record 1: {'student id': 1001, 'pet id': 2001}
Record 2: {'student id': 1002, 'pet id': 2002}
Record 3: {'student id': 1002, 'pet id': 2003}
### Messages History:
Bot: Hello, I'm here to help you as your Agent
Alice: I needs names and ages of the students who have a dog but do not have a cat as pet
Bot: Hi Alice, I couldn't find enough information in your documents to compile that informatiom directly. Bhushan and Cassey might have more information. Who should I contact?
Alice: You can decide whom to contact.
Bot: Sounds good. I will reach out if I need any clarifications.
### Next Response:
Alice: skip <eos>

## Example
### User Description: Alice Anastasiou is an employee at company. She is interested in knowing the names and ages of the students who have a dog but do not have a cat as pet which may be available in her documents or notes or Bot might have to reach out to other people. When Bot initates interaction, start with 'I needs names and ages of the students who have a dog but do not have a cat as pet'.
### Info: Today's date is Jan 1st, 2022, Monday.
### User Documents:
#### Document: 
Title: has pet
Record 1: {'student id': 1001, 'pet id': 2001}
Record 2: {'student id': 1002, 'pet id': 2002}
Record 3: {'student id': 1002, 'pet id': 2003}
### Messages History:
Bot: Hello, I'm here to help you as your Agent
Alice: I needs names and ages of the students who have a dog but do not have a cat as pet
Bot: Hi Alice, I couldn't find enough information in your documents to compile that informatiom directly. I can reach out to Bhushan and Cassey who may have more information. Sounds good?
Alice: yes
Bot: Hi Alice, first name of students who have a dog but not a cat as pet:
- "first name": "Tracy"
Do you want to know their last names as well?
Alice: Thanks! I had also asked for their age.
Bot: Thanks for pointing that out. I found that Tracy's age is 19. Do you need any more information?
### Next Response:
Alice: No, that would be all. <eos>

## Example
### User Description: Bhushan Magar is an employee at company. Bhushan will provide Bot with specific relevant information if it is available in his documents or notes
### User Documents:
#### Document:
Title: student
Record 1: {'student id': 1001, 'major': 600}
Record 2: {'student id': 1002, 'major': 600}
Record 3: {'student id': 1003, 'major': 600}
### Messages History:
Bot: Hi Bhushan, Can you please tell how many total students are there in the university?
### Next Response:
Bhushan: I know about major of 3 students. I do not specifically know if that is the total count of the students in the university. <eos>

## Example
### User Description: Cassie Hicks is an employee at company. Cassie will provide Bot with specific relevant information if it is available in her documents or notes.
### Info: Today's date is Jan 1st, 2022, Monday.
### User Documents:
#### Document:
Collection name: pets
Title: pets
Record 1: {'pet id': 2001, 'pet type': 'cat', 'pet age': 3, 'weight': 12.0}
Record 2: {'pet id': 2002, 'pet type': 'dog', 'pet age': 2, 'weight': 13.4}
Record 3: {'pet id': 2003, 'pet type': 'dog', 'pet age': 1, 'weight': 9.3}
### Messages History:
Bot: Hi Cassie, can you please let me know which students have a dog as pet ? 
Cassie: I do not have that information. But I do know about pet ids and their types if you need that.
Bot: Thanks Cassie! Can you please let me know the type of animal are the following `pet ids`?
`pet id`: 2001
`pet id`: 2002
`pet id`: 2003
### Next Response:
Cassie: Sure. Please find details as follows: 
- `pet id`: 2001 -> cat
- `pet id`: 2002 -> dog 
- `pet id`: 2003 -> dog <eos>

## Example
### User Description: Bhushan Magar is an employee at company. Bhushan will provide Bot with specific relevant information if it is available in his documents or notes
### User Documents:
#### Document:
Title: student
Record 1: {'student id': 1001, 'last name': 'Smith', 'first name': 'Linda', 'sex': 'F', 'major': 600, 'advisor': 1121, 'city code': 'BAL'}
Record 2: {'student id': 1002, 'last name': 'Kim', 'first name': 'Tracy', 'sex': 'F', 'major': 600, 'advisor': 7712, 'city code': 'HKG'}
Record 3: {'student id': 1003, 'last name': 'Jones', 'first name': 'Shiela', 'sex': 'F', 'major': 600, 'advisor': 7792, 'city code': 'WAS'}
### Messages History:
Bot: Hi Bhushan, Can you please tell the list of students who have a cat as pet ?
Bhushan: Sorry I do not have information about pet types. I have information on student ids and associated personal and academic details of the students such as their name, sex, their major, etc. Let me know if you need any of that information. 
Bot: Hi Bhushan, Can you please tell the first name and age of `student id`: 1002 ?
### Next Response:
Bhushan: First name of student id 1002 is 'Tracy'. I do not know their age. <eos>

# Task
## Example