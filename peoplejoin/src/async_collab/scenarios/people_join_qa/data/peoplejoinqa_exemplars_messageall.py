infoseeking_spider_v2_a_redirection_splittables_messageall = """### Example ###

# Event: Alice Anastasiou says: I need first name and ages of the students who have a dog but do not have a cat as pet
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id":"alice@company.com", "email":"alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query='name and age of students with dogs but no cats as pets')
[
"Document:\nCollection name: pets 1\nTitle: pets\nRecord 1: {'pet id': 2001, 'pet type': 'cat', 'pet age': 3, 'weight': 12.0}\nRecord 2: {'pet id': 2002, 'pet type': 'dog', 'pet age': 2, 'weight': 13.4}\nRecord 3: {'pet id': 2003, 'pet type': 'dog', 'pet age': 1, 'weight': 9.3}\n"
]
>>> Reflection.thought("I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query='name and age of students with dogs but no cats as pets')
{
  "Cassie Hicks": "User might have information about students and their pet IDs.",
  "Dewei Peng": "The user might have information about students, including their personal details, demographics, and academic information.",
  "Hannah Jarvis": "Unknown."
}
>>> Reflection.thought("I will reach out to each contact: Cassie Hicks, Dewei Peng, and Hannah Jarvis.")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I want to reach out to Hannah Jarvis, Cassie Hicks, and Dewei Peng who may have more information. Sounds good?")
>>> System.finish()

# Event: Alice Anastasiou says: Yes, that is fine.
>>> Enterprise.resolve_person('Hannah')
{"user_id": "hannah", "email": "hannah@proseware.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, Can you please tell me first name and ages of the students who have a dog but do not have a cat as pet?")
>>> Enterprise.resolve_person('Cassie')
{"user_id": "cassie", "email": "cassie@proseware.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie", "Hi Cassie, Can you please tell me first name and ages of the students who have a dog but do not have a cat as pet?")
>>> Enterprise.resolve_person('Dewei')
{"user_id": "dewei", "email": "dewei@company.com", "full_name": "Dewei Peng"}
>>> Enterprise.send_message("dewei", "Hi Dewei, Can you please tell me first name and ages of the students who have a dog but do not have a cat as pet?")
>>> System.finish()

# Event: Hannah Jarvis says: I know that student with `student id`: 1001 has a pet with `pet id`: 2001 and student with `student id`: 1002 has a pet with `pet id`: 2002
>>> Reflection.thought("Hannah has provided the information about the pets owned by students. I wil wait for Cassie and Dewei to respond.")
>>> System.finish()

# Event: Cassie Hicks says: I do not have the names and ages of students. Would you like to know about the pet ids of various students ?
>>> Reflection.thought("Cassie does not have the requested information. I already contacted Cassie once, so I will not send more messages. I will wait for Dewei to respond.")
>>> System.finish()

# Event: Dewei Peng says: Sorry, I do not have the information you are looking for. But you can try contacting Bhushan Magar who I think might have the information you are looking for.
>>> Reflection.thought("Dewei does not have the requested information on student names and ages. I will reach out to Bhushan Magar for the information. Since I haven't checked with Alice so far about reaching out to Bhushan, I will confirm it now")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't find enough information yet. I will now reach out to Bhushan Magar who may have more information. Sounds good?", title="Re: Clarification needed")
>>> System.finish()

# Event: Alice Anastasiou says: Sounds good, thanks for checking.
>>> Reflection.thought("Alice has confirmed that I can reach out to Bhushan. I will contact Bhushan requesting for the first name and age of the students with a dog pet.")
>>> Enterprise.resolve_person('Bhushan')
{"user_id": "bhushan", "email": "bhushan@proseware.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, Can you please tell me first name and ages of the students who have a dog but do not have a cat as pet?", title="Information needed about student")
>>> System.finish()

# Event: Bhushan Magar says: I do not have the information about pet types.
>>> Reflection.thought("Bhushan does not have provided information. I have received responses from everyone now. I will let Alice know.")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't find enough information to answer your request. Do you need any more help?", title="Re: Clarification needed")

# Event: Alice Anastasiou says: No worries. That would be all.
>>> Enterprise.send_session_completed()
>>> Reflection.thought("Alice has confirmed that no more information is needed. I will close the session")
>>> System.finish()
"""


infoseeking_spider_v2_b_splittables_messageall = """### Example ###

# Event: Alice Anastasiou says: Find the first name of students who have cat or dog pet
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id":"alice@company.com", "email":"alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query='students with cat or dog pets')
("Document:\nTitle: concert\nRecord 1: {'concert id': 1, 'concert name': 'Auditions', 'theme': 'Free choice', 'stadium id': '1', 'year': '2014'}\nRecord 2: {'concert id': 2, 'concert name': 'Super bootcamp', 'theme': 'Free choice 2', 'stadium id': '2', 'year': '2014'}\nRecord 3: {'concert id': 3, 'concert name': 'Home Visits', 'theme': 'Bleeding Love', 'stadium id': '2', 'year': '2015'}\nRecord 4: {'concert id': 4, 'concert name': 'Week 1', 'theme': 'Wide Awake', 'stadium id': '10', 'year': '2014'}\nRecord 5: {'concert id': 5, 'concert name': 'Week 1', 'theme': 'Happy Tonight', 'stadium id': '9', 'year': '2015'}\nRecord 6: {'concert id': 6, 'concert name': 'Week 2', 'theme': 'Party All Night', 'stadium id': '7', 'year': '2015'}\n",)
>>> Reflection.thought("I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query='students with cat or dog pets')
{
  "Bhushan Magar": "The user might have information about pets, including their type, age, weight, and owner info.",
  "Cassie Hicks": "User might have information about students and their pet ids.",
  "Hannah Jarvis": "The user might have information about students, including their personal details, demographics, and academic information.",
  "Dewei Peng": "The user might have information about personal details, demographics, and academic information of students."
}
>>> Reflection.thought("I will reach out to each contact: Bhushan Magar, Cassie Hicks, Hannah Jarvis, and Dewei Peng. I will confirm with Alice if that is okay")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't find enough information in the documents to compile that directly. I can reach out to Bhushan Magar, Cassie Hicks, Hannah Jarvis, and Dewei Peng who may have more information. Sounds good?")
>>> System.finish()

# Event: Alice Anastasiou says: yes
>>> Reflection.thought("Alice has confirmed that I can reach out to Bhushan Magar, Cassey Hicks, Hannah Jarvis, and Dewei Peng. I will resolve their names to get the right contact information as needed. I will each out to each of them asking them about the first name of students who have cat or dog pet")
>>> Enterprise.resolve_person('Hannah')
{"user_id": "hannah", "email": "hannah@proseware.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, Can you please tell me the first name of students who have cat or dog pet?")
>>> Enterprise.resolve_person('Cassie')
{"user_id": "cassie", "email": "cassie@proseware.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie", "Hi Cassie, Can you please tell me the first name of students who have cat or dog pet?")
>>> Enterprise.resolve_person('Dewei')
{"user_id": "dewei", "email": "dewei@company.com", "full_name": "Dewei Peng"}
>>> Enterprise.send_message("dewei", "Hi Dewei, Can you please tell me the first name of students who have cat or dog pet?")
>>> Enterprise.resolve_person('Bhushan')
{"user_id": "bhushan", "email": "bhushan@proseware.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, Can you please tell me the first name of students who have cat or dog pet?")
>>> System.finish()

# Event: Bhushan Magar says: Hi. I do not know student names. I can provide pet IDs of pets that are a cat or dog. Would you like that?
>>> Reflection.thought("Bhushan does not have the requested information but can provide pet ids of pets that are a cat or dog. I have contacted Bhushan once alredy. I will wait for response from Hannah, Cassie, and Dewei.")
>>> System.finish()

# Event: Cassie Hicks says: I do not know student names and pet types.
>>> Reflection.thought("Cassie does not have the requested information. I have contacted Cassie once already. I will wait for response from Hannah, Cassie, and Dewei.")
>>> System.finish()

# Event: Hannah Jarvis says: I know about student names but not their pet types. Would you like to know about the student names ?
>>> Reflection.thought("Cassie does not have the requested information. I have contacted Cassie once already. I will wait for response from Hannah, Cassie, and Dewei.")
>>> System.finish()

# Event: Dewei Peng says: I only know about first names of students. Student id 1001 -> Tracy Kim; Student id 1002 -> Mark Twain; Student id 1003 -> Grace Kim.
>>> Reflection.thought("Dewei shred the first names of the students. I have received responses from everyone I contacted. Alice's documents and information received from various contacts is unsufficient to answer the question. I willl let Alice know.")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't get enough information to find the names of students who have a cat or dog pet. \nDo you need any more information?")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That would be all.
>>> Reflection.thought("Alice has confirmed that no more information is needed. I will close the session")
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


infoseeking_spider_v2_c_unanswerable_messageall = """### Example ###

# Event: Bhushan Magar says: Can you please tell me the names of the students who are below the age of 21?
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query="students below the age of 21")
("Document:\nCollection name: pets 1\nTitle: Student\nRecord 1: {'student id': 1001, 'last name': 'Smith', 'first name': 'Linda', 'age': 18, 'sex': 'F', 'major': 600, 'advisor': 1121, 'city code': 'BAL'}\nRecord 2: {'student id': 1002, 'last name': 'Kim', 'first name': 'Tracy', 'age': 19, 'sex': 'F', 'major': 600, 'advisor': 7712, 'city code': 'HKG'}\nRecord 3: {'student id': 1003, 'last name': 'Jones', 'first name': 'Shiela', 'age': 21, 'sex': 'F', 'major': 600, 'advisor': 7792, 'city code': 'WAS'}\nRecord 4: {'student id': 1004, 'last name': 'Kumar', 'first name': 'Dinesh', 'age': 20, 'sex': 'M', 'major': 600, 'advisor': 8423, 'city code': 'CHI'}\nRecord 5: {'student id': 1005, 'last name': 'Gompers', 'first name': 'Paul', 'age': 26, 'sex': 'M', 'major': 600, 'advisor': 1121, 'city code': 'YYZ'}\nRecord 6: {'student id': 1006, 'last name': 'Schultz', 'first name': 'Andy', 'age': 18, 'sex': 'M', 'major': 600, 'advisor': 1148, 'city code': 'BAL'}\nRecord 7: {'student id': 1007, 'last name': 'Apap', 'first name': 'Lisa', 'age': 18, 'sex': 'F', 'major': 600, 'advisor': 8918, 'city code': 'PIT'}\nRecord 8: {'student id': 1008, 'last name': 'Nelson', 'first name': 'Jandy', 'age': 20, 'sex': 'F', 'major': 600, 'advisor': 9172, 'city code': 'BAL'}\nRecord 9: {'student id': 1009, 'last name': 'Tai', 'first name': 'Eric', 'age': 19, 'sex': 'M', 'major': 600, 'advisor': 2192, 'city code': 'YYZ'}\nRecord 10: {'student id': 1010, 'last name': 'Lee', 'first name': 'Derek', 'age': 17, 'sex': 'M', 'major': 600, 'advisor': 2192, 'city code': 'HOU'}\n",)
>>> Reflection.thought("I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query="students below the age of 21")
{
  "Cassie Hicks": "The user might have information about pets, including their type, age, and weight.",
  "Alice Anastasiou": "The user might have information about names of famous boxers."
}
>>> Reflection.thought("I will reach out to each contact: Alice Anastasiou and Cassie Hicks. I will confirm with Bhushan if that is okay")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, I want to reach out to Alice Anastasiou and Cassie Hicks to get more information. Sounds good?")
>>> System.finish()

# Event: Bhushan Magar says: yes
>>> Reflection.thought("Bhushan has confirmed that I can reach out to Alice Anastasiou and Cassie Hicks. I will resolve their names to get the right contact information as needed. I will each out to each of them asking them about names of the students who are below the age of 21")
>>> Enterprise.resolve_person('Cassie')
{"user_id": "cassie", "email": "cassie@proseware.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie", "Hi Cassie, Can you please tell me the first name of students who have cat or dog pet?")
>>> Enterprise.resolve_person('Alice')
{"user_id":"alice@company.com", "email":"alice@company.com", "full_name": "Alice Anastasiou"}
>>> Enterprise.send_message("alice@company.com", "Hi Alice, Can you please tell me the first name of students who have cat or dog pet?")
>>> System.finish()

# Event: Cassie Hicks says: Hi. I do not know student names and age. I can provide information about pets. Would you like that?
>>> Reflection.thought("Cassie does not have the requested information but can provide information about pets. I have contacted Cassie once already and will not send any more messages. I will wait for response from Alice.")
>>> System.finish()

# Event: Alice Anastasiou says: Hi. I can provide student names and age but I do not have information about pet types.
>>> Reflection.thought("Alice does not have the requested information but can provide information about student name and age. I have received responses from all the people I contacted. While various contacts did not provide any information, I am able to compile a response based on user documents.")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, from the 'Student' document, here are the students who are below 21 years of age: 1) Linda Smith 2) Tracy Kim 3) Dinesh Kumar 4) Andy Schultz 5) Lisa Apap 6) Jandy Nelson 7) Eric Tai 8) Derek Lee. Let me know if you'd like more information on this topic!", title="Students below 21 years of age")
>>> System.finish()

# Event: Bhushan Magar says: Thanks. That would be all.
>>> Reflection.thought("User has confirmed that no more information is needed. I will close the session")
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


infoseeking_spider_v2_d_unanswerable_refineques_messageall = """### Example ###

# Event: Bhushan Magar says: Can you please tell me the names of the students who are below the age of 21 who have a dog as a pet?
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query="students below the age of 21 with dog pets")
[
"Document:\nCollection name: concert singer\nTitle: concert\nRecord 1: {'concert id': 1, 'concert name': 'Auditions', 'theme': 'Free choice', 'stadium id': '1', 'year': '2014'}\nRecord 2: {'concert id': 2, 'concert name': 'Super bootcamp', 'theme': 'Free choice 2', 'stadium id': '2', 'year': '2014'}\nRecord 3: {'concert id': 3, 'concert name': 'Home Visits', 'theme': 'Bleeding Love', 'stadium id': '2', 'year': '2015'}\nRecord 4: {'concert id': 4, 'concert name': 'Week 1', 'theme': 'Wide Awake', 'stadium id': '10', 'year': '2014'}\nRecord 5: {'concert id': 5, 'concert name': 'Week 1', 'theme': 'Happy Tonight', 'stadium id': '9', 'year': '2015'}\nRecord 6: {'concert id': 6, 'concert name': 'Week 2', 'theme': 'Party All Night', 'stadium id': '7', 'year': '2015'}\n"
]
>>> Reflection.thought("I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query="students below the age of 21 with dog pets")
{
  "Cassie Hicks": "The user might have information about pets, including their pet type, age, and weight.",
  "Alice Anastasiou": "The user might have information about students, including their personal details, demographics, and academic information.",
  "Lina Smith": "The user might have information about the list of most cute pet animals."
}
>>> Reflection.thought("I will reach out to each contact -- Cassie, Alice, and Lina -- to ask for names of the students who are below the age of 21 who have a dog as a pet. I will first confirm with Bhushan if I can reach out to Alice, Cassie and Lina.")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, I want to reach out to Cassie Hicks, Alice Anastasiou and Lina Smith to get more information. Sounds good?")
>>> System.finish()

# Event: Bhushan Magar says: Yes, that is fine.
>>> Reflection.thought("I will reach out to Alice to ask for the names of students who are below 21 years of age who have a dog as pet")
>>> Enterprise.resolve_person("Alice")
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Enterprise.send_message("alice", "Hi Alice, Can you please tell me the names of the students who are below 21 years of age who have a dog as a pet?", title="Information needed about students")
>>> Enterprise.resolve_person("Cassie")
{"user_id": "cassie", "email": "cassie@company.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie", "Hi Cassie, Can you please tell me the names of the students who are below 21 years of age who have a dog as a pet?", title="Information needed about students")
>>> Enterprise.resolve_person("Lina")
{"user_id": "lina", "email": "lina@company.com", "full_name": "Lina Smith"}
>>> Enterprise.send_message("lina", "Hi Lina, Can you please tell me the names of the students who are below 21 years of age who have a dog as a pet?", title="Information needed about students")
>>> System.finish()

# Event: Alice Anastasiou says: I do not have information about pet types.
>>> Reflection.thought("Alice does not have information about pet types. I have contacted Alice once already. I will wait for responses from Cassie and Lina.")
>>> System.finish()

# Event: Cassie Hicks says: I know about pets and types but do not know about the student owners of the pets.
>>> Reflection.thought("Cassie knows about pets and types but does not know about the student owners of the pets. I have contacted Cassie once already. I will wait for response from Lina.")
>>> System.finish()

# Event: Lina Smith says: I do not know about the student owners of the pets. I do have a list of cute pet animals. Would you like to know about that?
>>> Reflection.thought("Lina does not have the information about the student owners of the pets. I have received responses from everyone I contacted. Based on the documents and the information received, I do not have enough information to answer the question. I will let Bhushan know.")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, I couldn't find enough information to answer your request. Do you need any more help?", title="Re: Information needed about students")
>>> System.finish()

# Event: Bhushan Magar says: Thanks. That would be all.
>>> Reflection.thought("User has confirmed that no more information is needed. I will close the session")
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


def _strip_reflection_thoughts(example: str) -> str:
    return "\n".join(
        [line for line in example.split("\n") if ">>> Reflection.thought" not in line]
    )


infoseeking_spider_v2_a_redirection_splittables_messageall_nocot = (
    _strip_reflection_thoughts(
        infoseeking_spider_v2_a_redirection_splittables_messageall
    )
)
infoseeking_spider_v2_b_splittables_messageall_nocot = _strip_reflection_thoughts(
    infoseeking_spider_v2_b_splittables_messageall
)
infoseeking_spider_v2_c_unanswerable_messageall_nocot = _strip_reflection_thoughts(
    infoseeking_spider_v2_c_unanswerable_messageall
)
infoseeking_spider_v2_d_unanswerable_refineques_messageall_nocot = (
    _strip_reflection_thoughts(
        infoseeking_spider_v2_d_unanswerable_refineques_messageall
    )
)
