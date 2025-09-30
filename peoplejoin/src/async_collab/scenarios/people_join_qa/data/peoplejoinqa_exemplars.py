infoseeking_spider_v2_a_redirection_splittables = """### Example ###

# Event: Alice Anastasiou says: I need first name and ages of the students who have a dog but do not have a cat as pet
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id":"alice@company.com", "email":"alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query="name and age of students with dogs but no cats as pets")
[
"Document:\nCollection name: pets 1\nTitle: pets\nRecord 1: {'pet id': 2001, 'pet type': 'cat', 'pet age': 3, 'weight': 12.0}\nRecord 2: {'pet id': 2002, 'pet type': 'dog', 'pet age': 2, 'weight': 13.4}\nRecord 3: {'pet id': 2003, 'pet type': 'dog', 'pet age': 1, 'weight': 9.3}\n"
]
>>> Reflection.thought("User documents have relevant information about pet ids and pet types. But there is not enough information to answer the query directly. I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query="name and age of students with dogs but no cats as pets")
{
  "Cassie Hicks": "User might have information about students and their pet IDs.",
  "Dewei Peng": "The user might have information about students, including their personal details, demographics, and academic information.",
  "Hannah Jarvis": "User might have information about students and their pet IDs."
}
>>> Reflection.thought("I can reach out to Hannah and Cassie who seem to have more information about pets owned by students. I might have to reach out to Dewei for the names and ages of students since Dewei seems to have personal details about students")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't find enough information in your documents to compile that directly. I can reach out to Hannah Jarvis, Cassie Hicks, and Dewei Peng who may have more information. Sounds good?")
>>> System.finish()

# Event: Alice Anastasiou says: Yes, that is fine.
>>> Reflection.thought("Both Hannah and Cassie have information about pets owned by students. Since I would need the list of all students who have a dog but not a cat as pet, I will contact both Hannah and Cassie. I will first resolve Hannah's contact information and then reach out to Hannah")
>>> Enterprise.resolve_person("Hannah")
{"user_id": "hannah", "email": "hannah@proseware.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, Can you please tell me the list of students and their pets?", title="Information needed about students")
>>> System.finish()

# Event: Hannah Jarvis says: Sure. I know that student with `student id`: 1001 has a pet with `pet id`: 2001 and student with `student id`: 1002 has a pet with `pet id`: 2002
>>> Reflection.thought("Hannah has provided the information about the pets owned by students. Since I need the list of all students who have a dog but not a cat as pet, I will reach out to Cassie in case Cassie knows more about the pets owned by students")
>>> Enterprise.resolve_person("Cassie")
{"user_id": "cassie", "email": "cassie@proseware.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie", "Hi Cassie, Can you please tell me the list of students and their pets?", title="Information needed about students")
>>> System.finish()

# Event: Cassie Hicks says: Sure. 1) `student id`: 1002 has a pet with `pet id`: 2002; 2) `student id`: 1002 has a pet with `pet id`: 2003
>>> Reflection.thought("From information provided by Hannah and Cassie, and information in user documents, I can compile the list of students and their pet_ids as follows: `student id`: 1001 -> `pet id`: 2001, `student id`: 1002 -> `pet id`: 2002, `student id`: 1002 -> `pet id`: 2003. From Alice`s documents, I can get the type of animal for the pet ids: `pet id`: 2001 -> cat, `pet id`: 2002 -> dog, `pet id`: 2003 -> dog. I can conclude that the student with `student id`: 1002 has a dog but not a cat as a pet. I will reach out to Dewei for the first name and age of the student")
>>> Enterprise.resolve_person("Dewei")
{"user_id": "dewei", "email": "dewei@company.com", "full_name": "Dewei Peng"}
>>> Enterprise.send_message("dewei", "Hi Dewei, Can you please tell the name and age of student with `student id`: 1002?", title="Information needed about students")
>>> System.finish()

# Event: Dewei Peng says: Sorry, I do not have the information you are looking for. But you can try contacting Bhushan Magar who I think might have the information you are looking for.
>>> Reflection.thought("Dewei does not have the requested information on student names and ages. I will reach out to Bhushan Magar for the information. Since I haven't checked with Alice so far about reaching out to Bhushan, I will confirm it now")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't find enough information after reaching out to Dewei. I will now reach out to Bhushan Magar who may have more information. Sounds good?", title="Re: Clarification needed")
>>> System.finish()

# Event: Alice Anastasiou says: Sounds good, thanks for checking.
>>> Reflection.thought("Alice has confirmed that I can reach out to Bhushan. I will contact Bhushan requesting for the first name and age of the specified students")
>>> Enterprise.resolve_person("Bhushan")
{"user_id": "bhushan", "email": "bhushan@proseware.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, Can you please tell the first name and age of `student id`: 1002?", title="Information needed about student")
>>> System.finish()

# Event: Bhushan Magar says: {"student id": 1002, "first name": "Tracy"}
>>> Reflection.thought("Bhushan has provided the first name of the student but not the age. I will reach out to Bhushan again for the age information. Since I already confirmed with Alice about reaching out to Bhushan, I will not confirm again")
>>> Enterprise.send_message("bhushan", "Thanks Bhushan! Can I also get the age information of `student id`: 1002?", title="Re: Information needed about student")
>>> System.finish()

# Event: Bhushan Magar says: Oh right, forgot about that. Age is 19.
>>> Reflection.thought("Bhushan has provided the name and age of the student. I now have all the information to answer Alice's query")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, the first name and age of the student who has a dog but not a cat as a pet is:\n- First name: Tracy\n- Age: 19\nDo you need any more information?", title="Re: Information needed about student")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That would be all.
>>> Enterprise.send_session_completed()
>>> Reflection.thought("Alice has confirmed that no more information is needed. I will close the session")
>>> System.finish()
"""


infoseeking_spider_v2_b_splittables = """### Example ###

# Event: Alice Anastasiou says: Find the first name of students who have cat or dog pet
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id":"alice@company.com", "email":"alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query="students with cat or dog pets")
("Document:\nTitle: concert\nRecord 1: {'concert id': 1, 'concert name': 'Auditions', 'theme': 'Free choice', 'stadium id': '1', 'year': '2014'}\nRecord 2: {'concert id': 2, 'concert name': 'Super bootcamp', 'theme': 'Free choice 2', 'stadium id': '2', 'year': '2014'}\nRecord 3: {'concert id': 3, 'concert name': 'Home Visits', 'theme': 'Bleeding Love', 'stadium id': '2', 'year': '2015'}\nRecord 4: {'concert id': 4, 'concert name': 'Week 1', 'theme': 'Wide Awake', 'stadium id': '10', 'year': '2014'}\nRecord 5: {'concert id': 5, 'concert name': 'Week 1', 'theme': 'Happy Tonight', 'stadium id': '9', 'year': '2015'}\nRecord 6: {'concert id': 6, 'concert name': 'Week 2', 'theme': 'Party All Night', 'stadium id': '7', 'year': '2015'}\n",)
>>> Reflection.thought("User documents does not have enough information to answer the query directly. I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query="students with cat or dog pets")
{
  "Bhushan Magar": "The user might have information about pets, including their type, age, weight, and owner info.",
  "Cassie Hicks": "User might have information about students and their pet ids.",
  "Hannah Jarvis": "The user might have information about students, including their personal details, demographics, and academic information.",
  "Dewei Peng": "The user might have information about personal details, demographics, and academic information of students."
}
>>> Reflection.thought("I can reach out to Bhushan Magar, Cassie Hicks, Hannah Jarvis, and Dewei Peng since they seem to have more information about the query. I will confirm with Alice if that is okay")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, I couldn't find enough information in the documents to compile that directly. I can reach out to Bhushan Magar, Cassie Hicks, Hannah Jarvis, and Dewei Peng who may have more information. Sounds good?")
>>> System.finish()

# Event: Alice Anastasiou says: yes
>>> Reflection.thought("Alice has confirmed that I can reach out to Bhushan Magar, Cassey Hicks, Hannah Jarvis, and Dewei Peng. I will resolve their names to get the right contact information as needed. I will first reach out to Bhushan for the first name of the student who has cat or dog pet as Bhushan seems to have information about pet type and owner info")
>>> Enterprise.resolve_person("Bhushan")
{"user_id": "bhushan", "email": "bhushan@proseware.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, Can you please tell the first name of students who have a cat or dog as a pet?", title="Information needed about student")
>>> System.finish()

# Event: Bhushan Magar says: Hi. I do not know student names. I can provide pet IDs of pets that are a cat or dog. Would you like that?
>>> Reflection.thought("Bhushan does not have the requested information but can provide pet ids of pets that are a cat or dog. I will ask Bhushan for the pet ids. Once Bhushan provides the information, I can reach out to Cassie for the student ids")
>>> Enterprise.send_message("bhushan", "Yes, please provide the pet IDs of pets that are a cat or dog", title="Re: Information needed about student")
>>> System.finish()

# Event: Bhushan Magar says: Sure, here are the pet IDs of pets that are a cat or dog: 2001, 2002, 2003
>>> Reflection.thought("Bhushan has provided the pet ids of pets that are a cat or dog. I will reach out to Cassie for the student ids for the specified pet ids")
>>> Enterprise.resolve_person("Cassie")
{"user_id": "cassie", "email": "cassie@proseware.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie@proseware.com", "Hi Cassie, can you please let me know the student IDs for the following pet IDs?\n- `pet id`: 2001\n- `pet id`: 2002\n- `pet id`: 2003", title="Information needed about students")
>>> System.finish()

# Event: Cassie Hicks says: `pet id`: 2001 -> student id: 1001\n`pet id`: 2002 -> student id: 1002\n`pet id`: 2003 -> student id: 1002\n`pet id`: 2004 -> student id: 1015
>>> Reflection.thought("Cassie has provided the student ids for the pet ids. I might have to each out to both Hannah and Dewei to get student names. I will start by reaching out to Hannah for the student names.")
>>> Enterprise.resolve_person("Hannah")
{"user_id": "hannah", "email": "hannah@proseware.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah@proseware.com", "Hi Hannah, Can you please tell the first name of students with the following student IDs:\n- `student id`: 1001\n- `student id`: 1002", title="Information needed about students")
>>> System.finish()

# Event: Hannah Jarvis says: Sure. I only have information for student 1001, whose name is Linda Smith.
>>> Reflection.thought("Hannah has provided the student name for student 1001, but not for student 1002. I will reach out to Dewei since Dewei also seems to have information about personal details of students")
>>> Enterprise.resolve_person("Dewei")
{"user_id": "dewei", "email": "dewei@company.com", "full_name": "Dewei Peng"}
>>> Enterprise.send_message("dewei@company.com", "Hi Dewei, Can you please tell the first name of the student with student ID: 1002", title="Information needed about students")
>>> System.finish()

# Event: Dewei Peng says: Sure. The first name of student 1002 is Tracy.
>>> Reflection.thought("Dewei has provided the student name for student 1002. I now have all the information needed to answer the query. I will send the information to Alice")
>>> Enterprise.send_message("alice@company.com", "Hi Alice, the first names of students who have a cat or dog pet are:\n- Linda\n- Tracy\nDo you need any more information?")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That would be all.
>>> Reflection.thought("Alice has confirmed that no more information is needed. I will close the session")
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


infoseeking_spider_v2_c_unanswerable = """### Example ###

# Event: Bhushan Magar says: Can you please tell me the names of the students who are below the age of 21?
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query="students below the age of 21")
("Document:\nCollection name: pets 1\nTitle: Student\nRecord 1: {'student id': 1001, 'last name': 'Smith', 'first name': 'Linda', 'age': 18, 'sex': 'F', 'major': 600, 'advisor': 1121, 'city code': 'BAL'}\nRecord 2: {'student id': 1002, 'last name': 'Kim', 'first name': 'Tracy', 'age': 19, 'sex': 'F', 'major': 600, 'advisor': 7712, 'city code': 'HKG'}\nRecord 3: {'student id': 1003, 'last name': 'Jones', 'first name': 'Shiela', 'age': 21, 'sex': 'F', 'major': 600, 'advisor': 7792, 'city code': 'WAS'}\nRecord 4: {'student id': 1004, 'last name': 'Kumar', 'first name': 'Dinesh', 'age': 20, 'sex': 'M', 'major': 600, 'advisor': 8423, 'city code': 'CHI'}\nRecord 5: {'student id': 1005, 'last name': 'Gompers', 'first name': 'Paul', 'age': 26, 'sex': 'M', 'major': 600, 'advisor': 1121, 'city code': 'YYZ'}\nRecord 6: {'student id': 1006, 'last name': 'Schultz', 'first name': 'Andy', 'age': 18, 'sex': 'M', 'major': 600, 'advisor': 1148, 'city code': 'BAL'}\nRecord 7: {'student id': 1007, 'last name': 'Apap', 'first name': 'Lisa', 'age': 18, 'sex': 'F', 'major': 600, 'advisor': 8918, 'city code': 'PIT'}\nRecord 8: {'student id': 1008, 'last name': 'Nelson', 'first name': 'Jandy', 'age': 20, 'sex': 'F', 'major': 600, 'advisor': 9172, 'city code': 'BAL'}\nRecord 9: {'student id': 1009, 'last name': 'Tai', 'first name': 'Eric', 'age': 19, 'sex': 'M', 'major': 600, 'advisor': 2192, 'city code': 'YYZ'}\nRecord 10: {'student id': 1010, 'last name': 'Lee', 'first name': 'Derek', 'age': 17, 'sex': 'M', 'major': 600, 'advisor': 2192, 'city code': 'HOU'}\n",)
>>> Reflection.thought("User documents have information about student ids and ages. There is enough information to answer the query directly")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, from the 'Student' document, here are the students who are below 21 years of age: 1) Linda Smith 2) Tracy Kim 3) Dinesh Kumar 4) Andy Schultz 5) Lisa Apap 6) Jandy Nelson 7) Eric Tai 8) Derek Lee. Let me know if you'd like more information on this topic!", title="Students below 21 years of age")
>>> System.finish()

# Event: Bhushan Magar says: Thanks! Of these, how many students have dual citizenship?
>>> Reflection.thought("User has asked a follow-up question about the students who have dual citizenship. The information is not available in the documents searched earlier. I will search for more documents")
>>> EnterpriseSearch.search_documents(query="students with dual citizenship")
[
"Document:\nCollection name: pets 1\nTitle: Student\nRecord 1: {'student id': 1001, 'last name': 'Smith', 'first name': 'Linda', 'age': 18, 'sex': 'F', 'major': 600, 'advisor': 1121, 'city code': 'BAL'}\nRecord 2: {'student id': 1002, 'last name': 'Kim', 'first name': 'Tracy', 'age': 19, 'sex': 'F', 'major': 600, 'advisor': 7712, 'city code': 'HKG'}\nRecord 3: {'student id': 1003, 'last name': 'Jones', 'first name': 'Shiela', 'age': 21, 'sex': 'F', 'major': 600, 'advisor': 7792, 'city code': 'WAS'}\nRecord 4: {'student id': 1004, 'last name': 'Kumar', 'first name': 'Dinesh', 'age': 20, 'sex': 'M', 'major': 600, 'advisor': 8423, 'city code': 'CHI'}\nRecord 5: {'student id': 1005, 'last name': 'Gompers', 'first name': 'Paul', 'age': 26, 'sex': 'M', 'major': 600, 'advisor': 1121, 'city code': 'YYZ'}\nRecord 6: {'student id': 1006, 'last name': 'Schultz', 'first name': 'Andy', 'age': 18, 'sex': 'M', 'major': 600, 'advisor': 1148, 'city code': 'BAL'}\nRecord 7: {'student id': 1007, 'last name': 'Apap', 'first name': 'Lisa', 'age': 18, 'sex': 'F', 'major': 600, 'advisor': 8918, 'city code': 'PIT'}\nRecord 8: {'student id': 1008, 'last name': 'Nelson', 'first name': 'Jandy', 'age': 20, 'sex': 'F', 'major': 600, 'advisor': 9172, 'city code': 'BAL'}\nRecord 9: {'student id': 1009, 'last name': 'Tai', 'first name': 'Eric', 'age': 19, 'sex': 'M', 'major': 600, 'advisor': 2192, 'city code': 'YYZ'}\nRecord 10: {'student id': 1010, 'last name': 'Lee', 'first name': 'Derek', 'age': 17, 'sex': 'M', 'major': 600, 'advisor': 2192, 'city code': 'HOU'}\n"
]
>>> Reflection.thought("User documents do not have information about students with dual citizenship. I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query="students with dual citizenship")
{
  "Cassie Hicks": "The user might have information about pets, including their type, age, and weight.",
  "Alice Anastasiou": "The user might have information about students, including their personal details, demographics, and academic information."
}
>>> Reflection.thought("No relevant people found who can answer the query. I will inform the user that the information is not available")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, I searched thoroughly, but unfortunately, I couldn`t find any detailed information about students with dual citizenship in your documents. No relevant people were identified who could provide this specific information either. Let me know if there`s anything else I can assist you with!", title="Information about students with dual citizenship")
>>> System.finish()

# Event: Bhushan Magar says: Thanks for checking. That would be all.
>>> Reflection.thought("User has confirmed that no more information is needed. I will close the session")
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


infoseeking_spider_v2_d_unanswerable_refineques = """### Example ###

# Event: Bhushan Magar says: Can you please tell me the names of the students who are below the age of 21 who have a dog as a pet?
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Reflection.thought("I will search user documents to see if the user request can be answered directly")
>>> EnterpriseSearch.search_documents(query="students below the age of 21 with dog pets")
[
"Document:\nCollection name: concert singer\nTitle: concert\nRecord 1: {'concert id': 1, 'concert name': 'Auditions', 'theme': 'Free choice', 'stadium id': '1', 'year': '2014'}\nRecord 2: {'concert id': 2, 'concert name': 'Super bootcamp', 'theme': 'Free choice 2', 'stadium id': '2', 'year': '2014'}\nRecord 3: {'concert id': 3, 'concert name': 'Home Visits', 'theme': 'Bleeding Love', 'stadium id': '2', 'year': '2015'}\nRecord 4: {'concert id': 4, 'concert name': 'Week 1', 'theme': 'Wide Awake', 'stadium id': '10', 'year': '2014'}\nRecord 5: {'concert id': 5, 'concert name': 'Week 1', 'theme': 'Happy Tonight', 'stadium id': '9', 'year': '2015'}\nRecord 6: {'concert id': 6, 'concert name': 'Week 2', 'theme': 'Party All Night', 'stadium id': '7', 'year': '2015'}\n"
]
>>> Reflection.thought("User documents do not have information about students with dog pets. I will search for relevant people who may have more information")
>>> EnterpriseSearch.search_relevant_people(query="students below the age of 21 with dog pets")
{
  "Cassie Hicks": "The user might have information about pets, including their pet type, age, and weight.",
  "Alice Anastasiou": "The user might have information about students, including their personal details, demographics, and academic information.",
  "Lina Smith": "The user might have information about the list of most cute pet animals."
}
>>> Reflection.thought("I can reach out to Alice and Cassie who seem to have more information about pets owned by students. I can first reach out to Alice to ask for the names of students who are below 21 years of age. I will first confirm with Bhushan if I can reach out to Alice and Cassie.")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, I couldn't find enough information in your documents to compile that information directly. I can reach out to Alice and Cassie who may have more information. Sounds good?")
>>> System.finish()

# Event: Bhushan Magar says: Yes, that is fine.
>>> Reflection.thought("I will reach out to Alice to ask for the names of students who are below 21 years of age who have a dog as pet")
>>> Enterprise.resolve_person("Alice")
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Enterprise.send_message("alice", "Hi Alice, Can you please tell me the names of the students who are below 21 years of age who have a dog as a pet?", title="Information needed about students")
>>> System.finish()

# Event: Alice Anastasiou says: I do not have information about pet types.
>>> Reflection.thought("Alice does not have information about pet types. Alice did not deny having information about students below 21 years of age. I will ask Alice for the names of students who are below 21 years of age, and then ask Cassie for the pet types of students.")
>>> Enterprise.send_message("alice", "Hi Alice, Can you please tell me the names of the students who are below 21 years of age?", title="Information needed about students")
>>> System.finish()

# Event: Alice Anastasiou says: Sure, here is the info: 1) "student id": 1001, "last name": "Smith", "first name": "Linda", "age": 18; 2) "student id": 1002, "last name": "Kim", "first name": "Tracy", "age": 19.
>>> Reflection.thought("Alice provided the names of students who are below 21 years of age. I will now ask Cassie for the pet types of students.")
>>> Enterprise.resolve_person("Cassie")
{"user_id": "cassie", "email": "cassie@proseware.com", "full_name": "Cassie Hicks"}
>>> Enterprise.send_message("cassie", "Hi Cassie, Can you please tell me the pet types of the following students: 1) Linda Smith, student id 1001 2) Tracy Kim, student id 1002", title="Information needed about students")
>>> System.finish()

# Event: Cassie Hicks says: I know about pets and types but do not know about the student owners of the pets.
>>> Reflection.thought("Cassie knows about pets and types but does not know about the student owners of the pets. I do nont think there are other people whoc can provide the remaining information. I will inform Bhushan that I could get information about students who are below 21 years of age, but could not get information about students who have a dog as pet.")
>>> Enterprise.send_message("bhushan", "Hi Bhushan, I contacted Alice and Cassie. I could get information about students who are below 21 years of age: 1) Linda Smith, student id 1001 2) Tracy Kim, student id 1002. However, I could not get information about students who have a dog as a pet.", title="Information about students")
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


def _strip_reflection_thoughts(example: str) -> str:
    return "\n".join(
        [line for line in example.split("\n") if ">>> Reflection.thought" not in line]
    )


infoseeking_spider_v2_a_redirection_splittables_nocot = _strip_reflection_thoughts(
    infoseeking_spider_v2_a_redirection_splittables
)
infoseeking_spider_v2_b_splittables_nocot = _strip_reflection_thoughts(
    infoseeking_spider_v2_b_splittables
)
infoseeking_spider_v2_c_unanswerable_nocot = _strip_reflection_thoughts(
    infoseeking_spider_v2_c_unanswerable
)
infoseeking_spider_v2_d_unanswerable_refineques_nocot = _strip_reflection_thoughts(
    infoseeking_spider_v2_d_unanswerable_refineques
)
