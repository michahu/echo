from dataclasses import dataclass

import jsons

from src.async_collab.core.calendar import Calendar
from src.async_collab.core.document import Document, DocumentCollection
from src.async_collab.core.person import Person


@dataclass
class Tenant:
    tenant_id: str
    people: tuple[Person, ...]
    person_id_to_document_collection: dict[str, DocumentCollection]
    person_id_to_calendar: dict[str, Calendar]
    person_id_to_description: dict[str, str]

    def get_all_documents_of_person(self, person_id: str) -> list[Document]:
        assert (
            person_id in self.person_id_to_document_collection
        ), f"Person {person_id} not found"
        return self.person_id_to_document_collection[person_id].documents

    def search_documents_of_person(
        self, person_id: str, query: str, top_n: int = 5
    ) -> list[Document]:
        assert (
            person_id in self.person_id_to_document_collection
        ), f"Person {person_id} not found"
        return self.person_id_to_document_collection[person_id].search_documents(
            query, top_n
        )

    def get_calendar_of_person(self, person_id: str) -> Calendar:
        assert person_id in self.person_id_to_calendar, f"Person {person_id} not found"
        return self.person_id_to_calendar[person_id]

    def get_person_by_id(self, person_id: str) -> Person | None:
        for person in self.people:
            if person.person_id == person_id:
                return person
        return None

    def get_person_by_id_relaxed(self, person_id: str) -> Person | None:
        # ignore any case and any part after @
        person_id = person_id.split("@")[0].lower()
        for person in self.people:
            if person.person_id.lower() == person_id:
                return person
        return None

    @property
    def person_id_to_person_dict(self) -> dict[str, Person]:
        return {person.person_id: person for person in self.people}

    @staticmethod
    def from_dict(data: dict) -> "Tenant":
        return jsons.load(data, cls=Tenant)
