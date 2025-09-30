from dataclasses import dataclass


@dataclass
class Event:
    start_time: str
    end_time: str
    title: str


@dataclass
class Calendar:
    events: list[Event]
