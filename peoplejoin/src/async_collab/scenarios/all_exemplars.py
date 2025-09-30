from datetime import datetime

from async_collab.core.bot import Bot
from async_collab.core.message import Message
from async_collab.core.person import Person
from async_collab.orchestrator.datum import (
    ActionObservation,
    AsyncCollabDatum,
    AsyncCollabDatumMetadata,
    EventTriggeredActions,
)
from async_collab.scenarios.people_join_qa.data.peoplejoinqa_exemplars import (
    infoseeking_spider_v2_a_redirection_splittables,
    infoseeking_spider_v2_a_redirection_splittables_nocot,
    infoseeking_spider_v2_b_splittables,
    infoseeking_spider_v2_b_splittables_nocot,
    infoseeking_spider_v2_c_unanswerable,
    infoseeking_spider_v2_c_unanswerable_nocot,
    infoseeking_spider_v2_d_unanswerable_refineques,
    infoseeking_spider_v2_d_unanswerable_refineques_nocot,
)
from async_collab.scenarios.people_join_qa.data.peoplejoinqa_exemplars_messageall import (
    infoseeking_spider_v2_a_redirection_splittables_messageall,
    infoseeking_spider_v2_b_splittables_messageall,
    infoseeking_spider_v2_c_unanswerable_messageall,
    infoseeking_spider_v2_d_unanswerable_refineques_messageall,
)
from async_collab.scenarios.people_join_qa.data.peoplejoinqa_exemplars_messagenone import (
    infoseeking_spider_v2_a_redirection_splittables_messagenone,
    infoseeking_spider_v2_b_splittables_messagenone,
    infoseeking_spider_v2_c_unanswerable_messagenone,
    infoseeking_spider_v2_d_unanswerable_refineques_messagenone,
)
from async_collab.scenarios.peoplejoin_doc_creation.data.peoplejoindoccreation_exemplars import (
    multinews_all_dialogue1,
    multinews_all_dialogue2,
    multinews_dialogue1,
    multinews_dialogue2,
    multinews_dialogue3,
    multinews_nocot_dialogue1,
    multinews_nocot_dialogue2,
    multinews_nocot_dialogue3,
    multinews_none_dialogue1,
    multinews_none_dialogue2,
)

bot = Bot(owner=Person("1", "Alice", "alice@company.com"))
example1 = AsyncCollabDatum(
    datum_id="sample1",
    tenant_id="peoplejoinqa/battle_death",
    primary=Person("1", "Alice", "alice@company.com"),
    bot=bot,
    initial_message=Message(
        sender=bot,
        content="Hello, how can I help you?",
        message_type="chat",
        created_on=datetime(2021, 1, 1, 0, 0, 1),
        recipient=Person("1", "Alice", "alice@company.com"),
    ),
    flow=[
        EventTriggeredActions(
            trigger_event=Message(
                sender=Person("alice", "Alice", "alice@company.com"),
                recipient=bot,
                content="Hello",
                message_type="chat",
                created_on=datetime(2021, 1, 1, 0, 0, 0),
            ),
            bot_actions=[
                ActionObservation(
                    statement="Hello, how can I help you?",
                    messages=(
                        Message(
                            sender=bot,
                            content="Hello, how can I help you?",
                            message_type="chat",
                            created_on=datetime(2021, 1, 1, 0, 0, 1),
                            recipient=Person("1", "Alice", "alice@company.com"),
                        ),
                    ),
                    required_plugins=("System",),
                )
            ],
        )
    ],
    metadata=AsyncCollabDatumMetadata(
        tenant_id="peoplejoinqa/battle_death",
        description="A simple greeting conversation",
        description_assertions=["The bot should respond to the user's greeting"],
        description_reference_response="Hello, how can I help you?",
        description_reference_people=["1"],
    ),
)


exemplar_by_id: dict[str, AsyncCollabDatum | str] = {
    "sample1": example1,
    "peoplejoinqa_1": infoseeking_spider_v2_a_redirection_splittables,
    "peoplejoinqa_2": infoseeking_spider_v2_b_splittables,
    "peoplejoinqa_3": infoseeking_spider_v2_c_unanswerable,
    "peoplejoinqa_4": infoseeking_spider_v2_d_unanswerable_refineques,
    "peoplejoinqa_1_nocot": infoseeking_spider_v2_a_redirection_splittables_nocot,
    "peoplejoinqa_2_nocot": infoseeking_spider_v2_b_splittables_nocot,
    "peoplejoinqa_3_nocot": infoseeking_spider_v2_c_unanswerable_nocot,
    "peoplejoinqa_4_nocot": infoseeking_spider_v2_d_unanswerable_refineques_nocot,
    "peoplejoinqa_messageall_1": infoseeking_spider_v2_a_redirection_splittables_messageall,
    "peoplejoinqa_messageall_2": infoseeking_spider_v2_b_splittables_messageall,
    "peoplejoinqa_messageall_3": infoseeking_spider_v2_c_unanswerable_messageall,
    "peoplejoinqa_messageall_4": infoseeking_spider_v2_d_unanswerable_refineques_messageall,
    "peoplejoinqa_messagenone_1": infoseeking_spider_v2_a_redirection_splittables_messagenone,
    "peoplejoinqa_messagenone_2": infoseeking_spider_v2_b_splittables_messagenone,
    "peoplejoinqa_messagenone_3": infoseeking_spider_v2_c_unanswerable_messagenone,
    "peoplejoinqa_messagenone_4": infoseeking_spider_v2_d_unanswerable_refineques_messagenone,
    "peoplejoindoccreation_1": multinews_dialogue1,
    "peoplejoindoccreation_2": multinews_dialogue2,
    "peoplejoindoccreation_3": multinews_dialogue3,
    "peoplejoindoccreation_1_nocot": multinews_nocot_dialogue1,
    "peoplejoindoccreation_2_nocot": multinews_nocot_dialogue2,
    "peoplejoindoccreation_3_nocot": multinews_nocot_dialogue3,
    "peoplejoindoccreation_messageall_1": multinews_all_dialogue1,
    "peoplejoindoccreation_messageall_2": multinews_all_dialogue2,
    "peoplejoindoccreation_messagenone_1": multinews_none_dialogue1,
    "peoplejoindoccreation_messagenone_2": multinews_none_dialogue2,
}
