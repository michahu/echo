import builtins
import dataclasses
import inspect
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from types import GenericAlias, UnionType
from typing import (
    NewType,
    _GenericAlias,  # pyright: ignore[reportAttributeAccessIssue]
    _UnionGenericAlias,  # pyright: ignore[reportAttributeAccessIssue]
    get_args,
    get_origin,
)

from async_collab.agent.agent_config import AgentConfig
from async_collab.core.message import Message
from async_collab.tenant.tenant import Tenant


@dataclass(frozen=True)
class Argument:
    name: str
    description: str
    type: str
    required: bool = True
    validation_args: list[str] | None = None


@dataclass(frozen=True)
class FunctionSignature:
    name: str
    description: str
    arguments: tuple[Argument, ...]
    return_type: str
    return_description: str | None = None
    id: str | None = None
    alternative_names: list[str] | None = None


def get_type_str(type_) -> str:
    # type(None) == builtins.NoneType per help(type(None)), but it is not accessible
    # Both None and NoneType appear as types
    if type_ == type(None):
        return "None"

    match type_:
        case builtins.float:
            return "float"
        case builtins.int:
            return "int"
        case builtins.str:
            return "str"
        case builtins.list:
            return "list"
        case builtins.tuple:
            return "tuple"
        case builtins.Ellipsis:
            return "..."
        case None:
            return "None"
        case NewType():
            # NewType has no public accessors
            return type_.__name__  # pyright: ignore[reportAttributeAccessIssue]
        # e.g. int | float
        case UnionType():
            inner_types = get_args(type_)
            return " | ".join([get_type_str(x) for x in inner_types])
        # e.g. Optional[x], int | Random (yes, this is not handled the same as int | float)
        case _UnionGenericAlias():
            inner_types = get_args(type_)
            return " | ".join([get_type_str(x) for x in inner_types])
        # _GenericAlias has to be below _UnionGenericAlias due to subtyping
        # e.g. list[int]
        case GenericAlias() | _GenericAlias():
            container_type = get_origin(type_)
            inner_types = get_args(type_)
            return f"{get_type_str(container_type)}[{', '.join([get_type_str(x) for x in inner_types])}]"
        case inspect.Parameter.empty:
            raise ValueError("Parameter type must be ascribed")
        # e.g. dataclasses, NewType
        case _:
            return type_.__name__


@dataclasses.dataclass(frozen=True)
class Type:
    """Total type replacement. Use when the type is wrapped in a container type (e.g., list[Event] -> list[Calendar.Event])."""

    # Name of type
    type_: str
    description: str


@dataclasses.dataclass(frozen=True)
class ReturnType:
    """Same as `Type`, but no description as it's a return type."""

    # Name of type
    type_: str


@dataclass
class Plugin(ABC):
    plugin_id: str  # id of the plugin e.g. 'simple_reasoning' or 'teams_react' or 'teams_sps' or ...
    plugin_name: str  # plugin name as used in annotations. e.g. Reflection.thought() or Teams.send_message() or ...
    tenant: Tenant  # plugins operate on Tenant data. Lets tie it to a tenant. TenantView is Tenant along with the specification of the primary user
    primary_user_id: str  # primary user of the tenant
    plugin_impls: dict[str, Callable]  # plugin implementations
    dependencies: tuple[str, ...] = ()  # dependencies of the plugin

    def __init__(
        self, tenant: Tenant, primary_user_id: str, plugin_impls: dict[str, Callable]
    ):
        self.tenant = tenant
        self.primary_user_id = primary_user_id
        self.plugin_impls = plugin_impls

    @staticmethod
    def get_plugin(
        agent_conf: AgentConfig,
        tenant: Tenant,
        message_send_queue: list[Message] | None,
    ) -> "Plugin":
        """
        # todo
        """
        raise NotImplementedError
