import json
from typing import Literal

from rustic_ai.core import Message
from rustic_ai.core.guild.agent import Agent, ProcessContext
from rustic_ai.core.guild.agent_ext.depends.llm.llm import LLM
from rustic_ai.core.guild.agent_ext.depends.llm.models import (
    ArrayOfContentParts,
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    LLMMessage,
    TextContentPart,
    UserMessage,
)
from rustic_ai.core.utils.basic_class_utils import get_qualified_class_name
from rustic_ai.llm_agent.memories.memories_store import MemoriesStore


class HistoryBasedMemoriesStore(MemoriesStore):
    memory_type: Literal["history_based"] = "history_based"
    text_only: bool = True

    def remember(self, agent: Agent, ctx: ProcessContext, message: LLMMessage) -> None:
        # No-op: History-based memory does not store messages explicitly.
        pass

    @staticmethod
    def _sender_identity(message: Message) -> str | None:
        sender = getattr(message, "sender", None)
        if sender is None:
            return None
        return sender.name or sender.id

    @staticmethod
    def _name_user_messages(messages: list[LLMMessage], sender_name: str | None) -> list[LLMMessage]:
        if not sender_name:
            return messages
        return [
            (
                message.model_copy(update={"name": sender_name})
                if isinstance(message, UserMessage) and not message.name
                else message
            )
            for message in messages
        ]

    @staticmethod
    def _name_assistant_message(message: LLMMessage, sender_name: str | None) -> LLMMessage:
        if sender_name and isinstance(message, AssistantMessage) and not message.name:
            return message.model_copy(update={"name": sender_name})
        return message

    def _filter_text_messages(self, messages):
        text_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                text_messages.append(msg)
            elif isinstance(msg.content, ArrayOfContentParts):
                is_text = False
                for part in msg.content.root:
                    if isinstance(part, TextContentPart):
                        is_text = True
                    else:
                        msg.content.root.remove(part)

                    if is_text:
                        text_messages.append(msg)
        messages = text_messages
        return messages

    def _deduplicate_messages(self, messages):
        seen_content = set()
        unique_messages = []

        for msg in messages:
            # Create a stable key for possibly unhashable content (e.g., dict, complex types)
            try:
                hash(msg.content)
                key = msg.content
            except TypeError:
                try:
                    key = json.dumps(
                        msg.content,
                        default=lambda o: getattr(o, "model_dump", lambda: repr(o))(),
                        sort_keys=True,
                    )
                except TypeError:
                    key = repr(msg.content)

            identity_key = (getattr(msg, "role", None), getattr(msg, "name", None), key)
            if identity_key not in seen_content:
                seen_content.add(identity_key)
                unique_messages.append(msg)
        return unique_messages

    def _extract_messages_from_history(self, enriched_history):
        messages: list[LLMMessage] = []

        for entry in enriched_history:
            message = Message.from_json(entry)
            sender_name = self._sender_identity(message)
            if message.format == get_qualified_class_name(ChatCompletionRequest):
                payload = ChatCompletionRequest.model_validate(message.payload)
                messages.extend(self._name_user_messages(payload.messages, sender_name))
            elif message.format == get_qualified_class_name(ChatCompletionResponse):
                payload = ChatCompletionResponse.model_validate(message.payload)
                messages.append(self._name_assistant_message(payload.choices[0].message, sender_name))
        return messages

    def preprocess(
        self,
        agent: Agent,
        ctx: ProcessContext,
        request: ChatCompletionRequest,
        llm: LLM,
    ) -> ChatCompletionRequest:
        sender_name = self._sender_identity(ctx.message)
        named_messages = self._name_user_messages(request.messages, sender_name)
        named_request = request.model_copy(update={"messages": named_messages})
        return super().preprocess(agent, ctx, named_request, llm)

    def recall(self, agent: Agent, ctx: ProcessContext, context: list[LLMMessage]) -> list[LLMMessage]:
        enriched_history = ctx.get_context().get("enriched_history", [])

        messages = self._extract_messages_from_history(enriched_history)

        # Deduplicate messages by their content

        unique_messages = self._deduplicate_messages(messages)

        messages = unique_messages

        if self.text_only:
            messages = self._filter_text_messages(messages)

        return messages
