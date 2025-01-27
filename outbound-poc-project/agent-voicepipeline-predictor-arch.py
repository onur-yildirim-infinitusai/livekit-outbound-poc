from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from time import perf_counter
from typing import Annotated, AsyncIterable
from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
from colorama import Fore, Style
from components.call_conversation import Action, initialize_conversation, initialize_conversation_with_action_tree
from components.llm import InfinitusLLMStream, InfinitusDummyLLM
from components.chat_log_manager import ChatLogManager
# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

SYSTEM_PROMPT = """
You are an assistant calling the user to ask them a few questions.
Your interface with user will be voice.
You will be given a list of actions to perform, and you will need to identify the next action to perform. 
You can ONLY choose actions from the list and you can only utter the utterance of the provided action.
You will use the objectives to determine the next action to perform.
If the user is not answering the question, you will need to ask the question again.
If the action has outputs to be extracted and the outputs are not extracted correctly, you will need to ask the question again.

Start the conversation by saying the utterance of the current action.

Current action: {current_action}
The current action utterance to use: {action_utterance}
Pushback actions and utterances if needed: 
{pushback_actions}
Action actions and utterances if needed: 
{answer_actions}""" + ("""
Outputs to extract: {outputs_to_extract}""" if "{outputs_to_extract}" else "")

ACTION_PROMPT = """
Your objective is extracting the outputs for the current action.
Continue the conversation by saying the utterance of the current action.
If the user is not answering the question, you will need to ask the question again.
If the outputs are not extracted correctly, you will need to ask the question again.
If user asks a question, you can answer it by using the answer actions.

Current action: {current_action}
The current action utterance to use: {action_utterance}
Pushback actions and utterances if needed:
{pushback_actions}
Action actions and utterances if needed:
{answer_actions}""" + ("""
Outputs to extract: {outputs_to_extract}""" if "{outputs_to_extract}" else "")


# state_tracker = initialize_conversation()
state_tracker = initialize_conversation_with_action_tree()
chat_log_manager = ChatLogManager()

def get_prompt(prompt: str, action: Action, pushback_actions: list[Action], answer_actions: list[Action]):
    pushback_actions_str = "\n".join([f"{action.name}: {action.utterance} ({action.instructions})" for action in pushback_actions])
    answer_actions_str = "\n".join([f"{action.name}: {action.utterance} ({action.instructions})" for action in answer_actions])
    return prompt.format(current_action=action.name, action_utterance=action.utterance, pushback_actions=pushback_actions_str, answer_actions=answer_actions_str, outputs_to_extract=action.outputs)

async def entrypoint(ctx: JobContext):
    global _default_instructions, outbound_trunk_id
    print(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    user_identity = "phone_user"
    # the phone number to dial is provided in the job metadata
    phone_number = ctx.job.metadata
    print(f"dialing {phone_number} to room {ctx.room.name}")

    # `create_sip_participant` starts dialing the user
    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=phone_number,
            participant_identity=user_identity,
        )
    )

    # a participant is created as soon as we start dialing
    participant = await ctx.wait_for_participant(identity=user_identity)

    # start the agent, either a VoicePipelineAgent or MultimodalAgent
    # this can be started before the user picks up. The agent will only start
    # speaking once the user answers the call.
    current_action = state_tracker.get_current_action()
    pushback_actions = state_tracker.get_pushback_actions()
    answer_actions = state_tracker.get_answer_actions()
    system_prompt = get_prompt(SYSTEM_PROMPT, current_action, pushback_actions, answer_actions)
    print(system_prompt)

    await run_voice_pipeline_agent(ctx, participant, system_prompt)

def before_tts_cb(
        agent: VoicePipelineAgent,
        text: str | AsyncIterable[str]   
) -> str | AsyncIterable[str]:
    if isinstance(text, str):
        print(Fore.YELLOW + f"before_tts_cb (string): {text}" + Style.RESET_ALL)
        return text
    else:
        # Collect all chunks into a single string
        async def generate_full_sentence():
            result = ""
            async for chunk in text:
                result += chunk
            print(Fore.YELLOW + f"before_tts_cb (chunks): {result}" + Style.RESET_ALL)
            yield result
        return generate_full_sentence()
    
async def before_llm_cb(
    agent: VoicePipelineAgent, chat_ctx: llm.ChatContext
) -> llm.LLMStream:
    print(Fore.YELLOW + "Custom before LLM callback invoked", chat_ctx.messages[-1], Style.RESET_ALL)
    print(Fore.YELLOW + f"before_llm_cb: {chat_ctx}" + Style.RESET_ALL)
    chat_log_manager.print_chat_log()
    current_action = state_tracker.get_current_action()
    current_action_utterance = current_action.utterance
    current_action_outputs = current_action.outputs

    if current_action_outputs:
        for output in current_action_outputs:
            state_tracker.get_outputs_bag().set_output(output, "yes")

    next_action = state_tracker.next_action()
    if next_action:
        next_action_utterance = next_action.utterance
        return InfinitusLLMStream(llm=agent.llm, chat_ctx=chat_ctx, value=next_action_utterance)

    # NOTE: set add_to_chat_ctx=True will add the message to the end
    #   of the chat context of the function call for answer synthesis
    # speech_handle = agent.say(source="Give me one second!", add_to_chat_ctx=True)  # noqa: F841
    
    # return InfinitusLLMStream(llm=agent.llm, chat_ctx=chat_ctx, value="You have reached a callback! Please continue the conversation.You have reached a callback! Please continue the conversation.You have reached a callback! Please continue the conversation.")
    return InfinitusLLMStream(llm=agent.llm, chat_ctx=chat_ctx, value=current_action_utterance)


async def run_voice_pipeline_agent(
    ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str
):
    print("starting voice pipeline agent")


    initial_ctx = llm.ChatContext().append(
        role="system",
        text=instructions, # TODO: remove this
    )

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-phonecall"),
        # llm=openai.LLM(),
        llm=InfinitusDummyLLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        # fnc_ctx=CallActions(api=ctx.api, participant=participant, room=ctx.room),
        before_tts_cb=before_tts_cb,
        before_llm_cb=before_llm_cb,
    )

    agent.start(ctx.room, participant)

    @agent.on("user_started_speaking")
    def user_started_speaking():
        print(Fore.BLUE + "user_started_speaking" + Style.RESET_ALL)

    @agent.on("user_stopped_speaking")
    def user_stopped_speaking():
        print(Fore.BLUE + "user_stopped_speaking" + Style.RESET_ALL)

    @agent.on("agent_started_speaking")
    def agent_started_speaking():
        print(Fore.CYAN + "agent_started_speaking" + Style.RESET_ALL)

    @agent.on("agent_stopped_speaking")
    def agent_stopped_speaking():
        print(Fore.CYAN + "agent_stopped_speaking" + Style.RESET_ALL)

    @agent.on("user_speech_committed")
    def user_speech_committed(msg: llm.ChatMessage):
        print(Fore.BLUE + f"user_speech_committed: {msg}" + Style.RESET_ALL)
        chat_log_manager.add_message(msg.content, "USER")
    @agent.on("agent_speech_committed")
    def agent_speech_committed(msg: llm.ChatMessage):
        print(Fore.CYAN + f"agent_speech_committed: {msg}" + Style.RESET_ALL)
        chat_log_manager.add_message(msg.content, "AGENT")
    @agent.on("agent_speech_interrupted")
    def agent_speech_interrupted(msg: llm.ChatMessage):
        print(Fore.CYAN + f"agent_speech_interrupted: {msg}" + Style.RESET_ALL)
        chat_log_manager.add_message(msg.content, "AGENT - INTERRUPTED")

    # @agent.on("function_calls_collected")
    # def function_calls_collected(fnc_call_infos: list[llm.FunctionCallInfo]):
    #     print(Fore.YELLOW + "function_calls_collected" + Style.RESET_ALL)
    #     for fnc_call_info in fnc_call_infos:
    #         print(Fore.YELLOW + f"Function call info: {fnc_call_info}" + Style.RESET_ALL)
    #         if fnc_call_info.function_info.name == "call_if_output_is_extracted":
    #             print(f"Output extracted: {fnc_call_info}")
    #         elif fnc_call_info.function_info.name == "call_if_output_is_not_extracted":
    #             print(Fore.YELLOW + f"No output extracted: {fnc_call_info}" + Style.RESET_ALL)

    # @agent.on("function_calls_finished")
    # def function_calls_finished(fnc_call_infos: list[llm.FunctionCallInfo]):
    #     print(Fore.LIGHTYELLOW_EX + f"function_calls_finished: {fnc_call_infos}" + Style.RESET_ALL)

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    write_task = asyncio.create_task(chat_log_manager.write_to_file())

    async def finish_queue():
        chat_log_manager.log_queue.put_nowait(None)
        await write_task

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(finish_queue)

    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


class CallActions(llm.FunctionContext):
    """
    Detect user intent and perform actions by following the instructions. Always use the most recent instruction.
    """

    def __init__(
        self, *, api: api.LiveKitAPI, participant: rtc.RemoteParticipant, room: rtc.Room
    ):
        super().__init__()

        self.api = api
        self.participant = participant
        self.room = room

    async def hangup(self):
        try:
            await self.api.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=self.room.name,
                    identity=self.participant.identity,
                )
            )
        except Exception as e:
            # it's possible that the user has already hung up, this error can be ignored
            print(f"received error while ending call: {e}")

    @llm.ai_callable()
    async def end_call(self):
        """Called when the user wants to end the call"""
        print(Fore.RED + f"ending the call for {self.participant.identity}" + Style.RESET_ALL)
        await self.hangup()
        print(json.dumps(state_tracker.get_outputs_bag().to_dict()))
    @llm.ai_callable()
    async def detected_answering_machine(self):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        print(Fore.RED + f"detected answering machine for {self.participant.identity}" + Style.RESET_ALL)
        await self.hangup()

    @llm.ai_callable()
    async def call_if_output_is_extracted(self,
                                          extracted_output_name: Annotated[str, "The name of the extracted output if there is any"],
                                          extracted_output_value: Annotated[str, "The extracted output value from the user response if there is any"],
                                          ):
        """
        This function is used to determine the next action to perform when the user response is clear and related to the current action.
        If the extracted output name is not empty, call the function with the extracted output name and value.
        """

        old_action = state_tracker.get_current_action()

        print(Fore.GREEN + f"Extracted output name: {extracted_output_name}" + Style.RESET_ALL)
        print(Fore.GREEN + f"Extracted output value: {extracted_output_value}" + Style.RESET_ALL)

        if extracted_output_name:
            print(Fore.GREEN + f"Bag before: {state_tracker.get_outputs_bag().to_dict()}" + Style.RESET_ALL)
            state_tracker.get_outputs_bag().set_output(extracted_output_name, extracted_output_value)
            print(Fore.GREEN + f"Bag after: {state_tracker.get_outputs_bag().to_dict()}" + Style.RESET_ALL)

        state_tracker.next_action()
        current_action = state_tracker.get_current_action()
        pushback_actions = state_tracker.get_pushback_actions()
        answer_actions = state_tracker.get_answer_actions()
        print(Fore.GREEN + f"Next action in call_if_output_is_extracted: {current_action}" + Style.RESET_ALL)
        action_prompt = get_prompt(ACTION_PROMPT,current_action, pushback_actions, answer_actions)
        return f"""
        Output is extracted for {old_action.name} and the value is set as {extracted_output_value}. 
        Your new instructions are: \n{action_prompt}
        """

    @llm.ai_callable()
    async def call_if_output_is_not_extracted(self, 
                                   response_type: Annotated[str, "The intent of the user response for the uttered action. It can be ONLY one of the following: 'answer', 'question' or 'other'."],
                                   response_text: Annotated[str, "The transcription of the user response"],
                                   ):
        """
        This function is used to determine the next action to perform when the user response is not clear or the user response is not related to the current action or there is no output to extract.
        If the extracted output name is empty, call the function with the user response type and incoming user response text.
        """
        print(Fore.YELLOW + f"Response type: {response_type}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Response text: {response_text}" + Style.RESET_ALL)

        current_action = state_tracker.get_current_action()
        print(Fore.YELLOW + f"Current action in call_if_output_is_not_extracted: {current_action}" + Style.RESET_ALL)

        if current_action.outputs:
            return f"""
            Output is not extracted for {current_action.name}.
            Use the most recent instructions to continue the conversation.
            """
        else:
            state_tracker.next_action()
            new_action = state_tracker.get_current_action()
            pushback_actions = state_tracker.get_pushback_actions()
            answer_actions = state_tracker.get_answer_actions()
            print(Fore.YELLOW + f"Next action in call_if_output_is_not_extracted: {new_action}" + Style.RESET_ALL)
            action_prompt = get_prompt(ACTION_PROMPT,new_action, pushback_actions, answer_actions)

            return f"""
            There is no output to extract for {new_action.name} so you can continue the conversation.
            Your new instructions are: \n{action_prompt}
            """ 

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        raise ValueError(
            "SIP_OUTBOUND_TRUNK_ID is not set. Please follow the guide at https://docs.livekit.io/agents/quickstarts/outbound-calls/ to set it up."
        )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # giving this agent a name will allow us to dispatch it via API
            # automatic dispatch is disabled when `agent_name` is set
            agent_name="outbound-caller",
            # prewarm by loading the VAD model, needed only for VoicePipelineAgent
            prewarm_fnc=prewarm,
        )
    )
