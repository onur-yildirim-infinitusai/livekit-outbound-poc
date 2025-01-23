from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from time import perf_counter
from typing import Annotated
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


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
# _default_instructions = (
#     "You are a scheduling assistant for a dental practice. Your interface with user will be voice."
#     "You will be on a call with a patient who has an upcoming appointment. Your goal is to confirm the appointment details."
#     "As a customer service representative, you will be polite and professional at all times. Allow user to end the conversation."
# )

SYSTEM_PROMPT = """
 {default_instructions}

 User Information:
 {user_info}

 Sections in the questionnaire:
 {sections}

"""
_default_instructions = (
   "You are an expert care coordinator, calling on behalf of a healthcare company called 'Zing Health' that is calling a patient to do a health risk assessment."
   "Your interface with user will be voice."
   "Your goal is to follow the rules in the sections of the questionnaire provided below and use the correct utterances provided for the actions."
   "Rules:"
   "- Confirm that your response is on the list of allowed responses before you respond with it."
   "- Always interact with the patient with respect, professional courtesy and empathy."
   "- If there is no relevant clarification question, you may repeat your previous request/answer to the agent by responding with 'RepeatPreviousAction' action."
   "- If the patient asks you to repeat yourself or to repeat your most recent question, you should respond with 'RepeatPreviousAction' action."
#    "- In some cases, if the patient either doesn't know the answer to your question or is unable to look up the information you have requested, it is safe to continue to the next appropriate question."
   "- You have to use one of the allowed actions in your response for the section of the questionnaire you are currently on."
)

GREETING_SECTION = {
    "section_name": "GREETING_SECTION",
    "allowed_actions": [
        {"action": "HRAAskToSpeakToMember", "utterance": "Hi good morning, may I speak to <user_info.first_name> please?"},
        {"action": "HRAGiveGreeting", "utterance": "Hi there! This is Eva and I'm a digital assistant calling from Zing Health, your Medicare plan. I'm calling to ask a few health related questions to ensure you can get the most out of your plan with supplemental benefits, as well as the care you need. Do you by chance have a few minutes right now to go over those questions with me?"},
        {"action": "CallbackLater", "utterance": "I will call you back later. Goodbye!"},
        {"action": "EndCall", "utterance": "Sorry for bothering you. Goodbye!"},
    ],
    "instructions":
    """
    - Greet the patient with a friendly and professional tone.
    - If the patient is not the right person, hangup the call.
    - If they are not available right now, tell them you will call them back later and hangup the call.
    - If the patient is the right person, proceed to the next section.
    """
}

HIPAA_SECTION = {
    "section_name": "HIPAA_SECTION",
    "allowed_actions": [
        {"action": "HRAConfirmDOB", "utterance": "Ok great, thanks. Your account security is very important to us. Can you please confirm your date of birth?"},
        {"action": "RetryPatientDOB", "utterance": "Sorry can you please repeat your answer?"},
        {"action": "ThankYou", "utterance": "Thank you for your response. It's matched with our records."},
        {"action": "EndCall", "utterance": "Date of birth is not matched with our records. We will call you back later."},
    ],
    "instructions":
    """
    - This should be the first section of the questionnaire.
    - You get two attempts to verify their date of birth
    - If they fail both attempts, you must gracefully exit the call
    - If they answer correctly, proceed to HEALTH_ASSESSMENT_SECTION
    - For each HIPAA question:
     - Ask the question clearly
        - If they don't answer, do one pushback
        - If they still don't answer after pushback, gracefully exit
        - If the answer is not valid after pushback, gracefully exit
        - If the answer is valid, proceed to the next section
    """
}
HEALTH_ASSESSMENT_SECTION = {
    "section_name": "HEALTH_ASSESSMENT_SECTION",
    "allowed_actions": [
        {"action": "HRAIntroMedicalSection", "utterance": "Thank you! This should take less than 10 minutes. And just so you know, I'm an automated system and your call may be recorded for quality and training purposes. First, I'm going to list some medical conditions, and can you please let me know if you've been diagnosed with any of them?"},
        {"action": "HRAAskBehavioralHealthIssues", "utterance": "Have you been diagnosed with any behavioral health issues such as depression or anxiety?"},
        {"action": "WhichOneSpesifically", "utterance": "Which one specifically?"},
        {"action": "EndCall", "utterance": "Thank you for your responses. Goodbye!"},
    ],
    "instructions":
    """
    - For each health assessment question:
     - Ask the question clearly
        - If they don't answer, do one pushback by asking the same action again.
        - If they do answer, capture their response and move to next question.
        - If they still don't answer after pushback, move to next question.
        - If all the questions are asked, hangup the call.
    """
}

USER_INFO = {
    "first_name": "Adam",
    "last_name": "Smith",
    "dob": "1992-01-01",
}

def build_prompt(default_instructions: str, user_info: dict[str, str], sections: list[dict]) -> str:
    sections_prompt = ""
    for section in sections:
        actions_prompt = ""
        for action in section['allowed_actions']:
            actions_prompt += f"""
            {action['action']}: {action['utterance']}
            """
        sections_prompt += f"""
        {section['section_name']}: 
        {section['instructions']}
        Allowed actions and utterances -- you can only use one of the following actions in your response::
        {actions_prompt}
        """
    user_info_prompt = ""
    for key, value in user_info.items():
        user_info_prompt += f"{key}: {value}\n"

    return SYSTEM_PROMPT.format(
        default_instructions=default_instructions,
        user_info=user_info_prompt,
        sections=sections_prompt
    )



async def entrypoint(ctx: JobContext):
    global _default_instructions, outbound_trunk_id
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    user_identity = "phone_user"
    # the phone number to dial is provided in the job metadata
    phone_number = ctx.job.metadata
    logger.info(f"dialing {phone_number} to room {ctx.room.name}")

    # look up the user's phone number and appointment details

    #Build sections prompt
    sections = [GREETING_SECTION, HIPAA_SECTION, HEALTH_ASSESSMENT_SECTION]
    instructions = build_prompt(
        default_instructions=_default_instructions,
        user_info=USER_INFO,
        sections=sections
    )

    # instructions = (
    #     _default_instructions
    #     + "The customer's name is Jayden. His appointment is next Tuesday at 3pm."
    # )

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
    run_voice_pipeline_agent(ctx, participant, instructions)
    run_multimodal_agent(ctx, participant, instructions)

    # in addition, you can monitor the call status separately
    start_time = perf_counter()
    while perf_counter() - start_time < 30:
        call_status = participant.attributes.get("sip.callStatus")
        if call_status == "active":
            logger.info("user has picked up")
            return
        elif call_status == "automation":
            # if DTMF is used in the `sip_call_to` number, typically used to dial
            # an extension or enter a PIN.
            # during DTMF dialing, the participant will be in the "automation" state
            pass
        elif call_status == "hangup":
            # user hung up, we'll exit the job
            logger.info("user hung up, exiting job")
            break
        await asyncio.sleep(0.1)

    logger.info("session timed out, exiting job")
    ctx.shutdown()


class CallActions(llm.FunctionContext):
    """
    Detect user intent and perform actions
    """

    def __init__(
        self, *, api: api.LiveKitAPI, participant: rtc.RemoteParticipant, room: rtc.Room, patient_info: dict[str, str]
    ):
        super().__init__()

        self.api = api
        self.participant = participant
        self.room = room
        self.patient_info = patient_info
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
            logger.info(f"received error while ending call: {e}")

    @llm.ai_callable()
    async def end_call(self):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")
        await self.hangup()

    # @llm.ai_callable()
    # async def look_up_availability(
    #     self,
    #     date: Annotated[str, "The date of the appointment to check availability for"],
    # ):
    #     """Called when the user asks about alternative appointment availability"""
    #     logger.info(
    #         f"looking up availability for {self.participant.identity} on {date}"
    #     )
    #     asyncio.sleep(3)
    #     return json.dumps(
    #         {
    #             "available_times": ["1pm", "2pm", "3pm"],
    #         }
    #     )

    # @llm.ai_callable()
    # async def confirm_appointment(
    #     self,
    #     date: Annotated[str, "date of the appointment"],
    #     time: Annotated[str, "time of the appointment"],
    # ):
    #     """Called when the user confirms their appointment on a specific date. Use this tool only when they are certain about the date and time."""
    #     logger.info(
    #         f"confirming appointment for {self.participant.identity} on {date} at {time}"
    #     )
    #     return "reservation confirmed"

    @llm.ai_callable()
    async def look_up_patient_name(self,
                                   name: Annotated[str, "The name of the patient"]):
        """Called when the user answers the HRAAskToSpeakToMember question"""
        logger.info(f"asking if we are talking to the right person for {self.participant.identity} with value {name}")
        return name == self.patient_info["first_name"]
    
    @llm.ai_callable()
    async def look_up_patient_dob(self,
                                  dob: Annotated[str, "The date of birth to look up and format is YYYY-MM-DD"]):
        """Called when the user provides a date of birth"""
        logger.info(f"looking up patient's date of birth for {self.participant.identity} with value {dob}")
        return dob == self.patient_info["dob"]
    
    @llm.ai_callable()
    async def evaluate_patient_behavioral_health_answer(self,
                                  behavioral_health: Annotated[str, "The behavioral health answer to check whether the patient has been diagnosed with any behavioral health issues such as depression or anxiety"]):
        """Called when the user provides a behavioral health answer"""
        logger.info(f"looking up patient's answer for behavioral health for {self.participant.identity} with value {behavioral_health}")
        return behavioral_health.lower() in ["depression", "anxiety"]
    
    
    @llm.ai_callable()
    async def detected_answering_machine(self):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()


def run_voice_pipeline_agent(
    ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str
):
    logger.info("starting voice pipeline agent")

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=instructions,
    )

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-phonecall"),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=CallActions(api=ctx.api, participant=participant, room=ctx.room, patient_info=USER_INFO),
    )

    agent.start(ctx.room, participant)


def run_multimodal_agent(
    ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str
):
    logger.info("starting multimodal agent")

    # https://docs.livekit.io/agents/openai/customize/parameters/
    model = openai.realtime.RealtimeModel(
        instructions=instructions,
        modalities=["audio", "text"],
    )
    agent = MultimodalAgent(
        model=model,
        fnc_ctx=CallActions(api=ctx.api, participant=participant, room=ctx.room, patient_info=USER_INFO),
    )

    @agent.on("user_started_speaking")
    def user_started_speaking():
        print("user_started_speaking")

    @agent.on("user_stopped_speaking")
    def user_stopped_speaking():
        print("user_stopped_speaking")

    @agent.on("agent_started_speaking")
    def agent_started_speaking():
        print("agent_started_speaking")

    @agent.on("agent_stopped_speaking")
    def agent_stopped_speaking():
        print("agent_stopped_speaking")

    @agent.on("user_speech_committed")
    def user_speech_committed(msg: llm.ChatMessage):
        print(f"user_speech_committed: {msg}")

    @agent.on("agent_speech_committed")
    def agent_speech_committed(msg: llm.ChatMessage):
        print(f"agent_speech_committed: {msg}")

    @agent.on("agent_speech_interrupted")
    def agent_speech_interrupted():
        print("agent_speech_interrupted")

    @agent.on("function_calls_collected")
    def function_calls_collected(fnc_call_infos: list[llm.FunctionCallInfo]):
        print(f"function_calls_collected: {fnc_call_infos}")

    @agent.on("function_calls_finished")
    def function_calls_finished(fnc_call_infos: list[llm.FunctionCallInfo]):
        print(f"function_calls_finished: {fnc_call_infos}")

    @agent.on("metrics_collected")
    def metrics_collected(metrics: metrics.MultimodalLLMMetrics):
        print(f"metrics_collected: {metrics}")

    agent.start(ctx.room, participant)


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
