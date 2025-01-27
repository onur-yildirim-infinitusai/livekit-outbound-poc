from livekit.agents import llm, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
import uuid
class InfinitusDummyLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        
    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> llm.LLMStream:
        raise RuntimeError("DummyLLM should not be used - all LLM calls should go through before_llm_cb")
    
class InfinitusLLMStream(llm.LLMStream):
    def __init__(self,
                 llm: llm.LLM,
                 chat_ctx: llm.ChatContext,
                 value: str,
                 fnc_ctx: llm.FunctionContext | None = None,
                 conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
                 ):
        super().__init__(llm=llm,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options)
        self.value = value
        
    async def _run(self) -> None:
        request_id = str(uuid.uuid4())
        chunk = llm.ChatChunk(
                request_id=request_id,
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(
                            content=self.value,
                            role="assistant"
                        ),
                        index=0
                    )
                ]
            )
            
        self._event_ch.send_nowait(chunk)
        pass
