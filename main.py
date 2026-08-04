"""
Anton Automation — Live Call Bridge (Pipecat Edition)
Clean architecture: Plivo → WebSocket → Pipecat → Sarvam AI (STT/LLM/TTS)
"""

import asyncio
import hashlib
import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://web-production-710a9d.up.railway.app")
N8N_CALLBACK_URL = os.getenv("N8N_CALLBACK_URL", "https://workflow.parthkalyani.in/webhook/anton-call-callback")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "MAMTDKN2NIZDKTZJQ2ZI")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
WS_BASE_URL = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
CALL_CONTEXT: dict = {}

SYSTEM_PROMPT = """You are Akriti, sales agent at Anton Automation (conveyor system manufacturer, India).

STRICT RULES:
- NEVER say "Namaste" or repeat the greeting again — it was already said once at the start
- NEVER introduce yourself again — already done
- LANGUAGE: Match caller's language instantly. Never mix languages.
- LENGTH: Maximum 1 sentence per response. Phone call style.

CONVERSATION FLOW:
- After greeting confirmed (Haan ji / Hello / OK / Ji): Ask "Aap kaunsa material convey karna chahte hain?"
- After material: Ask about weight
- After weight: Ask about conveyor length
- After length: Ask about location/application  
- After location: Ask about timeline
- After all 5: Say "Perfect! Main free site visit aur quotation arrange karta hoon."

If asked about company: "Hum stone crusher, cement, pharma aur mining ke liye custom conveyor systems banate hain."
If not interested: Thank briefly and end call.
If wants engineer/human: "Abhi connect karti hoon."

GOAL: Collect material, weight, length, location, timeline — one question at a time."""


def normalize_india(num: str) -> str:
    digits = ''.join(filter(str.isdigit, str(num)))
    if digits.startswith('91') and len(digits) == 12:
        return '+' + digits
    elif len(digits) == 10:
        return '+91' + digits
    return '+' + digits


async def post_to_n8n(lead: dict, transcript: list, status: dict):
    full_transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in transcript)
    payload = {
        "mobile": lead.get("leadId", ""),
        "phone": lead.get("leadId", ""),
        "call_status": "answered" if transcript else "no-answer",
        "lead_name": lead.get("name", ""),
        "interest": lead.get("interest", ""),
        "transcript": full_transcript,
        "criterion_scores": {k: 20 for k in ["product", "weight", "distance", "application", "timeline"] if status.get(k)},
        "criterion_answers": {k: status[k] for k in ["product", "weight", "distance", "application", "timeline"] if status.get(k)},
        "not_interested": status.get("not_interested", False),
        "wants_human": status.get("wants_human", False),
        "attempt_number": 1,
        "direction": "outbound",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(N8N_CALLBACK_URL, json=payload)
            logger.info(f"n8n callback: {resp.status_code}")
    except Exception as e:
        logger.error(f"n8n callback failed: {e}")


app = FastAPI(title="Anton Automation Voice Bridge")


@app.get("/health")
async def health():
    plivo_token_set = len(PLIVO_AUTH_TOKEN) > 10
    return {
        "status": "ok",
        "service": "anton-voice-bridge",
        "pipecat": "enabled",
        "sarvam_key_length": len(SARVAM_API_KEY),
        "plivo_auth_id": PLIVO_AUTH_ID[:8] + "...",
        "plivo_token_set": plivo_token_set,
    }


@app.post("/plivo-answer")
async def plivo_answer(request: Request):
    params = dict(request.query_params)
    lead_id = params.get("lead_id", "")
    name = params.get("name", "there")
    interest = params.get("interest", "your enquiry")
    language = params.get("language", "hi-IN")

    token = hashlib.md5(f"{lead_id}{name}{language}".encode()).hexdigest()[:12]
    CALL_CONTEXT[token] = {"leadId": lead_id, "name": name, "interest": interest, "language": language}
    logger.info(f"plivo-answer: token={token} lead={name} lang={language}")

    ws_url = f"{WS_BASE_URL}/ws/{token}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" streamTimeout="3600" contentType="audio/x-mulaw;rate=8000">{ws_url}</Stream>
</Response>"""
    return PlainTextResponse(content=xml, media_type="text/xml")


@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    logger.info(f"WebSocket /ws/{token} connected")

    lead = CALL_CONTEXT.get(token, {
        "leadId": "", "name": "there", "interest": "your enquiry", "language": "hi-IN"
    })
    logger.info(f"Lead: name={lead['name']} lang={lead['language']}")

    transcript = []
    final_status = {}
    stream_id = ""
    call_id = ""

    try:
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
        data = json.loads(first_msg)
        if data.get("event") == "start":
            stream_id = data["start"].get("streamId", "")
            call_id = data["start"].get("callId", "")
            logger.info(f"Call started: stream={stream_id} call={call_id}")
    except Exception as e:
        logger.error(f"Failed to get start event: {e}")
        await websocket.close()
        return

    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=PLIVO_AUTH_ID,
        auth_token=PLIVO_AUTH_TOKEN,
        params=PlivoFrameSerializer.InputParams(auto_hang_up=False),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=8000,
            audio_in_passthrough=False,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            language="unknown",
            vad_signals=True,
            negative_speech_threshold=0.6,
            min_speech_frames=5,
        ),
        mode="transcribe",
    )

    tts = SarvamTTSService(
        api_key=SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3-beta",
            voice="ishita",
            language="hi-IN",
            pace=1.1,
        ),
    )

    llm = SarvamLLMService(
        api_key=SARVAM_API_KEY,
        settings=SarvamLLMService.Settings(
            model="sarvam-105b",
            max_tokens=150,
        ),
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("on_client_connected fired — sending greeting")
        greeting = (
            f"Namaste {lead['name']} ji! "
            f"Main Akriti Patel hoon Anton Automation se. "
            f"Aapne {lead['interest']} ke liye enquiry ki thi — "
            f"kya aap abhi baat kar sakte hain?"
        )
        # Add to context so LLM knows greeting was already said
        messages.append({"role": "assistant", "content": greeting})
        # Also add a system reminder so LLM never repeats it
        messages.append({"role": "system", "content": "The greeting above was already spoken. Do NOT repeat it. Ask the first qualification question when the caller responds."})
        await task.queue_frames([TextFrame(text=greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Caller disconnected — posting to n8n")
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                content_text = msg.get("content", "")
                if isinstance(content_text, str) and content_text.strip():
                    transcript.append({"role": msg["role"], "content": content_text.strip()})
        await task.cancel()
        await post_to_n8n(lead, transcript, final_status)

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.post("/dial-out")
async def dial_out(request: Request):
    body = await request.json()
    plivo_auth_id = body.get("plivo_auth_id", PLIVO_AUTH_ID).strip()
    plivo_auth_token = body.get("plivo_auth_token", PLIVO_AUTH_TOKEN).strip()

    to_number = normalize_india(body["to"])
    from_number = normalize_india(body["from"])

    lead_id = body.get("lead_id", "")
    name = body.get("name", "there")
    interest = body.get("interest", "your enquiry")
    language = body.get("language", "hi-IN")

    answer_url = (
        f"{PUBLIC_BASE_URL}/plivo-answer"
        f"?lead_id={lead_id}&name={name}&interest={interest}&language={language}"
    )

    logger.info(f"Dial-out: from={from_number} to={to_number}")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"https://api.plivo.com/v1/Account/{plivo_auth_id}/Call/",
            auth=(plivo_auth_id, plivo_auth_token),
            json={"from": from_number, "to": to_number, "answer_url": answer_url, "answer_method": "POST"},
        )

    result = resp.json()
    logger.info(f"Plivo: {resp.status_code} uuid={result.get('request_uuid', '')}")
    return JSONResponse(content=result, status_code=resp.status_code)


@app.post("/plivo-hangup")
async def plivo_hangup(request: Request):
    logger.info("Plivo hangup")
    return JSONResponse(content={"received": True})
