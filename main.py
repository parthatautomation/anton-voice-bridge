"""
Anton Automation — Live Call Bridge (Pipecat Edition)
======================================================
Uses Pipecat's native Plivo serializer + Sarvam AI services (STT, LLM, TTS)
Based on Sarvam's official documentation: docs.sarvam.ai/api/integration

Requirements:
  pip install "pipecat-ai[websocket,sarvam,silero]" fastapi uvicorn httpx python-dotenv loguru aiohttp
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
from pipecat.audio.vad.silero import SileroVADAnalyzer
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

SYSTEM_PROMPT = """You are Akriti Patel, sales executive at Anton Automation — a conveyor system manufacturer in India.

LANGUAGE: Start in Hindi. Switch instantly to whatever language the caller uses. Never mix languages.

STYLE: Very short — max 1 sentence per response. Like a real phone call. Direct and warm.

YOUR TASK — collect these 5 details, one at a time:
1. Material to be conveyed (stone, coal, cement, etc.)
2. Weight per unit or per meter
3. Conveyor length needed (metres)
4. Where it will be used (plant, warehouse, quarry, etc.)
5. When they need it

Once you have all 5, say: "Perfect. I'll arrange a free site visit and quotation. Can I confirm your location?"

If not interested: thank briefly and end.
If wants human: say "Connecting you to our engineer now."

After your response add on a new line (NEVER SPEAK THIS):
###STATUS###{"product":null,"weight":null,"distance":null,"application":null,"timeline":null,"not_interested":false,"wants_human":false,"call_complete":false}"""


def normalize_india(num: str) -> str:
    digits = ''.join(filter(str.isdigit, str(num)))
    if digits.startswith('91') and len(digits) == 12:
        return '+' + digits
    elif len(digits) == 10:
        return '+91' + digits
    return '+' + digits


def extract_status(text: str) -> tuple:
    if "###STATUS###" in text:
        spoken, _, status_json = text.partition("###STATUS###")
        try:
            status = json.loads(status_json.strip())
        except json.JSONDecodeError:
            status = {}
        return spoken.strip(), status
    return text.strip(), {}


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
        "not_interested_reason": status.get("not_interested_reason", ""),
        "wants_human": status.get("wants_human", False),
        "frustrated": status.get("frustrated", False),
        "attempt_number": 1,
        "direction": "outbound",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(N8N_CALLBACK_URL, json=payload)
            logger.info(f"n8n callback: {resp.status_code}")
    except Exception as e:
        logger.error(f"n8n callback failed: {e}")


def strip_status(text: str) -> str:
    """Remove ###STATUS### JSON block from LLM response before TTS speaks it."""
    if "###STATUS###" in text:
        spoken = text.split("###STATUS###")[0].strip()
        return spoken
    return text.strip()


app = FastAPI(title="Anton Automation Voice Bridge")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "anton-voice-bridge", "pipecat": "enabled", "sarvam_key_length": len(SARVAM_API_KEY)}


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
    logger.info(f"Stream XML: {ws_url}")
    return PlainTextResponse(content=xml, media_type="text/xml")


@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    logger.info(f"WebSocket /ws/{token} connected")

    lead = CALL_CONTEXT.get(token, {"leadId": "", "name": "there", "interest": "your enquiry", "language": "hi-IN"})
    logger.info(f"Lead: name={lead['name']} lang={lead['language']} interest={lead['interest']}")

    transcript = []
    final_status = {}
    stream_id = ""
    call_id = ""

    # Get start event for stream_id and call_id
    try:
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
        data = json.loads(first_msg)
        if data.get("event") == "start":
            stream_id = data["start"].get("streamId", "")
            call_id = data["start"].get("callId", "")
            logger.info(f"Call started: stream={stream_id} call={call_id}")
        else:
            logger.warning(f"Expected start event, got: {data.get('event')}")
    except Exception as e:
        logger.error(f"Failed to get start event: {e}")
        await websocket.close()
        return

    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=PLIVO_AUTH_ID,
        auth_token=PLIVO_AUTH_TOKEN,
        params=PlivoFrameSerializer.InputParams(auto_hang_up=True),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            language="unknown",
        ),
        mode="transcribe",
    )

    tts = SarvamTTSService(
        api_key=SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="ishita",
            language="hi-IN",
            pace=1.1,
        ),
    )

    llm = SarvamLLMService(
        api_key=SARVAM_API_KEY,
        settings=SarvamLLMService.Settings(model="sarvam-105b"),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.frames.frames import TextFrame, LLMTextFrame

    class StatusStripperProcessor(FrameProcessor):
        """Strips ###STATUS### JSON from LLM text before TTS speaks it."""
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, (TextFrame, LLMTextFrame)):
                text = frame.text.strip()
                # Block entire frame if it contains or starts with ###STATUS###
                if "###STATUS###" in text:
                    spoken = text.split("###STATUS###")[0].strip()
                    if spoken:
                        new_frame = type(frame)(text=spoken)
                        await self.push_frame(new_frame, direction)
                    # Don't push the status part at all
                    return
            await self.push_frame(frame, direction)

    status_stripper = StatusStripperProcessor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        status_stripper,
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
        logger.info("Caller connected — sending Hindi greeting via TTS directly")
        greeting = f"Namaste {lead['name']} ji! Main Akriti Patel bol rahi hoon Anton Automation se. Aapne {lead['interest']} ke liye enquiry ki thi. Kya aap 2 minute baat kar sakte hain?"
        # Send greeting directly as TTS text frame — bypasses LLM for instant response
        from pipecat.frames.frames import TextFrame
        messages.append({"role": "assistant", "content": greeting})
        await task.queue_frames([TextFrame(text=greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Caller disconnected — posting to n8n")
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    spoken, status = extract_status(content)
                    transcript.append({"role": msg["role"], "content": spoken})
                    if status:
                        final_status.update(status)
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
