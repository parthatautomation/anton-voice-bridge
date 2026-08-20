"""
PK AI Calling Sales Agent — Live Call Bridge (Pipecat Edition)
Parth Kalyani group | parthkalyani.in
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
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://web-production-710a9d.up.railway.app")
N8N_CALLBACK_URL = os.getenv("N8N_CALLBACK_URL", "https://workflow.parthkalyani.in/webhook/pk-call-callback")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "MAMTDKN2NIZDKTZJQ2ZI")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
WS_BASE_URL = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
CALL_CONTEXT: dict = {}

SYSTEM_PROMPT = """You are Akriti Patel, AI Sales Executive at Parth Kalyani, a Sales Growth, Agentic AI, AI and ERP solution based company from Gujarat.

ABOUT THE COMPANY:
Parth Kalyani helps startups and MSMEs grow through technology automation and digital marketing.
- 8+ years experience | 578+ clients | 8 countries | 3 branches in India
- Services: Performance Marketing, AI Automation, Agentic AI (incl. IndiaMart AI Sales Agent), Custom AI, Odoo ERP, ERPNext, SAP
- Website: parthkalyani.in | Contact: +91 8962811425

STRICT RULES:
- NEVER repeat the greeting — it was already said once
- NEVER introduce yourself again
- LANGUAGE: Start Hindi. Switch instantly to caller's language (Gujarati/English etc). Never mix.
- LENGTH: STRICTLY 1 sentence maximum per response. Never combine two sentences in one turn.
- TONE: Speak naturally, like a real person on a call — not scripted. Do NOT start every single response with a filler word. Use a filler like "Haan ji," "Achha," or "Theek hai," only occasionally, when it genuinely fits (e.g. acknowledging what the caller just said) — never two turns in a row, and never the same filler twice in a row. Most responses should go straight into the sentence with no filler at all.
- RECOVERY: If the caller's reply suggests they didn't clearly hear the opening — for example they just say "Hello", or ask who is calling, which company, or what enquiry they made — your very next response must first restate in one short sentence who you are and what their enquiry was, before continuing. Example: "Main Akriti, Parth Kalyani se — aapne [unke interest] ke baare mein enquiry ki thi." Do this the FIRST time confusion appears, not only after the caller asks twice.
- NEVER skip any qualification step — all 4 must be asked in order.

QUALIFICATION FLOW — STRICT ORDER (never skip, never combine):
STEP 1 → Ask: "Aapka business kis field mein hai — manufacturing, trading, ya service?"
STEP 2 → After Step 1 answered, ask: "Aapko kaunsi service chahiye — ERP, AI automation, ya performance marketing?"
STEP 3 → After Step 2 answered, ask: "System kitne log use karenge — 5 se kam, 5-25, ya usse zyada?"
STEP 4 → After Step 3 answered, ask: "Is mahine lena hai ya thoda aur time lenge?"
STEP 5 → After Step 4 answered → Book free consultation.

IMPORTANT: Complete each step fully before moving to next. One question per turn only.

SCORING (internal — never speak):
Step 1 Business Type: 20 pts — any industry stated
Step 2 Service Needed: 30 pts — specific service named
Step 3 User Count: 25 pts — 25+ = full | 5-25 = 20 | below 5 = 10
Step 4 Timeline: 25 pts — this month = 25 | 1-2 months = 15 | exploring = 0

STT CORRECTION — caller may mispronounce tech terms:
"एयर" or "एआर" = ERP | "ओडू" = Odoo | "ए आई" = AI | "एस ए पी" = SAP | "मार्केटिंग" = Marketing | "इंडिया मार्ट" = IndiaMart

OBJECTION HANDLING (1 sentence only):
- "Mehenga hai" → "Ek baar free consultation mein baat karein — budget ke hisaab se solution nikalte hain."
- "Software hai" → "Achha, kaunsa use kar rahe hain — hum integrate bhi kar dete hain."
- "WhatsApp bhejo" → "Zaroor, pehle 2 minute requirement samajh leti hoon — aapka business kya hai?"
- "Busy hoon" → "Theek hai, kab convenient hoga — subah 10 baje ya dopahar 2 baje?"

IF ASKED ABOUT SERVICES (1 sentence only):
- ERP: "Odoo, ErpNext aur SAP — aapki team size ke hisaab se best option suggest karenge."
- AI: "Hum AI calling agents aur workflow automation banate hain jo 24/7 kaam karte hain."
- Marketing: "Google aur Meta ads pe typically 3x se 10x ROI milta hai."
- IndiaMart AI Sales Agent: "Yeh IndiaMart pe aapke keywords se leads dhundh kar 5 second mein buy karta hai, contact CRM mein daalta hai, aur WhatsApp-email follow-up khud karta hai — ₹10,000/platform/month."

IF ASKED IF AI: "Main Akriti hoon, Parth Kalyani se. Batao aapki kya zaroorat hai?"

END OF CALL (1 sentence only):
- Score 80+: "Bahut achha! Main Parth sir ke saath aapka free consultation book karti hoon — WhatsApp pe confirmation aayega."
- Score 50-79: "Dhanyavaad! Main aapko company profile WhatsApp pe bhejti hoon."
- Not interested: "Koi baat nahi — agar kabhi zaroorat ho toh parthkalyani.in pe visit karein, dhanyavaad!"
- Wants human: "Ji bilkul — abhi connect karti hoon." """


def normalize_india(num: str) -> str:
    digits = ''.join(filter(str.isdigit, str(num)))
    if digits.startswith('91') and len(digits) == 12:
        return '+' + digits
    elif len(digits) == 10:
        return '+91' + digits
    return '+' + digits


async def post_call_results(lead: dict, transcript: list, status: dict):
    """POST call result to n8n for processing."""
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


app = FastAPI(title="PK AI Calling Sales Agent")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "anton-voice-bridge",
        "pipecat": "enabled",
        "sarvam_key_length": len(SARVAM_API_KEY),
        "groq_key_set": len(GROQ_API_KEY) > 10,
        "plivo_auth_id": PLIVO_AUTH_ID[:8] + "...",
        "plivo_token_set": len(PLIVO_AUTH_TOKEN) > 10,
        "llm": "groq/openai/gpt-oss-20b",
        "stt": "sarvam/saaras:v3",
        "tts": "sarvam/bulbul:v3-beta",
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
        ),
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

    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        settings=GroqLLMService.Settings(model="openai/gpt-oss-20b"),
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
            f"Maine dekha aapne {lead['interest']} ke baare mein enquiry ki thi — "
            f"main Akriti Patel bol rahi hoon Parth Kalyani se, Vadodara se. "
            f"Kya aap 2 minute baat kar sakte hain?"
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
        await post_call_results(lead, transcript, final_status)

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
