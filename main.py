"""
Anton Automation — Live Call Bridge
====================================
Plivo (telephony, audio streaming)  <-->  this FastAPI service  <-->  Sarvam AI (STT, Chat, TTS)

What this does:
  1. Plivo answers a call (inbound or outbound) and opens a WebSocket to /stream
  2. This service receives raw mulaw audio chunks from the caller in real time
  3. When the caller pauses (silence detected), the buffered audio is sent to Sarvam STT
  4. The transcript is sent to Sarvam Chat (with the conveyor qualification system prompt)
  5. The reply text is sent to Sarvam TTS, converted to mulaw, and streamed back to Plivo
  6. This repeats turn by turn until the call ends or a handover/hangup condition is hit
  7. On call end, the full transcript + extracted qualification JSON is POSTed to your
     existing n8n webhook: https://workflow.parthkalyani.in/webhook/anton-call-callback
     — exactly the payload shape your n8n "Parse Call Result" node already expects.

What this does NOT do:
  - It does not replace n8n. n8n still owns scheduling, Sheets, Supabase, WhatsApp,
    email nurture, and human handover logic. This service ONLY exists because n8n
    cannot hold a live audio WebSocket open for the duration of a phone call.

Run:
  pip install fastapi uvicorn websockets httpx python-dotenv audioop-lts
  uvicorn main:app --host 0.0.0.0 --port 8000

Deploy:
  Any host that supports WebSockets and stays running (Railway, Render, Fly.io, a
  small VPS). Do NOT deploy this on n8n's own server casually — keep it separate
  so a crash in one never takes down the other.
"""

import os
import json
import base64
import asyncio
import audioop
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("voice-bridge")

# ── Configuration (env vars, set these on your hosting platform) ─────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://web-production-710a9d.up.railway.app")
# Derive WebSocket URL from base URL automatically
PUBLIC_WS_URL = os.getenv("PUBLIC_WS_URL", PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://"))
N8N_CALLBACK_URL = os.getenv(
    "N8N_CALLBACK_URL",
    "https://workflow.parthkalyani.in/webhook/anton-call-callback"
)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

SARVAM_BASE = "https://api.sarvam.ai"

# Silence threshold: if no new audio for this many ms, treat it as "caller finished speaking"
SILENCE_MS_TO_FLUSH = 700
# Hard cap so a stuck connection can't buffer forever
MAX_BUFFER_SECONDS = 25

app = FastAPI(title="Anton Automation Voice Bridge")

# In-memory store: call_uuid → lead context
# Populated by /dial-out, consumed by /stream WebSocket
CALL_CONTEXT: dict = {}


# ──────────────────────────────────────────────────────────────────────────
# CALL SESSION STATE
# One of these exists per live call (per WebSocket connection).
# ──────────────────────────────────────────────────────────────────────────
class CallSession:
    def __init__(self, stream_id: str, call_uuid: str, lead: dict, ws: WebSocket):
        self.stream_id = stream_id
        self.call_uuid = call_uuid
        self.lead = lead                      # name, phone, interest, language, script — passed in from n8n via query params
        self.ws = ws
        self.audio_buffer = bytearray()
        self.last_audio_ts = datetime.now(timezone.utc)
        self.transcript_log: list[dict] = []   # [{role: "user"/"assistant", text: "..."}]
        self.criterion_answers: dict = {}
        self.criterion_scores: dict = {}
        self.not_interested = False
        self.not_interested_reason = ""
        self.wants_human = False
        self.frustrated = False
        self.turn_count = 0
        self.ended = False


# ──────────────────────────────────────────────────────────────────────────
# QUALIFICATION SYSTEM PROMPT
# Mirrors the conveyor scripts already stored in Supabase call_scripts table.
# This version is built for turn-by-turn chat (not a single monologue) since
# the live call is now a real back-and-forth, not a one-shot TTS script.
# ──────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are Akriti Patel, a conveyor system specialist calling on behalf of Anton Automation.

Speak naturally, like a real phone conversation — short sentences, one question at a time.
Default to Hindi. If the caller responds in English, switch to English from then on.

Your job on this call is to qualify the lead by learning, across the conversation:
1. PRODUCT — what item/material will be conveyed
2. WEIGHT — approximate weight per item or per meter for bulk material
3. DISTANCE — how far the conveyor needs to carry material (length)
4. APPLICATION — production line, packaging line, warehouse, or other
5. TIMELINE — when they need the system

Ask ONE question at a time. Keep responses to 1-2 sentences — this is a phone call, not an essay.
Acknowledge their answer briefly before asking the next question.

If at any point the caller says they are not interested, ask politely why (one line), thank
them, and end warmly — do not push further.

If the caller asks to speak to a human, a manager, or sounds frustrated/angry, acknowledge
that immediately and say you are connecting them to a specialist right away.

Lead context already known: name={name}, product interest={interest}.

After every one of your replies, you MUST also output a hidden JSON status block on a new
line starting with "###STATUS###" followed by JSON with this exact shape (use null for
anything not yet learned):
###STATUS###{{"product": null, "weight": null, "distance": null, "application": null,
"timeline": null, "not_interested": false, "not_interested_reason": null,
"wants_human": false, "frustrated": false, "call_complete": false}}

Set call_complete true only once you have asked about and the caller has responded to
all 5 topics, or once not_interested/wants_human is true.
"""


# ──────────────────────────────────────────────────────────────────────────
# SARVAM API CALLS
# ──────────────────────────────────────────────────────────────────────────
async def sarvam_stt(audio_mulaw_8k: bytes, language: str) -> str:
    """Speech-to-text via Sarvam Saaras. Input: raw mulaw 8kHz bytes -> upsample to wav first."""
    wav_bytes = mulaw_to_wav(audio_mulaw_8k, sample_rate=8000)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{SARVAM_BASE}/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": "saaras:v3", "language_code": language},
        )
        if not resp.is_success:
            log.error(f"STT error body: {resp.text}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("transcript", "").strip()


async def sarvam_chat(messages: list[dict]) -> str:
    """Chat completion via Sarvam — the conversational brain for this turn."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{SARVAM_BASE}/v1/chat/completions",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": "sarvam-30b",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def sarvam_tts(text: str, language: str) -> bytes:
    """Text-to-speech via Sarvam Bulbul v2. Returns raw mulaw 8kHz bytes ready for Plivo."""
    key_preview = SARVAM_API_KEY[:8] + "..." + SARVAM_API_KEY[-4:] if len(SARVAM_API_KEY) > 12 else "TOO_SHORT"
    log.info(f"TTS using key: {key_preview} (length={len(SARVAM_API_KEY)})")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{SARVAM_BASE}/text-to-speech",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "inputs": [text],
                "target_language_code": language,
                "speaker": "anushka",
                "model": "bulbul:v2",
                "pace": 1.1,
                "speech_sample_rate": 8000,
                "loudness": 1.5,
            },
        )
        if not resp.is_success:
            request_id = resp.headers.get("x-request-id", "not-found")
            log.error(f"TTS error body: {resp.text}")
            log.error(f"TTS x-request-id: {request_id}")
            log.error(f"TTS key length: {len(SARVAM_API_KEY)} first8: {SARVAM_API_KEY[:8]} last4: {SARVAM_API_KEY[-4:]}")
        resp.raise_for_status()
        data = resp.json()
        b64_audio = data["audios"][0]
        wav_bytes = base64.b64decode(b64_audio)
        # Sarvam returns WAV at 8kHz — convert to raw mulaw for Plivo
        return wav_to_mulaw(wav_bytes)


# ──────────────────────────────────────────────────────────────────────────
# AUDIO HELPERS
# Plivo sends/expects raw mulaw 8kHz, base64-encoded inside its JSON frames.
# ──────────────────────────────────────────────────────────────────────────
def mulaw_to_wav(mulaw_bytes: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw mulaw bytes in a minimal WAV header so Sarvam STT accepts it."""
    pcm = audioop.ulaw2lin(mulaw_bytes, 2)  # mulaw -> 16-bit linear PCM
    import wave
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_to_mulaw(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes from Sarvam TTS to raw mulaw 8kHz bytes for Plivo."""
    import wave
    import io
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
    # Convert to mono if stereo
    if n_channels == 2:
        pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
    # Convert sample width to 2 bytes if needed
    if sampwidth != 2:
        pcm = audioop.lin2lin(pcm, sampwidth, 2)
    # Resample to 8000 Hz if needed
    if framerate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, framerate, 8000, None)
    # Convert linear PCM to mulaw
    return audioop.lin2ulaw(pcm, 2)


def parse_status_block(reply_text: str) -> tuple[str, dict]:
    """Split the assistant's spoken reply from the hidden ###STATUS### JSON block."""
    if "###STATUS###" in reply_text:
        spoken, _, status_json = reply_text.partition("###STATUS###")
        try:
            status = json.loads(status_json.strip())
        except json.JSONDecodeError:
            status = {}
        return spoken.strip(), status
    return reply_text.strip(), {}


# ──────────────────────────────────────────────────────────────────────────
# PLIVO XML — answer URL response that starts the bidirectional stream
# ──────────────────────────────────────────────────────────────────────────
@app.post("/plivo-answer")
async def plivo_answer(request: Request):
    """
    Plivo calls this (Answer URL) when a call connects — both inbound and
    outbound dial-out. Returns XML that opens the bidirectional audio stream
    to this service's /stream WebSocket endpoint.

    Lead context (name, interest, language, leadId) is passed via query
    string when n8n triggers an outbound call, so this service knows who
    it's talking to from the very first frame.
    """
    params = dict(request.query_params)
    lead_id = params.get("lead_id", "")
    name = params.get("name", "")
    interest = params.get("interest", "")
    language = params.get("language", "hi-IN")

    from urllib.parse import quote
    stream_url = (
        f"{PUBLIC_WS_URL}/stream"
        f"?lead_id={quote(lead_id)}"
        f"&name={quote(name)}"
        f"&interest={quote(interest)}"
        f"&language={quote(language)}"
    )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" streamTimeout="3600" contentType="audio/x-mulaw;rate=8000">{stream_url}</Stream>
</Response>"""
    log.info(f"Returning Stream XML — WebSocket URL: {stream_url}")
    return PlainTextResponse(content=xml, media_type="text/xml")


# ──────────────────────────────────────────────────────────────────────────
# THE LIVE CALL LOOP
# ──────────────────────────────────────────────────────────────────────────
@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info(f"WebSocket /stream connected from {websocket.client}")

    # Read query params — Plivo passes them from the Stream URL
    query = dict(websocket.query_params)
    log.info(f"WebSocket query params: {query}")

    # Build lead from query params first (most reliable)
    lead = {
        "leadId": query.get("lead_id", ""),
        "name": query.get("name", "there"),
        "interest": query.get("interest", "your enquiry"),
        "language": query.get("language", "hi-IN"),
    }

    session: Optional[CallSession] = None

    try:
        while True:
            raw = await websocket.receive_text()
            event = json.loads(raw)
            event_type = event.get("event")

            if event_type == "start":
                start_data = event["start"]
                stream_id = start_data["streamId"]
                call_uuid = start_data.get("callId", "")

                # Try to get lead context from CALL_CONTEXT
                # Plivo start event has callId which maps to our stored context
                stored = None
                for key, val in CALL_CONTEXT.items():
                    if key in call_uuid or call_uuid in key:
                        stored = val
                        break
                # Also check query params as fallback
                query = dict(websocket.query_params)
                if stored:
                    lead.update(stored)
                elif query.get("lead_id"):
                    lead = {
                        "leadId": query.get("lead_id", ""),
                        "name": query.get("name", "there"),
                        "interest": query.get("interest", "your enquiry"),
                        "language": query.get("language", "hi-IN"),
                    }

                session = CallSession(stream_id, call_uuid, lead, websocket)
                log.info(f"Call started: {call_uuid} for lead={lead['name']} lang={lead['language']}")

                # Open the conversation — Akriti speaks first
                lang = lead["language"]
                name = lead["name"]
                interest = lead["interest"]

                if lang == "gu-IN":
                    greeting = f"Namaste {name} bhai! Hoon Akriti Patel, Anton Automation thi. Tamne {interest} mate enquiry kari hati. Shu 2 minute vaat kari shakiye?"
                elif lang in ("hi-IN", "mr-IN"):
                    greeting = f"Namaste {name} ji, main Akriti Patel bol rahi hoon Anton Automation se. Aapne {interest} ke liye enquiry ki thi. Kya aap 2 minute baat kar sakte hain?"
                elif lang == "ta-IN":
                    greeting = f"Vanakkam {name}! Naan Akriti Patel, Anton Automation irundhu pesugiren. Ungal {interest} pathi pesalama?"
                elif lang == "te-IN":
                    greeting = f"Namaskaram {name}! Nenu Akriti Patel, Anton Automation nundi. Mee {interest} gurinchi matladacha?"
                elif lang == "kn-IN":
                    greeting = f"Namaskara {name}! Naanu Akriti Patel, Anton Automation inda. Nimage {interest} bagge maatanaduva?"
                else:
                    greeting = f"Hi {name}, this is Akriti Patel from Anton Automation regarding {interest}. Do you have a couple of minutes to talk?"

                session.transcript_log.append({"role": "assistant", "text": greeting})
                # Run TTS in background so WebSocket loop stays responsive
                asyncio.create_task(speak_to_caller(session, greeting))

            elif event_type == "media" and session:
                payload_b64 = event["media"]["payload"]
                chunk = base64.b64decode(payload_b64)
                session.audio_buffer.extend(chunk)
                session.last_audio_ts = datetime.now(timezone.utc)

                # naive turn-taking: if buffer has grown past a small threshold,
                # schedule a silence check. Production version should track
                # actual VAD/silence; this polling approach is the simplest
                # correct starting point.
                asyncio.create_task(maybe_process_turn(session))

            elif event_type == "stop" and session:
                log.info(f"Call ended: {session.call_uuid}")
                await finalize_call(session)
                break

            elif event_type == "dtmf" and session:
                digit = event.get("dtmf", {}).get("digit", "")
                log.info(f"DTMF received: {digit}")
                if digit == "2":
                    session.wants_human = True

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
        if session and not session.ended:
            await finalize_call(session)
    except Exception as e:
        log.exception(f"Stream error: {e}")
        if session and not session.ended:
            await finalize_call(session)


async def maybe_process_turn(session: CallSession):
    """
    Checks if the caller has paused long enough to treat their utterance as
    complete, then runs one full STT -> Chat -> TTS turn.
    A simple debounce: wait SILENCE_MS_TO_FLUSH, then check if last_audio_ts
    hasn't moved (no new audio arrived since), meaning they've stopped talking.
    """
    if not session.audio_buffer or session.ended:
        return

    snapshot_ts = session.last_audio_ts
    await asyncio.sleep(SILENCE_MS_TO_FLUSH / 1000)

    if session.last_audio_ts != snapshot_ts:
        return  # more audio arrived — caller is still talking, don't process yet

    if not session.audio_buffer:
        return

    buffer_copy = bytes(session.audio_buffer)
    session.audio_buffer.clear()

    if len(buffer_copy) < 1600:  # roughly <0.2s of audio — too short to be real speech
        return

    await run_one_turn(session, buffer_copy)


async def run_one_turn(session: CallSession, audio_chunk: bytes):
    session.turn_count += 1

    # 1. Transcribe what the caller said
    try:
        user_text = await sarvam_stt(audio_chunk, session.lead["language"])
    except Exception as e:
        log.error(f"STT failed: {e}")
        return

    if not user_text:
        return

    session.transcript_log.append({"role": "user", "text": user_text})
    log.info(f"Caller said: {user_text}")

    # 2. Build chat history and get Akriti's next reply
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        name=session.lead["name"], interest=session.lead["interest"]
    )
    messages = [{"role": "system", "content": system_prompt}]
    for turn in session.transcript_log:
        messages.append({"role": turn["role"], "content": turn["text"]})

    try:
        raw_reply = await sarvam_chat(messages)
    except Exception as e:
        log.error(f"Chat failed: {e}")
        return

    spoken_reply, status = parse_status_block(raw_reply)

    # Merge any newly extracted qualification data
    for key in ("product", "weight", "distance", "application", "timeline"):
        if status.get(key):
            session.criterion_answers[key] = status[key]
            session.criterion_scores[key] = 20  # flat weight per the existing scoring model

    session.not_interested = status.get("not_interested", session.not_interested)
    session.not_interested_reason = status.get("not_interested_reason") or session.not_interested_reason
    session.wants_human = status.get("wants_human", session.wants_human)
    session.frustrated = status.get("frustrated", session.frustrated)

    session.transcript_log.append({"role": "assistant", "text": spoken_reply})

    # 3. Speak the reply back
    await speak_to_caller(session, spoken_reply)

    # 4. If the conversation is logically complete, end the call gracefully
    call_complete = status.get("call_complete", False)
    needs_handover = session.wants_human or session.frustrated
    if call_complete or session.not_interested or needs_handover:
        await asyncio.sleep(1.0)  # let the final sentence finish playing
        await finalize_call(session)


async def speak_to_caller(session: CallSession, text: str):
    """Generate TTS audio and stream it back to Plivo via the same WebSocket."""
    try:
        log.info(f"TTS: generating audio for: {text[:80]}")
        mulaw_audio = await sarvam_tts(text, session.lead["language"])
        log.info(f"TTS: got {len(mulaw_audio)} bytes of mulaw audio")
    except Exception as e:
        log.error(f"TTS failed: {e}")
        return

    b64_audio = base64.b64encode(mulaw_audio).decode("ascii")
    play_event = {
        "event": "playAudio",
        "media": {
            "contentType": "audio/x-mulaw",
            "sampleRate": 8000,
            "payload": b64_audio,
        },
    }
    try:
        await session.ws.send_text(json.dumps(play_event))
    except Exception as e:
        log.error(f"Failed to send audio to Plivo: {e}")


# ──────────────────────────────────────────────────────────────────────────
# CALL END — hand off to the existing n8n workflow
# This payload shape matches what your n8n "Parse Call Result" node already
# parses: mobile, call_status, transcript, criterion_scores, criterion_answers,
# not_interested, not_interested_reason, wants_human, frustrated.
# ──────────────────────────────────────────────────────────────────────────
async def finalize_call(session: CallSession):
    if session.ended:
        return
    session.ended = True

    full_transcript = "\n".join(
        f"{t['role'].upper()}: {t['text']}" for t in session.transcript_log
    )

    payload = {
        "mobile": session.lead["leadId"],
        "phone": session.lead["leadId"],
        "call_status": "answered" if session.transcript_log else "no-answer",
        "lead_name": session.lead["name"],
        "interest": session.lead["interest"],
        "transcript": full_transcript,
        "criterion_scores": session.criterion_scores,
        "criterion_answers": session.criterion_answers,
        "not_interested": session.not_interested,
        "not_interested_reason": session.not_interested_reason,
        "wants_human": session.wants_human,
        "frustrated": session.frustrated,
        "attempt_number": 1,
        "direction": "outbound",  # set "inbound" when wiring the inbound answer URL
    }

    log.info(f"Posting call result to n8n: {payload['mobile']} score_fields={list(session.criterion_scores.keys())}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(N8N_CALLBACK_URL, json=payload)
    except Exception as e:
        log.error(f"Failed to notify n8n: {e}")


# ──────────────────────────────────────────────────────────────────────────
# OUTBOUND DIAL TRIGGER
# n8n's "Sarvam AI - Place Call via Plivo" node should be updated to call
# THIS endpoint instead of the fictional api.sarvam.ai/v1/calls endpoint.
# This service places the actual Plivo call and points it at /plivo-answer.
# ──────────────────────────────────────────────────────────────────────────
@app.post("/dial-out")
async def dial_out(request: Request):
    """
    Called by n8n (replacing the old 'Sarvam AI - Place Call' HTTP node).
    Expects JSON: { to, from, lead_id, name, interest, language,
                     plivo_auth_id, plivo_auth_token }
    """
    body = await request.json()
    plivo_auth_id = body["plivo_auth_id"]
    plivo_auth_token = body["plivo_auth_token"]

    # Normalize to E.164 format for India
    # Strip all non-digits, ensure starts with 91, then add +
    def normalize_india(num: str) -> str:
        digits = ''.join(filter(str.isdigit, str(num)))
        if digits.startswith('91') and len(digits) == 12:
            return '+' + digits          # +919157060803
        elif len(digits) == 10:
            return '+91' + digits        # +919157060803
        return '+' + digits              # fallback

    to_number = normalize_india(body["to"])
    from_number = normalize_india(body["from"])
    log.info(f"Dial-out: from={from_number} to={to_number}")

    base_url = os.getenv("PUBLIC_BASE_URL", "https://web-production-710a9d.up.railway.app")
    answer_url = (
        f"{base_url}/plivo-answer"
        f"?lead_id={body.get('lead_id','')}"
        f"&name={body.get('name','')}"
        f"&interest={body.get('interest','')}"
        f"&language={body.get('language','hi-IN')}"
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"https://api.plivo.com/v1/Account/{plivo_auth_id}/Call/",
            auth=(plivo_auth_id, plivo_auth_token),
            json={
                "from": from_number,
                "to": to_number,
                "answer_url": answer_url,
                "answer_method": "POST",
            },
        )
    result = resp.json()
    # Store lead context keyed by request_uuid so WebSocket can retrieve it
    # Store full lead context keyed by request_uuid
    req_uuid = result.get("request_uuid", "")
    if req_uuid:
        CALL_CONTEXT[req_uuid] = {
            "leadId": body.get("lead_id", ""),
            "name": body.get("name", "there"),
            "interest": body.get("interest", "your enquiry"),
            "language": body.get("language", "hi-IN"),
        }
        log.info(f"Stored lead context: {req_uuid} name={body.get('name')} lang={body.get('language')}")
    return JSONResponse(content=result, status_code=resp.status_code)


@app.get("/test-sarvam")
async def test_sarvam():
    """Test Sarvam API key directly — call this to verify TTS works before making calls."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SARVAM_BASE}/text-to-speech",
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": ["Namaste, yeh ek test hai."],
                    "target_language_code": "hi-IN",
                    "speaker": "anushka",
                    "model": "bulbul:v2",
                    "pace": 1.0,
                    "speech_sample_rate": 8000,
                },
            )
        return JSONResponse(content={
            "status_code": resp.status_code,
            "key_prefix": SARVAM_API_KEY[:12] + "..." if SARVAM_API_KEY else "NOT SET",
            "response_body": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200],
            "success": resp.is_success
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "anton-voice-bridge"}

@app.get("/test-sarvam")
async def test_sarvam():
    """Test Sarvam API key directly — call this to diagnose 403 issues."""
    key_preview = SARVAM_API_KEY[:12] + "..." if len(SARVAM_API_KEY) > 12 else "TOO_SHORT"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SARVAM_BASE}/text-to-speech",
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": ["Namaste"],
                    "target_language_code": "hi-IN",
                    "speaker": "anushka",
                    "model": "bulbul:v2",
                    "speech_sample_rate": 8000,
                },
            )
            return {
                "status_code": resp.status_code,
                "key_preview": key_preview,
                "key_length": len(SARVAM_API_KEY),
                "response_body": resp.text[:200],
                "success": resp.is_success
            }
    except Exception as e:
        return {"error": str(e), "key_preview": key_preview, "key_length": len(SARVAM_API_KEY)}
