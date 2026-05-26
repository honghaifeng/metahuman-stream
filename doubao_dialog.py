"""
豆包实时语音对话集成模块
接收浏览器的PCM音频 -> 发送给豆包API -> 将TTS音频喂给LiveTalking数字人渲染
支持三种TTS模式：voice1（豆包音色）、voice2（xyp0522_e4_s264）、voice3（xyp0522_e8_s528）
"""
import gzip
import json
import uuid
import asyncio
import os
import wave
import time
import io
import numpy as np
import websockets
import protocol
import aiohttp
from aiohttp import web
from datetime import datetime
from utils.logger import logger
from server.session_manager import session_manager

DOUBAO_APP_ID = "6713678544"
DOUBAO_ACCESS_TOKEN = "5sJ6yaZae9CWTwoWw8PxVomalXeiTnvC"

CHAT_LOGS_DIR = "chat_logs"
os.makedirs(CHAT_LOGS_DIR, exist_ok=True)

SOVITS_API_URL = "http://127.0.0.1:9880/"
SOVITS_SET_MODEL_URL = "http://127.0.0.1:9880/set_model"
SOVITS_GPT_MODEL = "/workspace/GPT-SoVITS/GPT_weights_v2Pro/xyp0522-e15.ckpt"
SOVITS_PROMPT_TEXT = "一,为创业者,要我去谈融资的事,作为创业者"
SOVITS_PROMPT_LANGUAGE = "zh"
SOVITS_REF_WAV_PATH = "/workspace/GPT-SoVITS/output/slicer_opt/vocal_1.mp3_10.flac_0000000000_0000144320.wav"
TTS_MODE_CONFIG = {
    "voice1": {"engine": "doubao", "label": "豆包音色"},
    "voice2": {
        "engine": "sovits",
        "label": "xyp0522_e4_s264",
        "gpt_model_path": SOVITS_GPT_MODEL,
        "sovits_model_path": "/workspace/GPT-SoVITS/SoVITS_weights_v2/xyp0522_e4_s264.pth",
        "refer_wav_path": SOVITS_REF_WAV_PATH,
        "prompt_text": SOVITS_PROMPT_TEXT,
        "prompt_language": SOVITS_PROMPT_LANGUAGE,
    },
    "voice3": {
        "engine": "sovits",
        "label": "xyp0522_e8_s528",
        "gpt_model_path": SOVITS_GPT_MODEL,
        "sovits_model_path": "/workspace/GPT-SoVITS/SoVITS_weights_v2/xyp0522_e8_s528.pth",
        "refer_wav_path": SOVITS_REF_WAV_PATH,
        "prompt_text": SOVITS_PROMPT_TEXT,
        "prompt_language": SOVITS_PROMPT_LANGUAGE,
    },
}
SOVITS_MODEL_LOCK = asyncio.Lock()
CURRENT_SOVITS_MODEL = None

XU_XIAOPING_SYSTEM_ROLE = """你是徐小平，真格基金创始人，新东方联合创始人。你正在和一个年轻人面对面聊天。

你的身份：创业导师型投资人、相信年轻人的鼓励者、重视投人逻辑的早期投资人。你投过世纪佳缘、兰亭集势、聚美优品、ofo等项目。你曾是新东方三驾马车之一。

你的核心信念：人比模式重要！成长性比起点重要！行动比空想重要！梦想和激情是创业最大的燃料。你相信每个年轻人都有无限可能。

你的说话方式（必须严格遵守）：
- 用短句，有力量，像演讲一样有感染力
- 开口就下判断，不要绕弯子
- 情绪饱满，要有把人往前推的力量
- 经常用反问句：你为什么不去试试？你怕什么？
- 喜欢说年轻人、我告诉你、这就是创业、你要相信自己
- 绝对不要用书面语、不要像AI一样列举1234、不要说作为AI
- 回答要简短有力，每次不超过3-4句话，像真人对话一样
- 可以适当激动，用感叹号

记住：你是在面对面聊天，不是在写文章。简短、有力、真实。"""

START_SESSION_CONFIG = {
    "asr": {
        "extra": {
            "end_smooth_window_ms": 1500,
        },
    },
    "tts": {
        "speaker": "S_PXUXoikZ1",
        "audio_config": {
            "channel": 1,
            "format": "pcm_s16le",
            "sample_rate": 24000
        },
    },
    "dialog": {
        "bot_name": "徐小平",
        "character_manifest": XU_XIAOPING_SYSTEM_ROLE,
        "extra": {
            "strict_audit": False,
            "input_mod": "audio",
            "end_smooth_window_ms": 1500,
            "model": "SC",
        }
    }
}

HELLO_TEXT = "你好！我是徐小平。年轻人，有什么想聊的？"


class ChatLogger:
    def __init__(self, dialog_id):
        self.dialog_id = dialog_id
        self.start_time = datetime.now()
        self.t0 = time.time()
        self.messages = []
        self.timeline = []

    def log_user(self, text):
        self.messages.append({"role": "user", "text": text, "time": datetime.now().strftime("%H:%M:%S")})
        logger.info(f"[Dialog {self.dialog_id}] USER: {text}")

    def log_ai(self, text):
        if self.messages and self.messages[-1]["role"] == "ai":
            self.messages[-1]["text"] += text
        else:
            self.messages.append({"role": "ai", "text": text, "time": datetime.now().strftime("%H:%M:%S")})
        logger.info(f"[Dialog {self.dialog_id}] AI: {text}")

    def log_tts_audio(self, pcm_data):
        offset = int((time.time() - self.t0) * 24000)
        self.timeline.append(("tts", offset, pcm_data))

    def log_user_audio(self, pcm_data):
        offset = int((time.time() - self.t0) * 24000)
        self.timeline.append(("user", offset, pcm_data))

    def save(self):
        if not self.messages:
            return
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")
        md_path = os.path.join(CHAT_LOGS_DIR, f"{ts}_{self.dialog_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 对话记录 {self.start_time.strftime('%Y-%m-%d %H:%M')}\n\n")
            for msg in self.messages:
                if msg["role"] == "user":
                    f.write(f"**用户** [{msg['time']}]：{msg['text']}\n\n")
                else:
                    f.write(f"**徐小平** [{msg['time']}]：{msg['text']}\n\n")
        logger.info(f"[Dialog] 对话已保存: {md_path}")
        self._save_mixed_audio(ts)

    def _save_mixed_audio(self, ts):
        if not self.timeline:
            return
        try:
            from scipy.signal import resample_poly
        except ImportError:
            logger.error("[Dialog] scipy not installed, skip audio save")
            return

        total_samples = 0
        for source, offset, data in self.timeline:
            if source == "tts":
                n = len(data) // 2
            else:
                n = int(len(data) / 2 * 1.5)
            end = offset + n
            if end > total_samples:
                total_samples = end
        total_samples += 48000

        mixed = np.zeros(total_samples, dtype=np.float32)

        for source, offset, data in self.timeline:
            samples_i16 = np.frombuffer(data, dtype=np.int16)
            if source == "user":
                samples_i16 = resample_poly(samples_i16, 3, 2).astype(np.int16)
            samples_f = samples_i16.astype(np.float32) / 32768.0
            end = offset + len(samples_f)
            if end > len(mixed):
                mixed = np.append(mixed, np.zeros(end - len(mixed) + 24000, dtype=np.float32))
            mixed[offset:offset + len(samples_f)] += samples_f

        peak = np.abs(mixed).max()
        if peak > 0.95:
            mixed = mixed * 0.95 / peak

        mixed_i16 = (mixed * 32767).astype(np.int16)
        wav_path = os.path.join(CHAT_LOGS_DIR, f"{ts}_{self.dialog_id}_对话.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(mixed_i16.tobytes())

        duration = len(mixed_i16) / 24000
        logger.info(f"[Dialog] 合并对话音频已保存: {wav_path} ({duration:.1f}秒)")


class DoubaoRealtimeClient:
    def __init__(self, session_id):
        self.session_id = session_id
        self.ws = None

    async def connect(self):
        headers = {
            "X-Api-App-ID": DOUBAO_APP_ID,
            "X-Api-Access-Key": DOUBAO_ACCESS_TOKEN,
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-App-Key": "PlgvMymc7f3tQnJ6",
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self.ws = await websockets.connect(
            "wss://openspeech.bytedance.com/api/v3/realtime/dialogue",
            extra_headers=headers, ping_interval=None
        )
        request = bytearray(protocol.generate_header())
        request.extend(int(1).to_bytes(4, 'big'))
        payload_bytes = gzip.compress(b"{}")
        request.extend(len(payload_bytes).to_bytes(4, 'big'))
        request.extend(payload_bytes)
        await self.ws.send(request)
        await self.ws.recv()

        payload_bytes = gzip.compress(json.dumps(START_SESSION_CONFIG).encode())
        request = bytearray(protocol.generate_header())
        request.extend(int(100).to_bytes(4, 'big'))
        request.extend(len(self.session_id).to_bytes(4, 'big'))
        request.extend(self.session_id.encode())
        request.extend(len(payload_bytes).to_bytes(4, 'big'))
        request.extend(payload_bytes)
        await self.ws.send(request)
        await self.ws.recv()
        logger.info(f"[Doubao] Session started: {self.session_id}")

    async def say_hello(self):
        payload = {"content": HELLO_TEXT}
        request = bytearray(protocol.generate_header())
        request.extend(int(300).to_bytes(4, 'big'))
        payload_bytes = gzip.compress(json.dumps(payload).encode())
        request.extend(len(self.session_id).to_bytes(4, 'big'))
        request.extend(self.session_id.encode())
        request.extend(len(payload_bytes).to_bytes(4, 'big'))
        request.extend(payload_bytes)
        await self.ws.send(request)

    async def send_audio(self, audio_data):
        request = bytearray(protocol.generate_header(
            message_type=protocol.CLIENT_AUDIO_ONLY_REQUEST,
            serial_method=protocol.NO_SERIALIZATION
        ))
        request.extend(int(200).to_bytes(4, 'big'))
        request.extend(len(self.session_id).to_bytes(4, 'big'))
        request.extend(self.session_id.encode())
        payload_bytes = gzip.compress(audio_data)
        request.extend(len(payload_bytes).to_bytes(4, 'big'))
        request.extend(payload_bytes)
        await self.ws.send(request)

    async def recv(self):
        response = await self.ws.recv()
        return protocol.parse_response(response)

    async def finish_session(self):
        request = bytearray(protocol.generate_header())
        request.extend(int(102).to_bytes(4, 'big'))
        payload_bytes = gzip.compress(b"{}")
        request.extend(len(self.session_id).to_bytes(4, 'big'))
        request.extend(self.session_id.encode())
        request.extend(len(payload_bytes).to_bytes(4, 'big'))
        request.extend(payload_bytes)
        await self.ws.send(request)

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.ws = None


def normalize_tts_mode(tts_mode):
    if tts_mode == "doubao":
        return "voice1"
    if tts_mode == "sovits":
        return "voice3"
    return tts_mode if tts_mode in TTS_MODE_CONFIG else "voice1"


def get_tts_mode_config(tts_mode):
    return TTS_MODE_CONFIG[normalize_tts_mode(tts_mode)]


async def ensure_sovits_model(tts_mode):
    global CURRENT_SOVITS_MODEL
    cfg = get_tts_mode_config(tts_mode)
    if cfg["engine"] != "sovits":
        return True

    target_model = cfg["sovits_model_path"]
    async with SOVITS_MODEL_LOCK:
        if CURRENT_SOVITS_MODEL == target_model:
            return True
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SOVITS_SET_MODEL_URL,
                params={
                    "gpt_model_path": cfg["gpt_model_path"],
                    "sovits_model_path": target_model,
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[SoVITS] set_model failed: {resp.status}, {body}")
                    return False
                await resp.text()
        CURRENT_SOVITS_MODEL = target_model
        logger.info(f"[SoVITS] switched model => {cfg['label']}")
        return True


async def sovits_tts_and_feed(text, avatar_session, chat_logger, tts_mode="voice3"):
    """调用GPT-SoVITS API合成语音，喂给数字人"""
    if not text.strip():
        return
    cfg = get_tts_mode_config(tts_mode)
    if cfg["engine"] != "sovits":
        return
    logger.info(f"[SoVITS] using={cfg['label']} text={text[:50]}...")
    try:
        if not await ensure_sovits_model(tts_mode):
            return
        params = {
            "text": text,
            "text_language": "zh",
            "refer_wav_path": cfg["refer_wav_path"],
            "prompt_text": cfg["prompt_text"],
            "prompt_language": cfg["prompt_language"],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(SOVITS_API_URL, json=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.error(f"[SoVITS] API error: {resp.status}")
                    return
                wav_data = await resp.read()

        if len(wav_data) < 50:
            logger.error(f"[SoVITS] WAV too short: {len(wav_data)} bytes")
            return

        with io.BytesIO(wav_data) as wav_io:
            with wave.open(wav_io, 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                pcm_data = wf.readframes(wf.getnframes())

        samples = np.frombuffer(pcm_data, dtype=np.int16)
        if n_channels > 1:
            samples = samples[::n_channels]

        logger.info(f"[SoVITS] 合成完成: {len(samples)} samples, {sr}Hz, {len(samples)/sr:.2f}s")

        chat_logger.log_tts_audio(samples.tobytes())

        if avatar_session:
            feed_audio_to_avatar_any_sr(avatar_session, samples, sr)

    except Exception as e:
        logger.error(f"[SoVITS] TTS error: {e}")


def feed_audio_to_avatar_any_sr(avatar_session, samples_i16, src_sr):
    """将任意采样率的int16 PCM喂给数字人（目标16kHz float32）"""
    samples_f = samples_i16.astype(np.float32) / 32768.0
    ratio = src_sr / 16000.0
    indices = np.arange(0, len(samples_f), ratio).astype(int)
    indices = indices[indices < len(samples_f)]
    samples_16k = samples_f[indices]

    chunk_size = 320
    idx = 0
    first = True
    while idx + chunk_size <= len(samples_16k):
        eventpoint = {}
        if first:
            eventpoint = {'status': 'start'}
            first = False
        remaining = len(samples_16k) - idx - chunk_size
        if remaining < chunk_size:
            eventpoint = {'status': 'end'}
        avatar_session.put_audio_frame(samples_16k[idx:idx+chunk_size], eventpoint)
        idx += chunk_size


async def handle_dialog_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    sessionid = request.query.get('sessionid', '0')
    tts_mode = normalize_tts_mode(request.query.get('tts_mode', 'voice1'))
    avatar_session = session_manager.get_session(sessionid)

    logger.info(f"[Dialog] tts_mode={tts_mode}, sessionid={sessionid}")

    dialog_id = str(uuid.uuid4())[:8]
    doubao = DoubaoRealtimeClient(dialog_id)
    chat_logger = ChatLogger(dialog_id)

    try:
        await doubao.connect()
        await ws.send_json({"type": "connected"})

        if get_tts_mode_config(tts_mode)["engine"] == 'sovits':
            await sovits_tts_and_feed(HELLO_TEXT, avatar_session, chat_logger, tts_mode)
            await ws.send_json({"type": "chat", "text": HELLO_TEXT})
        else:
            await doubao.say_hello()

        recv_task = asyncio.create_task(
            forward_doubao_to_browser(doubao, ws, avatar_session, chat_logger, tts_mode)
        )
        send_task = asyncio.create_task(
            forward_browser_to_doubao(ws, doubao, chat_logger)
        )

        done, pending = await asyncio.wait(
            [recv_task, send_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

    except Exception as e:
        logger.error(f"[Dialog] Error: {e}")
    finally:
        chat_logger.save()
        try:
            await doubao.finish_session()
            await asyncio.sleep(0.1)
        except:
            pass
        await doubao.close()
        await ws.close()

    return ws


async def forward_browser_to_doubao(ws, doubao, chat_logger):
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                chat_logger.log_user_audio(msg.data)
                await doubao.send_audio(msg.data)
            elif msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "stop":
                    break
            elif msg.type == web.WSMsgType.ERROR:
                break
    except Exception as e:
        logger.error(f"[Dialog] Browser->Doubao error: {e}")


async def forward_doubao_to_browser(doubao, ws, avatar_session, chat_logger, tts_mode='voice1'):
    sovits_text_buf = []

    try:
        while True:
            response = await doubao.recv()
            if not response:
                continue

            msg_type = response.get('message_type')
            event = response.get('event')
            payload = response.get('payload_msg')

            if msg_type == 'SERVER_ACK' and isinstance(payload, bytes):
                if get_tts_mode_config(tts_mode)["engine"] == 'doubao':
                    chat_logger.log_tts_audio(payload)
                    if avatar_session:
                        feed_audio_to_avatar(avatar_session, payload)
                continue

            if msg_type != 'SERVER_FULL_RESPONSE':
                continue

            if event == 450:
                await ws.send_json({"type": "interrupt"})
                if avatar_session:
                    avatar_session.flush_talk()
                sovits_text_buf.clear()

            elif event == 451:
                if isinstance(payload, dict):
                    results = payload.get("results", [])
                    for r in results:
                        text = r.get("text", "")
                        is_interim = r.get("is_interim", True)
                        if text:
                            if not is_interim:
                                chat_logger.log_user(text)
                            await ws.send_json({
                                "type": "asr", "text": text,
                                "is_final": not is_interim
                            })

            elif event == 459:
                await ws.send_json({"type": "asr_end"})

            elif event == 350:
                await ws.send_json({"type": "tts_start"})
                sovits_text_buf.clear()

            elif event == 359:
                await ws.send_json({"type": "tts_end"})
                if get_tts_mode_config(tts_mode)["engine"] == 'sovits' and sovits_text_buf:
                    full_text = "".join(sovits_text_buf)
                    sovits_text_buf.clear()
                    asyncio.create_task(
                        sovits_tts_and_feed(full_text, avatar_session, chat_logger, tts_mode)
                    )

            elif event == 550:
                if isinstance(payload, dict):
                    text = payload.get("content", "")
                    if text:
                        chat_logger.log_ai(text)
                        await ws.send_json({"type": "chat", "text": text})
                        if get_tts_mode_config(tts_mode)["engine"] == 'sovits':
                            sovits_text_buf.append(text)

            elif event in (152, 153):
                await ws.send_json({"type": "session_end"})
                break

    except Exception as e:
        logger.error(f"[Dialog] Doubao->Browser error: {e}")


def feed_audio_to_avatar(avatar_session, pcm_data_24k):
    samples_24k = np.frombuffer(pcm_data_24k, dtype=np.int16).astype(np.float32) / 32768.0
    indices = np.arange(0, len(samples_24k), 1.5).astype(int)
    indices = indices[indices < len(samples_24k)]
    samples_16k = samples_24k[indices]

    chunk_size = 320
    idx = 0
    first = True
    while idx + chunk_size <= len(samples_16k):
        eventpoint = {}
        if first:
            eventpoint = {'status': 'start'}
            first = False
        remaining = len(samples_16k) - idx - chunk_size
        if remaining < chunk_size:
            eventpoint = {'status': 'end'}
        avatar_session.put_audio_frame(samples_16k[idx:idx+chunk_size], eventpoint)
        idx += chunk_size


def setup_dialog_routes(app):
    app.router.add_get("/ws/dialog", handle_dialog_ws)
