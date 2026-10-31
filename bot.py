import io
import os
import json
import asyncio
import time
import html
import re
import urllib.request
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from google import genai
from google.genai import types as genai_types
from database import (
    init_db, make_text_hash, get_cached_verdict, save_public_analysis,
    get_user_lang_mode, set_user_lang_mode,
    log_token_usage, get_token_stats,
    get_channel_score, get_top_channels_report,
    save_message_context, save_error_report,
    delete_user_data, invalidate_cache_for_message, check_and_increment_quota,
    track_user_event, get_admin_analytics_summary, export_table_to_csv
)

load_dotenv()

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "viy_bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ViyBot")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing TELEGRAM_TOKEN or GEMINI_API_KEY in .env configuration file.")

MIN_STATUS_DURATION = 3.0

CONFIG_FILE = BASE_DIR / "config.json"
CONFIG = {}
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

BOT_USERNAME = CONFIG.get("bot_username", "@viy_truth_bot")
DAILY_USER_LIMIT = int(CONFIG.get("daily_user_limit", 15))
FOOTER_KEYWORDS = tuple(CONFIG.get("footer_keywords", ["сайт", "реклама", "помощь", "прислать", "подписаться"]))
NON_TEXT_KEYWORDS = tuple(CONFIG.get("non_text_keywords", ["tts", "audio", "image", "imagen", "vision", "embedding", "veo", "aqa", "thinking"]))
KEY_ALIASES = CONFIG.get("key_aliases", {})

LOCALES_FILE = BASE_DIR / "locales.json"
if not LOCALES_FILE.exists():
    raise FileNotFoundError(f"Localization file not found: {LOCALES_FILE}")
with open(LOCALES_FILE, "r", encoding="utf-8") as f:
    UI_LABELS = json.load(f)

VERDICT_FOUND_PREFIX = UI_LABELS["ru"]["found"]
VERDICT_CLEAN = UI_LABELS["ru"]["clean"]

MODELS_FILE = BASE_DIR / "models.json"
DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash"
]
SAFE_MODEL_PATTERN = re.compile(r"^gemini-[0-9]+\.[0-9]+-flash(?:-lite)?$")


def is_safe_model_name(name: str) -> bool:
    """Validates that the string is strictly a safe Gemini Flash model identifier."""
    if not isinstance(name, str):
        return False
    clean = name.strip()
    if any(skip in clean for skip in NON_TEXT_KEYWORDS):
        return False
    return bool(SAFE_MODEL_PATTERN.match(clean))


def load_models_from_file() -> list[str]:
    """Loads the active model cascade strictly in the priority order defined in models.json."""
    models = []
    if MODELS_FILE.exists():
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_list = data.get("active_models", []) if isinstance(data, dict) else []
            for m in raw_list:
                if is_safe_model_name(m) and m.strip() not in models:
                    models.append(m.strip())
        except Exception as e:
            logger.warning(f"Failed to read models.json ({e}). Falling back to default model list.")
    for fallback_m in DEFAULT_MODELS:
        if fallback_m not in models:
            models.append(fallback_m)
    return models


def save_models_to_file(models_list: list[str]):
    """Safely persists the validated model list to models.json."""
    clean_list = [m for m in models_list if is_safe_model_name(m)]
    if not clean_list:
        return
    try:
        payload = {
            "active_models": clean_list,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to update models.json: {e}")


UNIQUE_MODELS = load_models_from_file()

PROMPT_FILE = BASE_DIR / "prompt.txt"
PROMPT_EXAMPLE_FILE = BASE_DIR / "prompt.example.txt"
if PROMPT_FILE.exists():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
elif PROMPT_EXAMPLE_FILE.exists():
    with open(PROMPT_EXAMPLE_FILE, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    raise FileNotFoundError("Neither prompt.txt nor prompt.example.txt was found.")

TECHNIQUES_FILE = BASE_DIR / "techniques.json"
TECHNIQUE_NAMES = {}
if TECHNIQUES_FILE.exists():
    try:
        with open(TECHNIQUES_FILE, "r", encoding="utf-8") as f:
            TECHNIQUE_NAMES = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read techniques.json ({e}).")

client = genai.Client(api_key=GEMINI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


def refresh_available_models():
    """
    Synchronizes models.json with the live Google API catalog on startup while preserving
    our priority order (fast -lite models first) and appending newly released models at the end.
    """
    global UNIQUE_MODELS
    try:
        live_flash_models = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            short_name = name.replace("models/", "").strip()
            if (
                is_safe_model_name(short_name)
                and not short_name.startswith(("gemini-1.", "gemini-2."))
            ):
                live_flash_models.append(short_name)

        if live_flash_models:
            new_cascade = []
            # 1. Preserve existing priority order from models.json
            for preferred in UNIQUE_MODELS:
                if preferred in live_flash_models and preferred not in new_cascade:
                    new_cascade.append(preferred)
            # 2. Append any newly discovered Flash models to the end of the queue
            for candidate in live_flash_models:
                if candidate not in new_cascade:
                    new_cascade.append(candidate)

            if new_cascade:
                UNIQUE_MODELS = new_cascade
                save_models_to_file(UNIQUE_MODELS)
                logger.info(f"[Auto-Models] Verified model priority order ({len(UNIQUE_MODELS)} models): {UNIQUE_MODELS}")
                return
    except Exception as e:
        logger.warning(f"[Auto-Models] Could not query live model catalog ({type(e).__name__}: {e}). Using cached list: {UNIQUE_MODELS}")


def extract_clean_json(raw_text: str) -> list | dict:
    """Safely extracts JSON from the LLM response, stripping optional markdown code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def strip_channel_footer(text: str) -> str:
    """Strips boilerplate channel navigation footers ('Website | Ads | Subscribe') from the end of post text."""
    if not text:
        return ""
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    last_l = lines[-1].strip().lower()
    has_sep = any(s in last_l for s in ("|", "•", "▪"))
    has_kw = any(w in last_l for w in FOOTER_KEYWORDS)
    if (has_sep and has_kw) or any(w in last_l for w in ("надіслати новину", "підписатися", "прислать новость", "реклама на канале")):
        lines.pop()
    return "\n".join(lines).strip()


def extract_message_links(message: types.Message, raw_text: str) -> list[tuple[str, str]]:
    """
    Extracts (anchor_text, url) pairs from Telegram message entities and inline markdown.
    Ignores boilerplate navigation footers in the final line.
    """
    found_links = []
    entities = message.entities or message.caption_entities or []
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""
    last_line_start = raw_text.rfind(last_line) if (raw_text and last_line) else -1

    def is_footer_link(anchor_str: str, offset_pos: int) -> bool:
        in_last_paragraph = (last_line_start != -1 and offset_pos >= last_line_start)
        if not in_last_paragraph:
            return False
        has_menu_separators = any(sep in last_line for sep in ("|", "•", "▪"))
        anchor_lower = anchor_str.lower()
        has_footer_word = any(w in anchor_lower for w in FOOTER_KEYWORDS)
        return bool(has_menu_separators or has_footer_word)

    for ent in entities:
        url = None
        anchor = ""
        if ent.type == "text_link" and ent.url:
            url = ent.url
            anchor = raw_text[ent.offset : ent.offset + ent.length]
        elif ent.type == "url":
            url = raw_text[ent.offset : ent.offset + ent.length]
            anchor = url

        if url and re.match(r"^https?://t\.me/[a-zA-Z0-9_]+/\d+", url):
            if not is_footer_link(anchor, ent.offset):
                found_links.append((anchor.strip(), url.strip()))

    if not found_links and raw_text:
        for match in re.finditer(r"\[([^\]]+)\]\((https?://t\.me/[a-zA-Z0-9_]+/\d+)\)", raw_text):
            anchor, url = match.group(1), match.group(2)
            if not is_footer_link(anchor, match.start()):
                found_links.append((anchor.strip(), url.strip()))
        if not found_links:
            for match in re.finditer(r"(https?://t\.me/[a-zA-Z0-9_]+/\d+)", raw_text):
                url = match.group(1)
                if not is_footer_link(url, match.start()):
                    found_links.append((url, url))

    return found_links[:3]


def check_single_tg_post(url: str) -> str:
    """
    Fetches public Telegram post preview (?embed=1) within a 2.5s timeout,
    extracts text up to 3500 chars, and checks for external source links.
    """
    embed_url = url.split("?")[0] + "?embed=1&mode=tme"
    try:
        req = urllib.request.Request(embed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            html_page = resp.read().decode("utf-8", errors="ignore")

        text_match = re.search(
            r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            html_page,
            re.DOTALL
        )
        full_post_text = ""
        if text_match:
            clean_txt = re.sub(r"<[^>]+>", " ", text_match.group(1))
            clean_txt = html.unescape(re.sub(r"\s+", " ", clean_txt)).strip()
            full_post_text = f" Reference text inside linked post: «{clean_txt[:3500]}»."

        all_hrefs = re.findall(r'href=["\'](https?://[^"\']+)["\']', html_page)
        content_links = [
            h for h in all_hrefs
            if not any(s in h for s in ["telegram.org", "t.me/share", "tg://", "core.telegram"])
        ]
        current_channel = url.split("t.me/")[1].split("/")[0].lower()
        external_sources = [
            h for h in content_links
            if f"t.me/{current_channel}/" not in h.lower()
        ]

        if external_sources:
            return f"CONTAINS external source links.{full_post_text}"
        return f"NO external primary sources (self-referential post within the same channel).{full_post_text}"
    except Exception:
        return "could not verify linked content"


async def scout_telegram_links(message: types.Message, raw_text: str) -> str:
    """Asynchronously inspects t.me links in the post and appends a verification note for the LLM."""
    links = extract_message_links(message, raw_text)
    if not links:
        return ""

    loop = asyncio.get_running_loop()
    notes = []
    for anchor, url in links:
        channel_slug = url.split("t.me/")[1].split("/")[0]
        verdict = await loop.run_in_executor(None, check_single_tg_post, url)
        notes.append(f"• Link on phrase «{anchor}» points to @{channel_slug} ({url}): {verdict}")

    return (
        "\n\n[СИСТЕМНАЯ ПРОВЕРКА ССЫЛОК / SYSTEM LINK VERIFICATION (FOR REFERENCE ONLY)]:\n"
        + "\n".join(notes)
        + "\n(STRICT RULE: NEVER extract `span` quotes from the reference text inside linked posts! "
        "Evaluate only the main post: 1) if a link is completely unrelated to the claim — flag in whataboutism_red_herring; "
        "2) if a link under 'media reported' points to the same channel without primary sources — flag in anonymous_sources; "
        "3) if a link 'as we wrote earlier' points to a post without external sources — flag circular self-citation in doubt)."
    )


def detect_output_lang(text_sample: str) -> str:
    """Detects output language code (ru, uk, pl, en) using character heuristics."""
    sample_lower = text_sample.lower()
    if any(ch in sample_lower for ch in "іїєґ"):
        return "uk"
    if any(ch in sample_lower for ch in "ąęłńóśźż"):
        return "pl"
    if not any("а" <= ch <= "я" or ch == "ё" for ch in sample_lower) and any("a" <= ch <= "z" for ch in sample_lower):
        return "en"
    return "ru"


def resolve_technique_title(tech_raw: str, lang: str) -> str:
    """Resolves localized technique name from techniques.json with fuzzy key matching."""
    clean_key = tech_raw.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    while "__" in clean_key:
        clean_key = clean_key.replace("__", "_")
    clean_key = KEY_ALIASES.get(clean_key, clean_key)

    entry = TECHNIQUE_NAMES.get(clean_key)
    if not entry:
        for k, v in TECHNIQUE_NAMES.items():
            if k in clean_key or clean_key in k:
                entry = v
                break

    if isinstance(entry, dict):
        return entry.get(f"name_{lang}") or entry.get("name_ru") or clean_key
    elif isinstance(entry, str):
        return entry
    return tech_raw.replace("_", " ").capitalize()


def resolve_user_target_lang(user_id: int, tg_lang_code: str | None = None) -> str:
    """Returns target language code ('auto', 'ru', 'uk', 'pl', 'en') for post analysis."""
    mode = get_user_lang_mode(user_id)
    if mode in ("ru", "uk", "pl", "en"):
        return mode
    if mode == "tg_system" and tg_lang_code:
        short_code = tg_lang_code[:2].lower()
        if short_code in ("ru", "uk", "pl", "en"):
            return short_code
        return "en"
    return "auto"


def resolve_ui_lang(user_id: int, tg_lang_code: str | None = None) -> str:
    """Returns interface language code for menu commands (/start, /lang, /scores, /about, /privacy, /feedback)."""
    mode = get_user_lang_mode(user_id)
    if mode in ("ru", "uk", "pl", "en"):
        return mode
    if tg_lang_code:
        short_code = tg_lang_code[:2].lower()
        if short_code in ("ru", "uk", "pl", "en"):
            return short_code
    return "en"


def get_localized_bot_commands(lang: str = "ru") -> list[BotCommand]:
    """Builds localized command list for Telegram's blue Menu button."""
    cmds = UI_LABELS.get(lang, UI_LABELS["ru"])["menu_cmds"]
    return [
        BotCommand(command="lang", description=cmds["lang"]),
        BotCommand(command="scores", description=cmds["scores"]),
        BotCommand(command="about", description=cmds["about"]),
        BotCommand(command="privacy", description=cmds["privacy"]),
        BotCommand(command="feedback", description=cmds["feedback"])
    ]


def build_lang_keyboard(current_mode: str, ui_lang: str = "ru") -> InlineKeyboardMarkup:
    """Builds inline language selection keyboard with a checkmark on the active mode."""
    mode_names = UI_LABELS.get(ui_lang, UI_LABELS["ru"])["lang_modes"]

    def mark(code: str) -> str:
        label = mode_names[code]
        return f"✅ {label}" if current_mode == code else label

    keyboard = [
        [InlineKeyboardButton(text=mark("auto"), callback_data="setlang:auto")],
        [InlineKeyboardButton(text=mark("tg_system"), callback_data="setlang:tg_system")],
        [
            InlineKeyboardButton(text=mark("ru"), callback_data="setlang:ru"),
            InlineKeyboardButton(text=mark("uk"), callback_data="setlang:uk")
        ],
        [
            InlineKeyboardButton(text=mark("pl"), callback_data="setlang:pl"),
            InlineKeyboardButton(text=mark("en"), callback_data="setlang:en")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def format_analysis(
    raw_text: str,
    source_info: str = "",
    forced_lang: str = "auto",
    original_text: str = "",
    fallback_ui_lang: str = "ru"
) -> tuple[bool, str]:
    """
    Formats the model's JSON output into a sanitized HTML report.
    Handles specialized notices for private chats (private_chat) and non-text images (no_text_image).
    """
    try:
        parsed = extract_clean_json(raw_text)
        content_type = "news_or_public"
        if isinstance(parsed, dict):
            content_type = str(parsed.get("content_type", "news_or_public")).strip()
            data = parsed.get("manipulations", [])
            simple_summary = str(parsed.get("simple_summary", "")).strip()
        elif isinstance(parsed, list):
            data = parsed
            simple_summary = ""
        else:
            data = []
            simple_summary = ""

        if forced_lang in UI_LABELS:
            lang = forced_lang
        elif original_text and original_text.strip():
            lang = detect_output_lang(original_text)
        else:
            lang = fallback_ui_lang if fallback_ui_lang in UI_LABELS else "ru"

        labels = UI_LABELS.get(lang, UI_LABELS["ru"])

        # Gatekeeper 1: Private conversation (GDPR / Do No Harm)
        if content_type == "private_chat":
            return False, f"{labels['private_chat']}\n\n<i>{labels['watermark']}</i>"

        # Gatekeeper 2: Image/photo without news text
        if content_type == "no_text_image" and not original_text.strip():
            return False, f"{labels['no_text_image']}\n\n<i>{labels['watermark']}</i>"

        def build_footer(labels_dict: dict, summary_text: str = "") -> str:
            blocks = []
            if summary_text:
                esc_sum = html.escape(summary_text)
                blocks.append(
                    f"<blockquote expandable><b>{labels_dict['simple']}</b>\n\n{esc_sum}</blockquote>"
                )
            if original_text and len(original_text.strip()) > 0:
                trimmed_orig = original_text.strip()
                if len(trimmed_orig) > 1200:
                    trimmed_orig = trimmed_orig[:1200] + "..."
                esc_orig = html.escape(trimmed_orig)
                blocks.append(
                    f"<blockquote expandable><b>{labels_dict['orig_text']}</b>\n\n{esc_orig}</blockquote>"
                )
            blocks.append(f"<i>{labels_dict['watermark']}</i>")
            return "\n\n" + "\n\n".join(blocks)

        if not isinstance(data, list) or len(data) == 0:
            clean_source = source_info.replace(f" • {labels['src_img_txt']}", "")
            source_prefix = f"🔍 <b>{labels['source']}</b> {html.escape(clean_source)}\n\n" if clean_source else ""
            return False, f"{source_prefix}{labels['clean']}{build_footer(labels)}"

        combined_text = " ".join(
            f"{item.get('explanation', '')} {item.get('span', '')}" for item in data if isinstance(item, dict)
        )
        lang = forced_lang if forced_lang in UI_LABELS else detect_output_lang(combined_text)
        labels = UI_LABELS.get(lang, UI_LABELS["ru"])

        if labels["src_img_txt"] in source_info and not any(
            w in combined_text.lower() for w in ("скриншот", "скріншот", "zrzut", "screenshot")
        ):
            source_info = source_info.replace(f" • {labels['src_img_txt']}", "").replace(labels["src_img_txt"], "")

        source_prefix = f"🔍 <b>{labels['source']}</b> {html.escape(source_info)}\n\n" if source_info else ""
        report = [f"{source_prefix}{labels['found']}"]

        for idx, item in enumerate(data, 1):
            tech_raw = str(item.get("technique", "manipulation"))
            tech_title = resolve_technique_title(tech_raw, lang)
            span = str(item.get("span", "")).strip()
            explanation = str(item.get("explanation", "")).strip()

            escaped_tech = html.escape(tech_title)
            escaped_span = html.escape(span)
            escaped_expl = html.escape(explanation)

            report.append(
                f"<b>{idx}. {escaped_tech}</b>\n"
                f"   • <i>{labels['span']}</i> «{escaped_span}»\n"
                f"   • <i>{labels['expl']}</i> {escaped_expl}\n"
            )
        return True, "\n".join(report) + build_footer(labels, simple_summary)

    except (json.JSONDecodeError, Exception):
        escaped_raw = html.escape(raw_text.strip())
        source_prefix = f"🔍 <b>Source:</b> {html.escape(source_info)}\n\n" if source_info else ""
        return True, f"{source_prefix}{VERDICT_FOUND_PREFIX}{escaped_raw}"


def build_report_init_kb(lang: str = "ru") -> InlineKeyboardMarkup:
    """Step 1: Single inline button below Viy's verdict for reporting an analysis error."""
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=labels["err_btn"], callback_data=f"err_init:{lang}")]
    ])


def build_report_reasons_kb(lang: str = "ru") -> InlineKeyboardMarkup:
    """Step 2: Inline menu of specific error categories or cancellation."""
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    r = labels["reasons"]
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=r["false_pos"], callback_data=f"err_send:{lang}:false_pos"),
            InlineKeyboardButton(text=r["missed"], callback_data=f"err_send:{lang}:missed")
        ],
        [
            InlineKeyboardButton(text=r["wrong_cat"], callback_data=f"err_send:{lang}:wrong_cat"),
            InlineKeyboardButton(text=r["bad_summary"], callback_data=f"err_send:{lang}:bad_summary")
        ],
        [
            InlineKeyboardButton(text=r["wrong_lang"], callback_data=f"err_send:{lang}:wrong_lang"),
            InlineKeyboardButton(text=labels["err_cancel_btn"], callback_data=f"err_cancel:{lang}")
        ]
    ])


NO_PREVIEW = types.LinkPreviewOptions(is_disabled=True)


async def safe_edit_text(
    target_message: types.Message,
    new_text: str,
    parse_mode: str = "HTML",
    reply_markup: InlineKeyboardMarkup | None = None
):
    """Safely edits a Telegram message with link previews disabled and automatic flood-wait retry."""
    try:
        await target_message.edit_text(
            new_text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            link_preview_options=NO_PREVIEW
        )
    except TelegramBadRequest as e:
        if "message is not modified" in str(e).lower():
            pass
        else:
            raise
    except TelegramRetryAfter as e:
        await asyncio.sleep(e.retry_after)
        await target_message.edit_text(
            new_text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            link_preview_options=NO_PREVIEW
        )


async def deliver_verdict(chat_id: int, status_msg: types.Message, full_report: str, lang: str = "ru") -> int:
    """
    Splits long reports according to Telegram length limits, disables link previews,
    attaches the localized error-reporting button, and returns the final message_id.
    """
    report_kb = build_report_init_kb(lang)
    max_len = 3900
    if len(full_report) <= max_len:
        await safe_edit_text(status_msg, full_report, parse_mode="HTML", reply_markup=report_kb)
        return status_msg.message_id

    chunks = []
    current_chunk = ""
    for paragraph in full_report.split("\n\n"):
        if len(current_chunk) + len(paragraph) + 2 < max_len:
            current_chunk += (paragraph + "\n\n")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())

    last_msg_id = status_msg.message_id
    if chunks:
        if len(chunks) == 1:
            await safe_edit_text(status_msg, chunks[0], parse_mode="HTML", reply_markup=report_kb)
        else:
            await safe_edit_text(status_msg, chunks[0], parse_mode="HTML")
            for idx, chunk in enumerate(chunks[1:], 1):
                is_last = (idx == len(chunks) - 1)
                sent = await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="HTML",
                    reply_markup=report_kb if is_last else None,
                    link_preview_options=NO_PREVIEW
                )
                last_msg_id = sent.message_id
    return last_msg_id


async def call_llm_model(model_name: str, contents: list | str, lang: str = "ru") -> tuple[str, str]:
    """
    Executes an asynchronous Gemini API call with model name validation, token telemetry logging,
    and automatic self-healing if the server suggests a newer model identifier.
    """
    if not is_safe_model_name(model_name):
        raise ValueError(f"Blocked unsafe model identifier: {model_name}")

    config = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "temperature": 0.0
    }

    async def _execute_request(target_model: str):
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            return await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config
                ),
                timeout=30.0
            )
        else:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda c=contents, m=target_model: client.models.generate_content(
                        model=m,
                        contents=c,
                        config=config
                    )
                ),
                timeout=30.0
            )

    try:
        response = await _execute_request(model_name)
    except Exception as err:
        err_str = str(err)
        suggested_match = re.search(r"use models/(gemini-[a-zA-Z0-9\.\-]+)", err_str)
        if suggested_match:
            new_model = suggested_match.group(1).strip(".")
            if is_safe_model_name(new_model):
                logger.info(f"[Self-Healing] Replacing deprecated model {model_name} with {new_model} in models.json.")
                if model_name in UNIQUE_MODELS:
                    idx = UNIQUE_MODELS.index(model_name)
                    UNIQUE_MODELS[idx] = new_model
                elif new_model not in UNIQUE_MODELS:
                    UNIQUE_MODELS.insert(0, new_model)
                save_models_to_file(UNIQUE_MODELS)
                model_name = new_model
                response = await _execute_request(model_name)
            else:
                raise
        else:
            raise

    usage = getattr(response, "usage_metadata", None)
    p_tok = getattr(usage, "prompt_token_count", 0) or 0
    o_tok = getattr(usage, "candidates_token_count", 0) or 0
    t_tok = getattr(usage, "total_token_count", 0) or (p_tok + o_tok)

    try:
        log_token_usage(model_name, p_tok, o_tok, t_tok, from_cache=False)
    except Exception as e:
        logger.warning(f"Failed to log token usage to DB: {e}")

    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    token_line = labels["tok_api"].format(t=t_tok, p=p_tok, o=o_tok)
    return (response.text or ""), token_line


@dp.message(CommandStart())
async def start_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    tg_lang = message.from_user.language_code if message.from_user else None
    ui_lang = resolve_ui_lang(user_id, tg_lang)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])

    parts = (message.text or "").split(maxsplit=1)
    utm_source = parts[1].strip() if len(parts) > 1 else None
    try:
        track_user_event(user_id, tg_lang_code=tg_lang, acquisition_source=utm_source)
    except Exception as e:
        logger.warning(f"Failed to track /start event: {e}")

    current_mode = get_user_lang_mode(user_id)
    mode_label = labels["lang_modes"].get(current_mode, labels["lang_modes"]["auto"])
    try:
        await bot.set_my_commands(
            get_localized_bot_commands(ui_lang),
            scope=types.BotCommandScopeChat(chat_id=message.chat.id)
        )
    except Exception:
        pass
    await message.answer(labels["start_text"].format(mode_label=mode_label), parse_mode="HTML")


@dp.message(Command("lang"))
async def lang_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    tg_lang = message.from_user.language_code if message.from_user else None
    ui_lang = resolve_ui_lang(user_id, tg_lang)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])
    current_mode = get_user_lang_mode(user_id)
    await message.answer(
        labels["lang_text"],
        parse_mode="HTML",
        reply_markup=build_lang_keyboard(current_mode, ui_lang)
    )


@dp.callback_query(F.data.startswith("setlang:"))
async def setlang_callback(callback: types.CallbackQuery):
    if not callback.data or not callback.from_user:
        return
    new_mode = callback.data.split(":", 1)[1]
    set_user_lang_mode(callback.from_user.id, new_mode)

    ui_lang = resolve_ui_lang(callback.from_user.id, callback.from_user.language_code)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])
    label = labels["lang_modes"].get(new_mode, new_mode)

    if callback.message:
        try:
            await bot.set_my_commands(
                get_localized_bot_commands(ui_lang),
                scope=types.BotCommandScopeChat(chat_id=callback.message.chat.id)
            )
            await callback.message.edit_text(
                labels["lang_text"],
                parse_mode="HTML",
                reply_markup=build_lang_keyboard(new_mode, ui_lang)
            )
        except TelegramBadRequest:
            pass
    await callback.answer(labels["lang_toast"].format(label=label))


@dp.message(Command("scores"))
async def scores_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    tg_lang = message.from_user.language_code if message.from_user else None
    ui_lang = resolve_ui_lang(user_id, tg_lang)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])

    top_channels = get_top_channels_report(limit=7)
    token_stats = get_token_stats()

    lines = [labels["scores_title"]]
    if not top_channels:
        lines.append(labels["scores_empty"])
    else:
        lines.append(labels["scores_sub"])
        for idx, (ch_title, cnt, avg_tech) in enumerate(top_channels, 1):
            lines.append(
                labels["scores_item"].format(idx=idx, title=html.escape(ch_title), avg=avg_tech, cnt=cnt)
            )
        lines.append("")

    lines.append(
        labels["scores_telemetry"].format(
            today_tok=token_stats["today_tokens"],
            today_calls=token_stats["today_calls"],
            total_tok=token_stats["total_tokens"],
            total_calls=token_stats["total_api_calls"],
            cache_hits=token_stats["cache_hits"]
        )
    )
    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("about"))
async def about_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    tg_lang = message.from_user.language_code if message.from_user else None
    ui_lang = resolve_ui_lang(user_id, tg_lang)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])
    await message.answer(labels["about_text"], parse_mode="HTML")


@dp.message(Command("privacy"))
async def privacy_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    tg_lang = message.from_user.language_code if message.from_user else None
    ui_lang = resolve_ui_lang(user_id, tg_lang)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=labels["privacy_del_btn"], callback_data=f"priv_del:{ui_lang}")]
    ])
    await message.answer(labels["privacy_text"], parse_mode="HTML", reply_markup=kb)


@dp.callback_query(F.data.startswith("priv_del:"))
async def priv_del_callback(callback: types.CallbackQuery):
    if not callback.from_user:
        return
    lang = callback.data.split(":")[1] if (callback.data and ":" in callback.data) else "ru"
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    delete_user_data(callback.from_user.id)
    if callback.message:
        try:
            await callback.message.edit_text(labels["privacy_del_done"], parse_mode="HTML")
        except TelegramBadRequest:
            pass
    await callback.answer(labels["privacy_del_done"], show_alert=False)


def is_unlimited_user(user_id: int) -> bool:
    """Checks whether the user ID is in the comma-separated ADMIN_USER_ID whitelist in .env."""
    raw_ids = os.getenv("ADMIN_USER_ID", "")
    allowed_ids = {
        int(x.strip()) for x in raw_ids.split(",") if x.strip().isdigit()
    }
    return user_id in allowed_ids


def is_super_admin(user_id: int) -> bool:
    """Strictly checks if the user is the primary owner (the very first ID in ADMIN_USER_ID)."""
    raw_ids = os.getenv("ADMIN_USER_ID", "")
    first_id = next((int(x.strip()) for x in raw_ids.split(",") if x.strip().isdigit()), 0)
    return bool(first_id and user_id == first_id)


@dp.message(Command("admin_stats"))
async def admin_stats_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    if not is_super_admin(user_id):
        return

    s = get_admin_analytics_summary()
    total_u = s["total_users"]
    langs_str = ", ".join(
        f"<code>{code}</code>: {cnt} ({round(cnt / total_u * 100)}%)"
        for code, cnt in s["langs"]
    ) if total_u > 0 else "no data yet"

    sources_str = ", ".join(
        f"<code>{src}</code>: {cnt}" for src, cnt in s["sources"]
    ) if s["sources"] else "organic"

    report_text = (
        "🔒 <b>Viy Owner Analytics & Unit Economics</b>\n\n"
        f"👥 <b>Unique Users:</b> <code>{total_u}</code>\n"
        f"• Active Today (DAU): <code>{s['dau_today']}</code> | Last 7d (WAU): <code>{s['wau_7d']}</code>\n"
        f"• Multi-day Retention: <code>{s['retention_rate']}%</code> ({s['retained_users']} users)\n\n"
        f"🌍 <b>Telegram Regions/Langs:</b>\n{langs_str}\n\n"
        f"🔗 <b>Acquisition Sources (UTM):</b>\n{sources_str}\n\n"
        f"📊 <b>Usage & Unit Economics:</b>\n"
        f"• Text checks: <code>{s['text_requests']}</code> | Image checks: <code>{s['image_requests']}</code>\n"
        f"• Cache hits (0 tokens): <code>{s['cache_hits']}</code>\n"
        f"• Total tokens spent: <code>{s['total_tokens']}</code>\n"
        f"• Avg tokens / user: <code>{s['avg_tokens_per_user']}</code>\n"
        f"• Calibration error reports: <code>{s['errors_reported']}</code>"
    )

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📥 Users CSV (Excel)", callback_data="admin_csv:user_analytics"),
            InlineKeyboardButton(text="📥 Errors CSV (Excel)", callback_data="admin_csv:error_reports")
        ],
        [
            InlineKeyboardButton(text="📥 Channels CSV (Excel)", callback_data="admin_csv:channel_stats")
        ]
    ])
    await message.answer(report_text, parse_mode="HTML", reply_markup=kb)


@dp.callback_query(F.data.startswith("admin_csv:"))
async def admin_csv_callback(callback: types.CallbackQuery):
    if not callback.from_user or not is_super_admin(callback.from_user.id):
        await callback.answer("Access denied.", show_alert=True)
        return
    table_name = callback.data.split(":", 1)[1] if callback.data else "user_analytics"
    try:
        csv_bytes = export_table_to_csv(table_name)
        filename = f"viy_{table_name}_{time.strftime('%Y%m%d')}.csv"
        doc = types.BufferedInputFile(csv_bytes, filename=filename)
        if callback.message:
            await callback.message.answer_document(
                document=doc,
                caption=f"📊 Exported table: <code>{table_name}</code> (UTF-8 CSV for Excel/SQLite tools)",
                parse_mode="HTML"
            )
        await callback.answer("CSV exported!")
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        await callback.answer("Export error.", show_alert=True)


@dp.message(Command("myid"))
async def myid_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    await message.answer(
        f"🔑 Telegram User ID: <code>{user_id}</code>\n"
        f"<i>(tap the number to copy)</i>",
        parse_mode="HTML"
    )


@dp.message(Command("feedback"))
async def feedback_cmd(message: types.Message):
    user_id = message.from_user.id if message.from_user else message.chat.id
    tg_lang = message.from_user.language_code if message.from_user else None
    ui_lang = resolve_ui_lang(user_id, tg_lang)
    labels = UI_LABELS.get(ui_lang, UI_LABELS["ru"])
    await message.answer(labels["feedback_text"], parse_mode="HTML")


@dp.callback_query(F.data.startswith("err_init"))
async def err_init_callback(callback: types.CallbackQuery):
    lang = callback.data.split(":")[1] if (callback.data and ":" in callback.data) else "ru"
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    if callback.message:
        try:
            await callback.message.edit_reply_markup(reply_markup=build_report_reasons_kb(lang))
        except TelegramBadRequest:
            pass
    await callback.answer(labels["err_choose"])


@dp.callback_query(F.data.startswith("err_cancel"))
async def err_cancel_callback(callback: types.CallbackQuery):
    lang = callback.data.split(":")[1] if (callback.data and ":" in callback.data) else "ru"
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    if callback.message:
        try:
            await callback.message.edit_reply_markup(reply_markup=build_report_init_kb(lang))
        except TelegramBadRequest:
            pass
    await callback.answer(labels["err_canceled"])


@dp.callback_query(F.data.startswith("err_send:"))
async def err_send_callback(callback: types.CallbackQuery):
    if not callback.data or not callback.message or not callback.from_user:
        return
    parts = callback.data.split(":")
    lang = parts[1] if len(parts) >= 3 else "ru"
    reason_code = parts[2] if len(parts) >= 3 else parts[1]
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    reason_label = labels["reasons"].get(reason_code, reason_code)

    save_error_report(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        user_id=callback.from_user.id,
        error_reason=reason_label
    )
    invalidate_cache_for_message(callback.message.chat.id, callback.message.message_id)
    try:
        track_user_event(
            callback.from_user.id,
            tg_lang_code=callback.from_user.language_code,
            is_error_report=True
        )
    except Exception:
        pass

    done_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=labels["err_saved_btn"].format(reason=reason_label), callback_data=f"err_done:{lang}")]
    ])
    try:
        await callback.message.edit_reply_markup(reply_markup=done_kb)
    except TelegramBadRequest:
        pass
    await callback.answer(labels["err_saved_toast"], show_alert=False)


@dp.callback_query(F.data.startswith("err_done"))
async def err_done_callback(callback: types.CallbackQuery):
    lang = callback.data.split(":")[1] if (callback.data and ":" in callback.data) else "ru"
    labels = UI_LABELS.get(lang, UI_LABELS["ru"])
    await callback.answer(labels["err_already"])


@dp.message()
async def analyze_message(message: types.Message):
    chat_id = message.chat.id
    raw_text = strip_channel_footer((message.text or message.caption or "").strip())

    image_bytes: bytes | None = None
    image_mime: str = "image/jpeg"

    if message.photo:
        optimal_photo = message.photo[-2] if len(message.photo) > 1 else message.photo[-1]
        image_mime = "image/jpeg"
        media_obj = optimal_photo
    elif (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        image_mime = message.document.mime_type
        media_obj = message.document
    else:
        media_obj = None

    user_id = message.from_user.id if message.from_user else chat_id
    tg_lang = message.from_user.language_code if message.from_user else None
    user_ui_lang = resolve_ui_lang(user_id, tg_lang)
    target_lang = resolve_user_target_lang(user_id, tg_lang)
    if target_lang in UI_LABELS:
        status_lang = target_lang
    elif raw_text:
        status_lang = detect_output_lang(raw_text)
    else:
        status_lang = user_ui_lang
    status_labels = UI_LABELS.get(status_lang, UI_LABELS["ru"])

    if not raw_text and not media_obj:
        await message.answer(status_labels["empty_msg"])
        return

    if not is_unlimited_user(user_id) and not check_and_increment_quota(user_id, DAILY_USER_LIMIT):
        await message.answer(status_labels["limit_err"].format(limit=DAILY_USER_LIMIT), parse_mode="HTML")
        return

    source_parts = []
    if message.forward_origin:
        origin = message.forward_origin
        if origin.type == "channel" and origin.chat.title:
            source_parts.append(f"{status_labels['src_channel']} «{origin.chat.title}»")
        elif origin.type == "user" and origin.sender_user.full_name:
            source_parts.append(f"{status_labels['src_user']} {origin.sender_user.full_name}")
        elif origin.type == "hidden_user" and origin.sender_user_name:
            source_parts.append(f"{status_labels['src_author']} {origin.sender_user.name}")
        elif origin.type == "chat" and origin.sender_chat.title:
            source_parts.append(f"{status_labels['src_chat']} «{origin.sender_chat.title}»")

    if media_obj and raw_text:
        source_parts.append(status_labels["src_img_txt"])
    elif media_obj and not raw_text:
        source_parts.append(status_labels["src_img_only"])

    source_info = " • ".join(source_parts) if source_parts else status_labels["src_direct"]

    status_msg = await message.answer(status_labels["status_wait"], parse_mode="HTML")
    last_status_time = time.monotonic()

    async def update_status_safely(new_status_text: str):
        nonlocal last_status_time
        now = time.monotonic()
        elapsed = now - last_status_time
        if elapsed < MIN_STATUS_DURATION:
            await asyncio.sleep(MIN_STATUS_DURATION - elapsed)
        await safe_edit_text(status_msg, new_status_text, parse_mode="HTML")
        last_status_time = time.monotonic()

    if media_obj:
        try:
            file_info = await bot.get_file(media_obj.file_id)
            if not file_info.file_path:
                raise ValueError("File path not found on Telegram server.")
            downloaded_stream = await bot.download_file(file_info.file_path)
            if downloaded_stream:
                image_bytes = downloaded_stream.read()
        except Exception as dl_err:
            logger.error(f"Media download error: {dl_err}")
            await safe_edit_text(status_msg, status_labels["dl_err"], parse_mode="HTML")
            return

    lang_instruction = (
        f"\n\n[SYSTEM LANGUAGE REQUIREMENT: Write all `explanation` and `simple_summary` fields strictly in '{target_lang}' regardless of input language!]"
        if target_lang != "auto" else ""
    )

    links_scout_report = await scout_telegram_links(message, raw_text)
    text_for_llm = (raw_text + links_scout_report + lang_instruction) if raw_text else lang_instruction

    contents_payload = []
    if image_bytes:
        contents_payload.append(
            genai_types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
        )
    if text_for_llm:
        contents_payload.append(text_for_llm)
    elif image_bytes:
        contents_payload.append(
            "Recognize all text on this image and audit it for rhetorical manipulation according to the system instructions."
        )

    successful_response: str | None = None
    token_report_line: str = ""
    prev_channels: list[str] = []

    is_public_channel = bool(
        message.forward_origin
        and message.forward_origin.type == "channel"
        and getattr(message.forward_origin.chat, "username", None)
    )
    channel_name_for_db = (
        message.forward_origin.chat.title
        if is_public_channel and message.forward_origin.chat.title
        else ""
    )
    text_hash = (
        make_text_hash(raw_text, SYSTEM_PROMPT, target_lang, image_bytes)
        if (is_public_channel and (raw_text or image_bytes))
        else ""
    )

    if text_hash:
        cached_json, prev_channels = get_cached_verdict(text_hash)
        if cached_json:
            successful_response = cached_json
            log_token_usage("cache", 0, 0, 0, from_cache=True)
            token_report_line = status_labels["tok_cache"]

    if not successful_response:
        for index, model_name in enumerate(list(UNIQUE_MODELS)):
            if index > 0:
                await update_status_safely(status_labels["status_fallback"])

            try:
                raw_result, tok_line = await call_llm_model(model_name, contents_payload, lang=status_lang)
                if raw_result and raw_result.strip():
                    successful_response = raw_result
                    token_report_line = tok_line
                    break
            except Exception as model_err:
                logger.warning(f"[Fallback Log] Model {model_name} failed ({type(model_err).__name__}): {model_err}")
                if len(UNIQUE_MODELS) > 1 and UNIQUE_MODELS[0] == model_name:
                    UNIQUE_MODELS.append(UNIQUE_MODELS.pop(0))
                    logger.info(f"[Queue Rotation] Model {model_name} moved to the back of the in-memory queue: {UNIQUE_MODELS}")
                await asyncio.sleep(0.5)

        if successful_response and is_public_channel and text_hash:
            try:
                parsed_json = extract_clean_json(successful_response)
                parsed_items = parsed_json.get("manipulations", []) if isinstance(parsed_json, dict) else parsed_json
                tech_codes = [
                    str(it.get("technique", "")) for it in parsed_items if isinstance(it, dict)
                ] if isinstance(parsed_items, list) else []
                save_public_analysis(text_hash, successful_response, channel_name_for_db, tech_codes)
            except Exception as db_err:
                logger.error(f"[DB Log] Failed to save analysis to cache: {db_err}")

    if not successful_response:
        await update_status_safely(status_labels["api_err"])
        return

    display_source_info = source_info
    if is_public_channel and channel_name_for_db:
        ch_score = get_channel_score(channel_name_for_db)
        if ch_score:
            cnt_posts, avg_tech = ch_score
            display_source_info += status_labels["score_fmt"].format(cnt=cnt_posts, avg=avg_tech)

    await update_status_safely(status_labels["status_analysis"])

    now = time.monotonic()
    time_shown = now - last_status_time
    if time_shown < MIN_STATUS_DURATION:
        await asyncio.sleep(MIN_STATUS_DURATION - time_shown)

    _, final_report = format_analysis(
        successful_response,
        source_info=display_source_info,
        forced_lang=target_lang,
        original_text=raw_text,
        fallback_ui_lang=user_ui_lang
    )
    other_channels = [ch for ch in prev_channels if ch and ch != channel_name_for_db]
    if other_channels:
        channels_str = ", ".join(f"«{html.escape(c)}»" for c in other_channels[:3])
        final_report += "\n\n" + status_labels["trace"].format(channels=channels_str)

    if token_report_line:
        final_report += f"\n\n{token_report_line}"

    final_msg_id = await deliver_verdict(chat_id, status_msg, final_report, lang=status_lang)
    save_message_context(chat_id, final_msg_id, display_source_info, raw_text, successful_response)

    try:
        used_cache = (token_report_line == status_labels["tok_cache"])
        tok_match = re.search(r":\s*(\d+)", token_report_line) if not used_cache else None
        spent_tokens = int(tok_match.group(1)) if tok_match else 0
        track_user_event(
            user_id,
            tg_lang_code=tg_lang,
            is_text_req=bool(raw_text and not media_obj),
            is_image_req=bool(media_obj),
            is_cache_hit=used_cache,
            tokens_added=spent_tokens
        )
    except Exception as e:
        logger.warning(f"Failed to record user analytics: {e}")


async def periodic_models_refresh(interval_hours: int = 24):
    """Background task: refreshes the live Gemini model catalog every 24 hours without restarting the bot."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, refresh_available_models)
        except Exception as e:
            logger.warning(f"[Auto-Models] Scheduled 24h model refresh failed: {e}")


async def main():
    init_db()
    refresh_available_models()
    refresh_task = asyncio.create_task(periodic_models_refresh(24))
    for l_code in ("ru", "uk", "pl", "en"):
        try:
            await bot.set_my_commands(get_localized_bot_commands(l_code), language_code=l_code)
        except Exception:
            pass
    await bot.set_my_commands(get_localized_bot_commands("ru"))
    logger.info("Viy Bot started successfully! 4-language UI, 24h auto-model refresh, logging, and SQLite DB are active.")
    try:
        await dp.start_polling(bot)
    finally:
        refresh_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())