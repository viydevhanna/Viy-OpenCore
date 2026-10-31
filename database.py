import io
import csv
import sqlite3
import hashlib
import re
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "viy_cache.db"
CACHE_TTL_SECONDS = 14 * 24 * 60 * 60  # Retain cached verdicts and message context for 14 days

# DEV TOGGLE: When True, any modification in prompt.txt invalidates previous cache entries.
# Set to False for production to keep cache persistent across minor prompt tweaks.
RESET_CACHE_ON_PROMPT_CHANGE = False


def _hash_user_id(user_id: int) -> str:
    """Anonymizes Telegram User ID via salted SHA-256 hashing (GDPR data minimization)."""
    return hashlib.sha256(f"viy_user_salt_{user_id}".encode("utf-8")).hexdigest()


def init_db():
    """Initializes all SQLite tables (cache, channel scores, user settings, token telemetry, error reports, quotas, analytics) and purges expired entries."""
    with sqlite3.connect(DB_PATH) as conn:
        # 1. Hot cache table for public channel verdicts (14-day TTL)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS post_cache (
                text_hash TEXT PRIMARY KEY,
                raw_json_response TEXT,
                first_channel TEXT,
                created_at INTEGER
            )
        """)
        # 2. Persistent statistics for public channels (channel scoring)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_title TEXT,
                text_hash TEXT,
                techniques_count INTEGER,
                techniques_list TEXT,
                checked_at INTEGER
            )
        """)
        # 3. User language preferences (indexed by anonymized User ID hash)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_hash TEXT PRIMARY KEY,
                lang_mode TEXT DEFAULT 'auto',
                updated_at INTEGER
            )
        """)
        # 4. API token consumption and cache hit telemetry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER,
                model_name TEXT,
                prompt_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                from_cache INTEGER DEFAULT 0
            )
        """)
        # 5. Ephemeral message context for inline error reporting (14-day TTL)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_context (
                chat_id INTEGER,
                message_id INTEGER,
                source_info TEXT,
                original_text TEXT,
                bot_response_json TEXT,
                created_at INTEGER,
                PRIMARY KEY (chat_id, message_id)
            )
        """)
        # 6. Calibration log of user-submitted error reports
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER,
                user_hash TEXT,
                source_info TEXT,
                original_text TEXT,
                bot_response_json TEXT,
                error_reason TEXT
            )
        """)
        # 7. Auto-migration: drop legacy user_quotas schema if user_hash column is missing
        cols = [info[1] for info in conn.execute("PRAGMA table_info(user_quotas)").fetchall()]
        if cols and "user_hash" not in cols:
            conn.execute("DROP TABLE user_quotas")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_quotas (
                user_hash TEXT PRIMARY KEY,
                quota_date TEXT,
                calls_count INTEGER DEFAULT 0
            )
        """)

        # 8. Privacy-safe product & marketing analytics per anonymized user
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_analytics (
                user_hash TEXT PRIMARY KEY,
                tg_lang_code TEXT DEFAULT 'unknown',
                acquisition_source TEXT DEFAULT 'organic',
                first_seen_date TEXT,
                last_seen_date TEXT,
                active_days_count INTEGER DEFAULT 1,
                total_interactions INTEGER DEFAULT 0,
                text_requests INTEGER DEFAULT 0,
                image_requests INTEGER DEFAULT 0,
                cache_hits_count INTEGER DEFAULT 0,
                total_tokens_used INTEGER DEFAULT 0,
                errors_reported INTEGER DEFAULT 0
            )
        """)

        # Indexes for fast lookup
        conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON channel_stats(channel_title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON channel_stats(text_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tokens_time ON token_usage(created_at)")

        # Purge cache and ephemeral context older than 14 days
        cutoff = int(time.time()) - CACHE_TTL_SECONDS
        conn.execute("DELETE FROM post_cache WHERE created_at < ?", (cutoff,))
        conn.execute("DELETE FROM message_context WHERE created_at < ?", (cutoff,))
        conn.commit()


# --- SECTION 1: CONTENT HASHING & VERDICT CACHING ---

def make_text_hash(
    raw_text: str,
    system_prompt: str = "",
    target_lang: str = "auto",
    image_bytes: bytes | None = None
) -> str:
    """
    Computes an irreversible SHA-256 fingerprint from normalized post text, optional image bytes,
    target output language, and (if dev toggle is enabled) the current prompt version.
    """
    cleaned = re.sub(r"\s+", " ", (raw_text or "").strip().lower())
    cleaned = f"{cleaned}||lang:{target_lang}"
    if image_bytes:
        img_fingerprint = hashlib.md5(image_bytes).hexdigest()
        cleaned = f"{cleaned}||img:{img_fingerprint}"
    if RESET_CACHE_ON_PROMPT_CHANGE and system_prompt:
        prompt_fingerprint = hashlib.md5(system_prompt.encode("utf-8")).hexdigest()
        cleaned = f"{cleaned}||prompt_v:{prompt_fingerprint}"
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def get_cached_verdict(text_hash: str) -> tuple[str | None, list[str]]:
    """Retrieves a cached analysis JSON by hash and returns a list of public channels where this text appeared."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT raw_json_response FROM post_cache WHERE text_hash = ?",
            (text_hash,)
        ).fetchone()

        channels = conn.execute(
            "SELECT DISTINCT channel_title FROM channel_stats WHERE text_hash = ? AND channel_title != ''",
            (text_hash,)
        ).fetchall()
        channel_list = [c[0] for c in channels]

        if row:
            return row[0], channel_list
        return None, channel_list


def save_public_analysis(text_hash: str, raw_json_response: str, channel_title: str, techniques: list[str]):
    """Persists analysis results strictly for public channels (Confessional Mode compliance)."""
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO post_cache (text_hash, raw_json_response, first_channel, created_at)
            VALUES (?, ?, ?, ?)
        """, (text_hash, raw_json_response, channel_title, now))

        conn.execute("""
            INSERT INTO channel_stats (channel_title, text_hash, techniques_count, techniques_list, checked_at)
            VALUES (?, ?, ?, ?, ?)
        """, (channel_title, text_hash, len(techniques), ",".join(techniques), now))
        conn.commit()


# --- SECTION 2: USER LANGUAGE PREFERENCES ---

def get_user_lang_mode(user_id: int) -> str:
    """Returns the user's selected language mode ('auto', 'tg_system', 'ru', 'uk', 'pl', 'en')."""
    u_hash = _hash_user_id(user_id)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT lang_mode FROM user_settings WHERE user_hash = ?",
            (u_hash,)
        ).fetchone()
        return row[0] if row else "auto"


def set_user_lang_mode(user_id: int, lang_mode: str):
    """Saves the user's selected language mode."""
    u_hash = _hash_user_id(user_id)
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO user_settings (user_hash, lang_mode, updated_at)
            VALUES (?, ?, ?)
        """, (u_hash, lang_mode, now))
        conn.commit()


# --- SECTION 3: TOKEN TELEMETRY ---

def log_token_usage(model_name: str, prompt_tokens: int, output_tokens: int, total_tokens: int, from_cache: bool = False):
    """Logs API token usage per request or records a zero-cost cache hit."""
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO token_usage (created_at, model_name, prompt_tokens, output_tokens, total_tokens, from_cache)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now, model_name, prompt_tokens, output_tokens, total_tokens, 1 if from_cache else 0))
        conn.commit()


def get_token_stats() -> dict:
    """Returns token consumption statistics (last 24 hours and all-time) alongside total cache hits."""
    day_ago = int(time.time()) - 86400
    with sqlite3.connect(DB_PATH) as conn:
        today_row = conn.execute("""
            SELECT COALESCE(SUM(total_tokens), 0), COUNT(*)
            FROM token_usage WHERE from_cache = 0 AND created_at >= ?
        """, (day_ago,)).fetchone()

        total_row = conn.execute("""
            SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(output_tokens), 0),
                   COALESCE(SUM(total_tokens), 0), COUNT(*)
            FROM token_usage WHERE from_cache = 0
        """).fetchone()

        cache_hits = conn.execute("""
            SELECT COUNT(*) FROM token_usage WHERE from_cache = 1
        """).fetchone()[0]

        return {
            "today_tokens": today_row[0],
            "today_calls": today_row[1],
            "total_prompt_tokens": total_row[0],
            "total_output_tokens": total_row[1],
            "total_tokens": total_row[2],
            "total_api_calls": total_row[3],
            "cache_hits": cache_hits
        }


# --- SECTION 4: PUBLIC CHANNEL SCORING & LEADERBOARD ---

def get_channel_score(channel_title: str) -> tuple[int, float] | None:
    """Returns (checked_posts_count, avg_techniques_per_post) for a given public channel."""
    if not channel_title:
        return None
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT COUNT(*), AVG(techniques_count)
            FROM channel_stats WHERE channel_title = ?
        """, (channel_title,)).fetchone()
        if row and row[0] > 0:
            return int(row[0]), round(float(row[1]), 1)
        return None


def get_top_channels_report(limit: int = 5) -> list[tuple[str, int, float]]:
    """Returns top public channels ranked by average manipulation techniques per post."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT channel_title, COUNT(*) as cnt, ROUND(AVG(techniques_count), 1) as avg_tech
            FROM channel_stats
            WHERE channel_title != ''
            GROUP BY channel_title
            ORDER BY avg_tech DESC, cnt DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [(r[0], int(r[1]), float(r[2])) for r in rows]


# --- SECTION 5: ERROR REPORTING & CALIBRATION LOG ---

def save_message_context(chat_id: int, message_id: int, source_info: str, original_text: str, bot_response_json: str):
    """Stores ephemeral context of a delivered bot verdict so the inline error button can reference it."""
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO message_context
            (chat_id, message_id, source_info, original_text, bot_response_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (chat_id, message_id, source_info, original_text[:4000], bot_response_json, now))
        conn.commit()


def save_error_report(chat_id: int, message_id: int, user_id: int, error_reason: str) -> bool:
    """Moves a flagged analysis from message_context into the persistent error_reports table."""
    now = int(time.time())
    u_hash = _hash_user_id(user_id)
    with sqlite3.connect(DB_PATH) as conn:
        ctx = conn.execute("""
            SELECT source_info, original_text, bot_response_json
            FROM message_context WHERE chat_id = ? AND message_id = ?
        """, (chat_id, message_id)).fetchone()

        source_info = ctx[0] if ctx else "Unknown source"
        original_text = ctx[1] if ctx else "[No text stored or image-only post]"
        bot_response_json = ctx[2] if ctx else "[]"

        conn.execute("""
            INSERT INTO error_reports
            (created_at, user_hash, source_info, original_text, bot_response_json, error_reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now, u_hash, source_info, original_text, bot_response_json, error_reason))
        conn.commit()
        return True


# --- SECTION 6: GDPR RIGHT TO ERASURE, CACHE INVALIDATION & DAILY QUOTAS ---

def delete_user_data(user_id: int):
    """Permanently deletes user preferences, quota counters, and analytics profile (GDPR Right to Erasure)."""
    u_hash = _hash_user_id(user_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM user_settings WHERE user_hash = ?", (u_hash,))
        conn.execute("DELETE FROM user_quotas WHERE user_hash = ?", (u_hash,))
        conn.execute("DELETE FROM user_analytics WHERE user_hash = ?", (u_hash,))
        conn.commit()


def invalidate_cache_for_message(chat_id: int, message_id: int):
    """Removes a cached verdict from post_cache when a user reports an error in the analysis."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT bot_response_json FROM message_context WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id)
        ).fetchone()
        if row and row[0]:
            conn.execute("DELETE FROM post_cache WHERE raw_json_response = ?", (row[0],))
            conn.commit()


def check_and_increment_quota(user_id: int, daily_limit: int) -> bool:
    """
    Verifies and increments the daily request quota by anonymized User ID hash.
    Returns True if the request is allowed, or False if the daily limit is reached.
    """
    u_hash = _hash_user_id(user_id)
    today_str = time.strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT quota_date, calls_count FROM user_quotas WHERE user_hash = ?",
            (u_hash,)
        ).fetchone()
        if not row or row[0] != today_str:
            conn.execute(
                "INSERT OR REPLACE INTO user_quotas (user_hash, quota_date, calls_count) VALUES (?, ?, 1)",
                (u_hash, today_str)
            )
            conn.commit()
            return True
        if row[1] >= daily_limit:
            return False
        conn.execute(
            "UPDATE user_quotas SET calls_count = calls_count + 1 WHERE user_hash = ?",
            (u_hash,)
        )
        conn.commit()
        return True


# --- SECTION 7: PRODUCT ANALYTICS, UNIT ECONOMICS & CSV EXPORT (ADMIN ONLY) ---

def track_user_event(
    user_id: int,
    tg_lang_code: str | None = None,
    acquisition_source: str | None = None,
    is_text_req: bool = False,
    is_image_req: bool = False,
    is_cache_hit: bool = False,
    tokens_added: int = 0,
    is_error_report: bool = False
):
    """Updates anonymized user analytics and unit economics metrics in user_analytics."""
    u_hash = _hash_user_id(user_id)
    today_str = time.strftime("%Y-%m-%d")
    lang_clean = (tg_lang_code or "unknown")[:5].lower()
    src_clean = re.sub(r"[^a-zA-Z0-9_\-]", "", (acquisition_source or ""))[:32]

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT last_seen_date, active_days_count, acquisition_source FROM user_analytics WHERE user_hash = ?",
            (u_hash,)
        ).fetchone()

        if not row:
            conn.execute("""
                INSERT INTO user_analytics (
                    user_hash, tg_lang_code, acquisition_source,
                    first_seen_date, last_seen_date, active_days_count,
                    total_interactions, text_requests, image_requests,
                    cache_hits_count, total_tokens_used, errors_reported
                ) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?)
            """, (
                u_hash,
                lang_clean,
                src_clean if src_clean else "organic",
                today_str,
                today_str,
                1 if is_text_req else 0,
                1 if is_image_req else 0,
                1 if is_cache_hit else 0,
                int(tokens_added),
                1 if is_error_report else 0
            ))
        else:
            last_seen, active_days, current_src = row
            new_active_days = active_days + (1 if last_seen != today_str else 0)
            updated_src = src_clean if (src_clean and current_src == "organic") else current_src

            conn.execute("""
                UPDATE user_analytics SET
                    tg_lang_code = CASE WHEN ? != 'unknown' THEN ? ELSE tg_lang_code END,
                    acquisition_source = ?,
                    last_seen_date = ?,
                    active_days_count = ?,
                    total_interactions = total_interactions + 1,
                    text_requests = text_requests + ?,
                    image_requests = image_requests + ?,
                    cache_hits_count = cache_hits_count + ?,
                    total_tokens_used = total_tokens_used + ?,
                    errors_reported = errors_reported + ?
                WHERE user_hash = ?
            """, (
                lang_clean, lang_clean,
                updated_src,
                today_str,
                new_active_days,
                1 if is_text_req else 0,
                1 if is_image_req else 0,
                1 if is_cache_hit else 0,
                int(tokens_added),
                1 if is_error_report else 0,
                u_hash
            ))
        conn.commit()


def get_admin_analytics_summary() -> dict:
    """Computes aggregated audience, retention, region/language, and unit economics metrics for the owner."""
    today_str = time.strftime("%Y-%m-%d")
    seven_days_ago = time.strftime("%Y-%m-%d", time.localtime(time.time() - 7 * 86400))

    with sqlite3.connect(DB_PATH) as conn:
        totals = conn.execute("""
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN last_seen_date = ? THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN last_seen_date >= ? THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN active_days_count > 1 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(text_requests), 0),
                COALESCE(SUM(image_requests), 0),
                COALESCE(SUM(cache_hits_count), 0),
                COALESCE(SUM(total_tokens_used), 0),
                COALESCE(SUM(errors_reported), 0)
            FROM user_analytics
        """, (today_str, seven_days_ago)).fetchone()

        langs = conn.execute("""
            SELECT tg_lang_code, COUNT(*) as cnt
            FROM user_analytics
            GROUP BY tg_lang_code
            ORDER BY cnt DESC
        """).fetchall()

        sources = conn.execute("""
            SELECT acquisition_source, COUNT(*) as cnt
            FROM user_analytics
            GROUP BY acquisition_source
            ORDER BY cnt DESC
            LIMIT 5
        """).fetchall()

        total_users = totals[0] or 0
        retained_users = totals[3] or 0
        retention_rate = round((retained_users / total_users) * 100, 1) if total_users > 0 else 0.0
        avg_tokens_per_user = int((totals[7] or 0) / total_users) if total_users > 0 else 0

        return {
            "total_users": total_users,
            "dau_today": totals[1] or 0,
            "wau_7d": totals[2] or 0,
            "retained_users": retained_users,
            "retention_rate": retention_rate,
            "text_requests": totals[4] or 0,
            "image_requests": totals[5] or 0,
            "cache_hits": totals[6] or 0,
            "total_tokens": totals[7] or 0,
            "avg_tokens_per_user": avg_tokens_per_user,
            "errors_reported": totals[8] or 0,
            "langs": langs,
            "sources": sources
        }


def export_table_to_csv(table_name: str) -> bytes:
    """Exports an allowed analytics table (user_analytics or error_reports) to UTF-8-SIG CSV bytes for Excel."""
    if table_name not in ("user_analytics", "error_reports", "channel_stats"):
        raise ValueError("Unauthorized table export request.")

    output = io.StringIO()
    writer = csv.writer(output, delimiter=",")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(f"SELECT * FROM {table_name}")
        headers = [description[0] for description in cursor.description]
        writer.writerow(headers)
        for row in cursor.fetchall():
            writer.writerow(row)

    # utf-8-sig adds BOM so Microsoft Excel opens Cyrillic/Polish characters cleanly
    return output.getvalue().encode("utf-8-sig")