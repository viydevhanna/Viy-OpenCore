# В⌜І⌟Й (Viy) — Open-Core Linguistic Manipulation & Propaganda Scanner

**Viy** (`@viy_truth_bot`) is an asynchronous Telegram bot and closed-context linguistic analysis engine designed to detect rhetorical manipulation, logical fallacies, and emotional pressure in news publications across 4 languages: **Russian (RU), Ukrainian (UK), Polish (PL), and English (EN)**.

Unlike traditional fact-checking bots that rely on external web searches, Viy operates as a **stateless, closed-context discourse scanner**. It evaluates the internal argumentation structure and framing of texts and article screenshots based on the **SemEval-2020 Task 11** methodology, the **UNESCO Information Disorder** framework, and H. Lasswell's communication model.

---

## Key Features & Open-Core Architecture

* **14 Manipulation Techniques Taxonomy (`techniques.json`):** Detects Loaded Language, Name Calling, Exaggeration/Minimization, False Dilemma, Causal Oversimplification, Appeal to Fear/Prejudice, Appeal to Authority, Anonymous Sources, Doubt, Slogans, Whataboutism/Straw Man/Red Herring, Flag-Waving, Thought-Terminating Cliches, and Deagentization (hidden actor).
* **Multimodal & Link-Aware Auditing (`bot.py`):** Recognizes text on article screenshots, cross-checks headlines against direct quotes, and inspects embedded `t.me` links for circular self-citation.
* **Privacy by Design / Confessional Mode (`database.py`):** Complies with GDPR and Digital Public Goods (DPG) standards. Direct messages and user-uploaded screenshots are processed strictly in volatile memory (RAM) and are never cached. Users can erase their preferences at any time via `/privacy`.
* **Resilient Model Cascade:** Built on `aiogram 3.x` and `google-genai` with automatic model fallback and SQLite caching for public channel forwards.
* **Open-Core Separation:** The bot engine, database layer, UI localization (`locales.json`), and 14-technique taxonomy (`techniques.json`) are open-sourced. Production calibration prompts (`prompt.txt`) and API credentials (`.env`) remain decoupled via `.env.example` and `prompt.example.txt`.

---

## Quick Start

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Copy .env.example to .env and insert your TELEGRAM_TOKEN and GEMINI_API_KEY.
4. Copy prompt.example.txt to prompt.txt (or use it directly as a fallback template).
5. Run the bot:
    Bash
    python bot.py
    

License, Authorship & AI Transparency Disclosure
Software Code License: Licensed under the Apache License 2.0 (see LICENSE).

Documentation & Taxonomy License: The linguistic taxonomy (techniques.json), localization strings (locales.json), and project documentation (README.md) are licensed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0).

Authorship & LLM-Assisted Development Disclosure: In accordance with open-source transparency best practices, the author declares that the Python application boilerplate (bot.py, database.py) was developed using LLM-assisted programming ("vibe-coding"). The project architecture, closed-context auditing methodology, negative constraints against false positives, multilingual UI localization, and the 14-category manipulation taxonomy with domain-specific examples represent original human intellectual creation by the author.