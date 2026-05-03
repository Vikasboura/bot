"""
Vera Bot v2 — magicpin AI Challenge
Author: Vikas Boura
Key fixes vs v1:
  - Trigger merchant_id read from top-level (not payload) → fixes 0/6 trigger coverage
  - LLM-based reply handler with full context → fixes Merchant Fit & Specificity
  - Proper customer vs merchant reply branching
  - Single fast LLM call per trigger (no multi-candidate overhead)
  - Conversation history maintained for multi-turn context
"""

import os, json, time, asyncio, uuid
from datetime import datetime, timezone
from typing import Optional, Any
from fastapi import FastAPI, Request
from openai import AsyncOpenAI
import uvicorn

app = FastAPI(title="Vera Bot v2")
START = time.time()

# ── Storage ────────────────────────────────────────────────────────────────────
# (scope, context_id) → full body dict from /v1/context
CONTEXTS: dict[tuple[str, str], dict] = {}

# conversation_id → list of {from, msg} dicts
CONVERSATIONS: dict[str, list[dict]] = {}

# conversation_id → set of message bodies already sent (anti-repetition)
BOT_SENT: dict[str, set[str]] = {}

# suppression_key → unix timestamp when it expires
SUPPRESSION: dict[str, float] = {}

# merchant_ids that sent STOP/opted-out
HOSTILE: set[str] = set()

# ── LLM client ─────────────────────────────────────────────────────────────────
# Uses OpenRouter (OpenAI-compatible) — set OPENROUTER_API_KEY env var
# Defaults to the key in the original code if env var not set
_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    ""
)
client = AsyncOpenAI(api_key=_API_KEY, base_url="https://openrouter.ai/api/v1")

# Best available model on OpenRouter — Claude 3.5 Sonnet is best for this task
MODEL = os.environ.get("MODEL", "anthropic/claude-3.5-sonnet")


# ── Context helpers ─────────────────────────────────────────────────────────────

def get_ctx(scope: str, cid: str) -> dict:
    """Fetch stored context payload by scope + id."""
    return CONTEXTS.get((scope, cid), {})


def merchant_str(m: dict) -> str:
    """Serialise merchant context into a rich string for the LLM."""
    if not m:
        return "(no merchant data)"
    idn = m.get("identity", {})
    perf = m.get("performance", {})
    d7 = perf.get("delta_7d", {})
    sub = m.get("subscription", {})
    agg = m.get("customer_aggregate", {})
    offers = m.get("offers", [])
    active_offers = [o["title"] for o in offers if o.get("status") == "active"]
    signals = m.get("signals", [])
    hist = m.get("conversation_history", [])
    last_engagement = hist[-1].get("engagement", "") if hist else ""

    lines = [
        f"Name: {idn.get('name', 'Unknown')}",
        f"Location: {idn.get('locality', '')}, {idn.get('city', '')}",
        f"Languages: {', '.join(idn.get('languages', ['en']))}",
        f"Subscription: {sub.get('plan', 'N/A')} — {sub.get('days_remaining', '?')} days left, status={sub.get('status', '?')}",
        f"Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, directions={perf.get('directions', '?')}, CTR={perf.get('ctr', '?')}",
        f"7d trend: views {d7.get('views_pct', 0)*100:+.0f}%, calls {d7.get('calls_pct', 0)*100:+.0f}%",
        f"Active offers: {', '.join(active_offers) if active_offers else 'None'}",
        f"Customers YTD: {agg.get('total_unique_ytd', '?')} total, {agg.get('lapsed_180d_plus', '?')} lapsed >180d, {agg.get('retention_6mo_pct', 0)*100:.0f}% 6mo retention",
        f"Signals: {', '.join(signals) if signals else 'none'}",
        f"Last Vera engagement: {last_engagement or 'none'}",
    ]
    return "\n".join(lines)


def category_str(c: dict) -> str:
    """Serialise category context into a rich string for the LLM."""
    if not c:
        return "(no category data)"
    voice = c.get("voice", {})
    peer = c.get("peer_stats", {})
    digest = c.get("digest", [])
    catalog = c.get("offer_catalog", [])
    seasonal = c.get("seasonal_beats", [])
    trends = c.get("trend_signals", [])

    digest_lines = "\n".join(
        f"  [{d.get('source', '?')}] {d.get('title', '')} — n={d.get('trial_n', '?')}, segment={d.get('patient_segment', '?')}"
        for d in digest[:4]
    )
    catalog_titles = ", ".join(o.get("title", "") for o in catalog[:6])
    seasonal_notes = "; ".join(s.get("note", "") for s in seasonal[:3])
    trend_notes = "; ".join(
        f"{t.get('query', '')} {t.get('delta_yoy', 0)*100:+.0f}% YoY"
        for t in trends[:3]
    )

    lines = [
        f"Category: {c.get('slug', '?')}",
        f"Voice: tone={voice.get('tone', '?')}, allowed vocab={', '.join(voice.get('vocab_allowed', [])[:6])}, taboos={', '.join(voice.get('taboos', []))}",
        f"Peer stats: avg CTR={peer.get('avg_ctr', '?')}, avg rating={peer.get('avg_rating', '?')}, avg reviews={peer.get('avg_reviews', '?')}, scope={peer.get('scope', '?')}",
        f"Offer catalog: {catalog_titles or 'none'}",
        f"Latest digest:\n{digest_lines or '  (none)'}",
        f"Seasonal: {seasonal_notes or 'none'}",
        f"Trends: {trend_notes or 'none'}",
    ]
    return "\n".join(lines)


def customer_str(c: dict) -> str:
    """Serialise customer context for the LLM."""
    if not c:
        return ""
    idn = c.get("identity", {})
    rel = c.get("relationship", {})
    prefs = c.get("preferences", {})
    lines = [
        f"Customer name: {idn.get('name', '?')}",
        f"Language preference: {idn.get('language_pref', 'en')}",
        f"State: {c.get('state', '?')}",
        f"Visits: {rel.get('visits_total', '?')} total, last visit {rel.get('last_visit', '?')}, first visit {rel.get('first_visit', '?')}",
        f"Services received: {', '.join(rel.get('services_received', [])) or 'none'}",
        f"Preferred slots: {prefs.get('preferred_slots', '?')}",
        f"Consent scope: {', '.join(c.get('consent', {}).get('scope', []))}",
    ]
    return "\n".join(lines)


# ── Auto-reply & STOP detection ────────────────────────────────────────────────

_AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "thanks for reaching out",
    "i am an automated",
    "i'm an automated",
    "main ek automated",
    "we will get back",
    "our team will",
    "office hours",
    "business hours",
    "yeh ek automatic",
]

def is_auto_reply(history: list[dict], msg: str) -> bool:
    msg_l = msg.lower()
    if any(p in msg_l for p in _AUTO_REPLY_PHRASES):
        return True
    # Same message 3× in a row
    if len(history) >= 3 and len({t["msg"] for t in history[-3:]}) == 1:
        return True
    # Same message 2× in a row after a bot turn
    if len(history) >= 2 and len({t["msg"] for t in history[-2:]}) == 1:
        return True
    return False


_STOP_PHRASES = [
    "stop", "unsubscribe", "opt out", "opt-out", "remove me",
    "don't contact", "do not contact", "not interested",
    "band karo", "mat bhejo", "nahi chahiye", "nahin chahiye",
]

def is_stop(msg: str) -> bool:
    ml = msg.lower()
    return any(p in ml for p in _STOP_PHRASES)


# ── LLM composition ────────────────────────────────────────────────────────────

async def _llm(prompt: str, temp: float = 0.6, timeout: float = 22.0) -> Optional[dict]:
    """Single LLM call; returns parsed JSON dict or None on error."""
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temp,
            ),
            timeout=timeout,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[LLM error] {e}")
        return None


async def compose_proactive(
    merchant: dict,
    category: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compose a proactive WhatsApp message from Vera to a merchant (or from the
    merchant on behalf to a customer for customer-scoped triggers).
    """
    trigger_kind = trigger.get("kind", "")
    trigger_scope = trigger.get("scope", "merchant")
    trigger_payload = trigger.get("payload", {})
    trigger_urgency = trigger.get("urgency", 1)

    is_customer_scope = (trigger_scope == "customer") or (customer is not None)
    send_as = "merchant_on_behalf" if is_customer_scope else "vera"

    m_str = merchant_str(merchant)
    c_str = category_str(category)
    cu_str = customer_str(customer) if customer else ""

    # Pull digest item referenced by trigger (if any)
    digest_item_id = trigger_payload.get("top_item_id", "")
    digest_detail = ""
    if digest_item_id:
        for d in category.get("digest", []):
            if d.get("id") == digest_item_id:
                digest_detail = (
                    f"Title: {d.get('title', '')}\n"
                    f"Source: {d.get('source', '')}\n"
                    f"Trial N: {d.get('trial_n', 'N/A')}\n"
                    f"Patient segment: {d.get('patient_segment', '')}\n"
                    f"Summary: {d.get('summary', '')}"
                )
                break

    prompt = f"""You are Vera, magicpin's AI merchant assistant composing a single WhatsApp message.

━━━ MERCHANT ━━━
{m_str}

━━━ CATEGORY ━━━
{c_str}

━━━ TRIGGER ━━━
Kind: {trigger_kind}
Scope: {trigger_scope}
Urgency: {trigger_urgency}/5
Payload: {json.dumps(trigger_payload)}
{f"Referenced digest item:{chr(10)}{digest_detail}" if digest_detail else ""}

{f"━━━ CUSTOMER (send on merchant's behalf) ━━━{chr(10)}{cu_str}" if cu_str else ""}

━━━ YOUR TASK ━━━
Compose ONE WhatsApp message from {"the merchant to their customer" if is_customer_scope else "Vera to the merchant"}.

━━━ SCORING CRITERIA (what the judge checks) ━━━
1. SPECIFICITY — anchor on a real number/stat/price from the context (% / ₹ / count / source citation)
2. CATEGORY FIT — tone and vocab must match the category voice (taboos strictly off-limits)
3. MERCHANT FIT — personalise to THIS merchant's actual numbers, offers, signals, language pref
4. TRIGGER RELEVANCE — the message must clearly say WHY NOW (the trigger is the reason)
5. ENGAGEMENT COMPULSION — use 2–3 of: loss aversion, social proof, curiosity, effort externalization, single binary CTA

━━━ TRIGGER-SPECIFIC RULES ━━━
- research_digest → cite the EXACT study title and source; mention trial N and patient segment; offer to send abstract
- recall_due → use customer's REAL name, mention their last visit date, offer 2 specific time slots, show offer price
- perf_dip → show actual CTR ({category.get('peer_stats', {}).get('avg_ctr', '?')} peer avg vs merchant's), specific drop %, offer a concrete fix
- perf_spike → celebrate with exact numbers, offer to capitalise (e.g. run a post / update offer)
- festival_upcoming → name the festival, propose a specific campaign with price/copy, give deadline
- competitor_opened → frame as "someone new in your area" (don't fabricate name), suggest differentiation
- regulation_change → cite the regulation source, explain compliance impact for THIS category, offer to handle it
- dormant_with_vera → curiosity hook with a fresh insight from context (not a reminder)
- review_theme_emerged → name the review theme, show count, offer a response template
- milestone_reached → celebrate with exact number, offer next milestone action
- scheduled_recurring → education or curiosity hook from digest/trends, not a reminder

━━━ HARD RULES ━━━
- Language: match merchant's language preference ({', '.join(merchant.get('identity', {}).get('languages', ['en']))}) — hi-en code-mix is fine and encouraged
- Taboo words never allowed: {', '.join(category.get('voice', {}).get('taboos', []))}
- EXACTLY ONE CTA at the end (Reply YES / Reply 1 or 2 / etc.)
- No generic phrases: "boost your business", "increase sales", "grow revenue"
- No fabrication — use ONLY data that appears in context above
- Keep it concise (under 300 chars ideal for WhatsApp)
- Do NOT re-introduce yourself after the first message

━━━ COMPULSION LEVERS ━━━
Pick 2–3: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, single binary commit

━━━ OUTPUT ━━━
Return ONLY valid JSON:
{{
  "body": "<the WhatsApp message>",
  "cta": "binary_yes_no" | "slot_choice" | "open_ended" | "none",
  "send_as": "{send_as}",
  "rationale": "<1 sentence: trigger + merchant signal + compulsion lever used>"
}}"""

    result = await _llm(prompt, temp=0.65)
    return result


async def compose_reply_msg(
    conv_id: str,
    merchant_id: Optional[str],
    customer_id: Optional[str],
    from_role: str,
    message: str,
) -> dict:
    """
    Compose a reply to an inbound message from a merchant or customer.
    Uses full context + conversation history.
    """
    merchant = get_ctx("merchant", merchant_id).get("payload", {}) if merchant_id else {}
    cat_slug = merchant.get("category_slug", "")
    category = get_ctx("category", cat_slug).get("payload", {}) if cat_slug else {}
    customer = get_ctx("customer", customer_id).get("payload", {}) if customer_id else {}

    history = CONVERSATIONS.get(conv_id, [])
    history_str = "\n".join(
        f"  [{t['from'].upper()}]: {t['msg']}"
        for t in history[-8:]
    )

    m_str = merchant_str(merchant)
    cu_str = customer_str(customer) if customer else ""

    prompt = f"""You are Vera, magicpin's AI merchant assistant, handling an inbound WhatsApp reply.

━━━ MERCHANT ━━━
{m_str}

━━━ CATEGORY ━━━
Slug: {cat_slug}
Voice: {category.get('voice', {}).get('tone', 'conversational')}
Taboos: {', '.join(category.get('voice', {}).get('taboos', []))}
Active offers: {', '.join(o['title'] for o in merchant.get('offers', []) if o.get('status') == 'active') or 'none'}
Peer avg CTR: {category.get('peer_stats', {}).get('avg_ctr', '?')}

{f"━━━ CUSTOMER ━━━{chr(10)}{cu_str}{chr(10)}" if cu_str else ""}
━━━ CONVERSATION SO FAR ━━━
{history_str or "  (no prior turns)"}

━━━ LATEST INBOUND ━━━
from_role: {from_role}
message: "{message}"

━━━ YOUR TASK ━━━
Decide the best next move. Rules:
- from_role = "merchant" → you are Vera replying to the merchant
- from_role = "customer" → you are the merchant replying to their customer (use merchant's name/voice)
- If merchant/customer says YES / OK / do it / go ahead / let's do it → ACTION immediately, don't ask qualifying questions again
- If they ask for a specific service or booking → confirm with price from active offers + offer 2 slots
- If they ask a question → answer it with real data from context
- If they send a question after hostility → stay on-mission politely, don't drift
- Keep replies SHORT (2–3 sentences max), always end with ONE next step
- Language: match merchant pref ({', '.join(merchant.get('identity', {}).get('languages', ['en']))})

━━━ OUTPUT ━━━
Return EXACTLY ONE of these JSON shapes:

Send a message:
{{"action": "send", "body": "<reply>", "cta": "open_ended|binary_yes_no|slot_choice", "rationale": "<why>"}}

Wait for merchant to think:
{{"action": "wait", "wait_seconds": 1800, "rationale": "<why>"}}

End conversation (STOP / completed / 3 unanswered):
{{"action": "end", "rationale": "<why>"}}"""

    result = await _llm(prompt, temp=0.5)
    if result and result.get("action") in ("send", "wait", "end"):
        return result
    # Fallback — keep it safe
    return {
        "action": "send",
        "body": "Got it — let me take care of that. Reply YES to proceed.",
        "cta": "binary_yes_no",
        "rationale": "LLM fallback",
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in CONTEXTS:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vikas Boura",
        "team_members": ["Vikas"],
        "model": MODEL,
        "approach": (
            "Single high-quality LLM composition per trigger with full 4-context injection. "
            "LLM-based reply handler with conversation history. "
            "Auto-reply detection, STOP handling, suppression logic."
        ),
        "contact_email": "vikasboura942@gmail.com",
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(request: Request):
    data = await request.json()
    scope = data.get("scope")
    cid = data.get("context_id")
    version = data.get("version", 1)

    if scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"Unknown: {scope}"}

    key = (scope, cid)
    existing = CONTEXTS.get(key)
    if existing and existing.get("version", 0) >= version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": existing["version"],
        }

    CONTEXTS[key] = data
    return {
        "accepted": True,
        "ack_id": f"ack_{cid}_v{version}",
        "stored_at": datetime.now(timezone.utc).isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(request: Request):
    data = await request.json()
    available_triggers: list[str] = data.get("available_triggers", [])

    async def handle_trigger(tid: str) -> Optional[dict]:
        trigger_data = CONTEXTS.get(("trigger", tid))
        if not trigger_data:
            return None

        # ── CRITICAL FIX: merchant_id is at the TOP LEVEL of the trigger context,
        #    NOT inside trigger_data["payload"]. The original bot looked inside
        #    payload, always got None, so all 6 trigger kinds returned empty.
        merchant_id: Optional[str] = trigger_data.get("merchant_id")
        customer_id: Optional[str] = trigger_data.get("customer_id")

        if not merchant_id:
            return None

        if merchant_id in HOSTILE:
            return None

        # Suppression check
        sup_key = trigger_data.get("suppression_key", "")
        if sup_key and SUPPRESSION.get(sup_key, 0) > time.time():
            return None

        # Lookup merchant
        merchant_ctx = CONTEXTS.get(("merchant", merchant_id))
        if not merchant_ctx:
            return None
        merchant = merchant_ctx.get("payload", {})

        # Lookup category
        cat_slug = merchant.get("category_slug", "")
        category_ctx = CONTEXTS.get(("category", cat_slug))
        category = category_ctx.get("payload", {}) if category_ctx else {}

        # Lookup customer (optional)
        customer: Optional[dict] = None
        if customer_id:
            cust_ctx = CONTEXTS.get(("customer", customer_id))
            customer = cust_ctx.get("payload", {}) if cust_ctx else None

        # Conv id — unique per merchant + trigger
        conv_id = f"conv_{merchant_id}_{tid}"

        # Don't re-initiate a conversation already started by this trigger
        if BOT_SENT.get(conv_id):
            return None

        # Compose
        result = await compose_proactive(merchant, category, trigger_data, customer)
        if not result or not result.get("body", "").strip():
            return None

        body = result["body"].strip()

        # Anti-repetition across conversations
        BOT_SENT.setdefault(conv_id, set())
        if body in BOT_SENT[conv_id]:
            return None
        BOT_SENT[conv_id].add(body)

        # Set suppression
        if sup_key:
            SUPPRESSION[sup_key] = time.time() + 86400 * 7

        m_name = merchant.get("identity", {}).get("name", "")
        return {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": tid,
            "template_name": f"vera_{trigger_data.get('kind', 'generic')}_v2",
            "template_params": [m_name, body[:60]],
            "body": body,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": sup_key,
            "rationale": result.get("rationale", ""),
        }

    # Process all triggers in parallel (cap at 20 per tick per spec)
    results = await asyncio.gather(
        *[handle_trigger(tid) for tid in available_triggers[:20]]
    )
    actions = [r for r in results if r is not None]
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(request: Request):
    data = await request.json()
    conv_id: str = data.get("conversation_id", str(uuid.uuid4()))
    merchant_id: Optional[str] = data.get("merchant_id")
    customer_id: Optional[str] = data.get("customer_id")
    from_role: str = data.get("from_role", "merchant")
    message: str = data.get("message", "").strip()

    # Update history
    CONVERSATIONS.setdefault(conv_id, []).append({"from": from_role, "msg": message})
    history = CONVERSATIONS[conv_id]

    # STOP / opt-out
    if is_stop(message):
        if merchant_id:
            HOSTILE.add(merchant_id)
        return {"action": "end", "rationale": "Merchant sent STOP/opt-out signal. Respecting preference and suppressing future messages."}

    # Auto-reply loop
    if is_auto_reply(history, message):
        # Try once more with a human-bait question, then back off
        bot_turns = [t for t in history if t["from"] == "bot"]
        if len(bot_turns) >= 2:
            return {"action": "end", "rationale": "Auto-reply loop detected after 2 bot turns. Exiting gracefully."}
        return {
            "action": "send",
            "body": "Lagta hai automated reply hai 🙂 Kya main owner/manager se baat kar sakti hoon? 2-min ka kaam hai.",
            "cta": "open_ended",
            "rationale": "Detected potential auto-reply; trying once to reach a human before backing off.",
        }

    # LLM-based reply with full context
    return await compose_reply_msg(conv_id, merchant_id, customer_id, from_role, message)


@app.post("/v1/teardown")
async def teardown():
    """Wipe all state at end of test per §11 privacy rules."""
    CONTEXTS.clear()
    CONVERSATIONS.clear()
    BOT_SENT.clear()
    SUPPRESSION.clear()
    HOSTILE.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)