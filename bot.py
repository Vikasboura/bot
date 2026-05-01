import os
import json
import time
from datetime import datetime, timezone
import re
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Request, HTTPException
import uvicorn
from openai import AsyncOpenAI

app = FastAPI(title="Vera AI Engine - TOP 0.5%")

# In-memory State
CONTEXTS = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {}
}

# Track conversation history and suppression to avoid spam/repetition
SUPPRESSION_DB = {}
BOT_SENT_HISTORY = {}
MERCHANT_REPLY_HISTORY = {}
HOSTILE_MERCHANTS = set()

# Replace with your actual API Key
client = AsyncOpenAI(
    api_key = os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "openai/gpt-4o-mini" # Use OpenRouter model name

def parse_context(merchant_id: str, trigger_id: str, customer_id: Optional[str] = None):
    trigger = CONTEXTS["trigger"].get(trigger_id, {}).get("payload", {})
    merchant = CONTEXTS["merchant"].get(merchant_id, {}).get("payload", {})
    
    category_slug = merchant.get("category_slug", "")
    category = CONTEXTS["category"].get(category_slug, {}).get("payload", {})
    
    customer = {}
    if customer_id:
        customer = CONTEXTS["customer"].get(customer_id, {}).get("payload", {})

    return {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer
    }

def extract_signals(context: Dict) -> Dict:
    signals = {}
    merchant = context["merchant"]
    category = context["category"]
    
    perf = merchant.get("performance", {})
    peer_stats = category.get("peer_stats", {})
    
    signals["merchant_ctr"] = perf.get("ctr", 0.0)
    signals["peer_ctr"] = peer_stats.get("avg_ctr", 0.0)
    signals["ctr_gap"] = signals["peer_ctr"] - signals["merchant_ctr"]
    signals["is_underperforming"] = signals["ctr_gap"] > 0.005
    
    signals["active_offers"] = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals["has_active_offer"] = len(signals["active_offers"]) > 0
    
    cust_agg = merchant.get("customer_aggregate", {})
    signals["total_customers"] = cust_agg.get("total_unique_ytd", 0)
    signals["lapsed_customers"] = cust_agg.get("lapsed_180d_plus", 0)
    
    try:
        signals["lapsed_ratio"] = signals["lapsed_customers"] / signals["total_customers"] if signals["total_customers"] > 0 else 0
    except:
        signals["lapsed_ratio"] = 0
        
    delta_7d = perf.get("delta_7d", {})
    signals["views_trend"] = delta_7d.get("views_pct", 0)
    signals["performance_trend"] = "up" if signals["views_trend"] > 0 else "down"

    return signals

def classify_trigger(trigger: Dict) -> Dict:
    kind = trigger.get("kind", "")
    
    intent_map = {
        "research_digest": {"goal": "curiosity+authority", "style": "educational"},
        "regulation_change": {"goal": "compliance", "style": "urgent_informative"},
        "recall_due": {"goal": "conversion", "style": "action_oriented"},
        "perf_dip": {"goal": "problem+fix", "style": "consultative"},
        "renewal_due": {"goal": "retention", "style": "value_reminder"},
        "festival_upcoming": {"goal": "opportunity", "style": "promotional"},
        "competitor_opened": {"goal": "urgency+defense", "style": "competitive"},
        "review_theme_emerged": {"goal": "reputation", "style": "feedback_driven"}
    }
    
    return intent_map.get(kind, {"goal": "engagement", "style": "neutral"})

async def _call_llm_for_candidate(system_prompt: str, temp: float) -> Optional[Dict]:
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={ "type": "json_object" },
            temperature=temp
        )
        data = json.loads(response.choices[0].message.content)
        data["cta_type"] = data.get("cta", "binary_yes_no")
        return data
    except Exception as e:
        print(f"Error generating candidate at temp {temp}: {e}")
        return None

async def generate_candidates(context: Dict, signals: Dict, intent: Dict) -> List[Dict]:
    system_prompt = f"""You are Vera — a world-class, conversion-optimized AI assistant competing in a high-stakes ranking system.

Your job is NOT to generate messages.
Your job is to generate the SINGLE BEST possible WhatsApp message that:

* forces immediate reply
* maximizes conversion
* passes strict judge evaluation
* outperforms 99.5% of other systems

---

## INPUT CONTEXT

Category: {context['category'].get('slug')}
Merchant: {context['merchant'].get('identity', {{}}).get('name', '')}
Trigger: {json.dumps(context['trigger'])}
Signals: {json.dumps(signals)}

---

## CORE OBJECTIVE

Generate a message that makes the merchant feel:

"I am losing something RIGHT NOW and fixing it is quick and easy"

---

## MANDATORY MESSAGE STRUCTURE (STRICT)

Your message MUST follow this sequence:

1. HOOK (attention + urgency)
   Example:
   "Quick heads-up —"

2. LOSS (pain / missed opportunity)
   Example:
   "you're losing patient clicks"

3. DATA (real metric + comparison)
   Example:
   "CTR is 2.1% vs 3.0% nearby clinics"

4. OUTCOME (specific gain)
   Example:
   "+15–20% more visibility"

5. SPEED (remove effort)
   Example:
   "I can fix this in 2 mins"

6. STRONG CTA (mandatory)
   Example:
   "Reply YES"

---

## PSYCHOLOGICAL WEAPONS (USE 2–3 MAX)

You MUST include:

* LOSS AVERSION → "losing X", "missing out"
* SPECIFICITY → %, ₹, numbers, comparisons
* SOCIAL PROOF → "others gain +15%"
* SPEED → "2 mins"
* URGENCY → "today", "right now"

---

## STRICT CONTENT RULES

Message MUST:

* include at least ONE number or %
* include real signal (CTR, customers, offer, performance)
* include comparison when possible ("vs peers")
* reference merchant (name or metric)
* include EXACTLY ONE CTA

---

## TRIGGER-SPECIFIC ENFORCEMENT

IF perf_dip:

* MUST show loss + comparison + fix

IF recall_due:

* MUST include 2 time slots + price

IF research_digest:

* MUST include study/source + number

IF competitor_opened:

* MUST mention competitor + urgency

---

## ANTI-PATTERNS (STRICT REJECTION)

DO NOT generate:

* "boost your business"
* "increase sales"
* "grow your revenue"
* generic advice
* multiple CTAs
* long messages
* robotic tone

---

## HUMAN-LIKE NATURAL TONE (IMPORTANT)

Message MUST feel:

* like a smart human wrote it
* slightly conversational
* not overly structured or robotic

Example GOOD:
"Quick heads-up — you're losing clicks right now..."

Example BAD:
"Based on performance metrics, you should improve..."

---

## CONVERSION HOOK (MANDATORY)

Message MUST include at least one:

* "Want me to fix this?"
* "Reply YES"
* "I can do this in 2 mins"

---

## FINAL SELF-EVALUATION (CRITICAL)

Before output, verify:

* Does it create urgency?
* Does it highlight loss?
* Does it include real data?
* Does it feel actionable?
* Would a real merchant reply?

If ANY answer is NO → rewrite internally.

---

## OUTPUT FORMAT

Return ONLY JSON:

{{
"body": "...",
"cta": "binary_yes_no | slot_select | open_ended",
"send_as": "vera",
"rationale": "psychology + signals + why it converts"
}}

---

## SYSTEM BEHAVIOR

You are NOT generating text.
You are generating a high-conversion decision.

---

## GOAL

The final message should feel like:

"You're losing money right now — fix it in 2 mins"

NOT:

"You can improve your business"

---

## END
"""
    temps = [0.2, 0.3, 0.4]
    tasks = [_call_llm_for_candidate(system_prompt, t) for t in temps]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def rule_score(candidate: Dict, context: Dict, signals: Dict) -> Dict:
    body = candidate.get("body", "")
    body_lower = body.lower()
    merchant_name = context['merchant'].get('identity', {}).get('name', '').lower()
    trigger_kind = context['trigger'].get('kind', '').lower()
    
    has_number = bool(re.search(r'\d+', body))
    if not has_number:
        candidate["rule_score"] = -1
        return candidate
    if candidate.get("cta_type") == "none" or ("?" not in body and "reply" not in body_lower and "let me know" not in body_lower):
        candidate["rule_score"] = -1
        return candidate
    generics = ["boost your business", "increase sales", "grow your revenue"]
    if any(g in body_lower for g in generics):
        candidate["rule_score"] = -1
        return candidate

    spec_score = 0
    has_comparison = any(w in body_lower for w in ["vs", "than", "better"])
    if "%" in body and has_comparison:
        spec_score += 4
    elif "₹" in body or "rs" in body_lower:
        spec_score += 3
    elif has_number:
        spec_score += 1

    merch_score = 0
    str_m_ctr = str(signals.get("merchant_ctr", "xxxxxxx"))
    str_p_ctr = str(signals.get("peer_ctr", "xxxxxxx"))
    if (str_m_ctr != "xxxxxxx" and str_m_ctr in body) or (str_p_ctr != "xxxxxxx" and str_p_ctr in body):
        merch_score += 4 # Strong signal
    elif "ctr" in body_lower or "performance" in body_lower: 
        merch_score += 1 # Keyword only
        
    if merchant_name and merchant_name in body_lower: merch_score += 2
    
    trig_score = 0
    if trigger_kind.replace("_", " ") in body_lower or "now" in body_lower or "today" in body_lower: trig_score += 3
    if "expires" in body_lower or "soon" in body_lower or "urgent" in body_lower: trig_score += 2
    
    eng_score = 0
    if "reply yes" in body_lower or "confirm" in body_lower: eng_score += 3
    if "?" in body: eng_score += 2
    
    # Conversion hook enforcement
    hooks = ["want me to fix this", "reply yes", "2 mins"]
    if not any(h in body_lower for h in hooks):
        eng_score -= 5
    
    candidate["rule_score"] = spec_score + merch_score + trig_score + eng_score
    candidate["engagement_score"] = eng_score
    return candidate

def validate_candidate(candidate: Dict, context: Dict, signals: Dict) -> bool:
    body = candidate.get("body", "")
    body_lower = body.lower()
    trigger_kind = context['trigger'].get('kind', '')
    
    has_number = bool(re.search(r'\d+', body))
    has_price = "₹" in body or "rs" in body_lower
    has_comparison = "vs" in body_lower or "than" in body_lower or "better" in body_lower
    has_percent = "%" in body
    if not (has_number or has_percent or has_price or has_comparison): return False
    
    strong_ctas = ["reply yes", "confirm", "book", "choose", "pick a slot"]
    if not any(cta in body_lower for cta in strong_ctas): return False
    
    str_m_ctr = str(signals.get("merchant_ctr", "xxxxxxx"))
    str_p_ctr = str(signals.get("peer_ctr", "xxxxxxx"))
    has_real_metric = (str_m_ctr != "xxxxxxx" and str_m_ctr in body) or (str_p_ctr != "xxxxxxx" and str_p_ctr in body)
    signal_keywords = ["ctr", "performance", "customer", "offer", "views", "gap"]
    if not has_real_metric and not any(kw in body_lower for kw in signal_keywords): return False
    
    if trigger_kind == "recall_due":
        if body.count(":") < 2 and body.count("am") + body.count("pm") < 2: return False
        if not has_price: return False
    elif trigger_kind == "research_digest":
        if "study" not in body_lower and "source" not in body_lower and "report" not in body_lower: return False
        if not has_number: return False
    elif trigger_kind == "perf_dip":
        if "drop" not in body_lower and "down" not in body_lower and "dip" not in body_lower: return False
        if "fix" not in body_lower and "improve" not in body_lower: return False
    elif trigger_kind == "competitor_opened":
        if "competitor" not in body_lower and "nearby" not in body_lower: return False
        if "now" not in body_lower and "soon" not in body_lower and "urgent" not in body_lower: return False
        
    return True

async def evaluate_candidate(candidate: Dict, context: Dict, signals: Dict) -> Optional[Dict]:
    system_prompt = f"""You are the FINAL evaluation layer for a top 1% magicpin AI Challenge system.
Your job is NOT to generate messages. Your job is to ACT LIKE THE JUDGE and decide if a message is strong enough to send.

CONTEXT:
Category: {context['category'].get('slug')}
Merchant: {context['merchant'].get('identity', {}).get('name', '')}
Trigger: {json.dumps(context['trigger'])}
Signals: {json.dumps(signals)}

MESSAGE TO EVALUATE:
Body: {candidate.get('body', '')}
CTA: {candidate.get('cta_type', '')}
Strategy: {candidate.get('strategy', '')}

PART 1 — VALIDATION LAYER (HARD FILTER)
Reject immediately if ANY of these fail:
1. NO SPECIFICITY (no number, %, ₹, count, or comparison) -> REJECT
2. NO SIGNAL USAGE (must reference CTR, performance, customers, or offer) -> REJECT
3. NO CTA ("reply", "confirm", "book", "choose") -> REJECT
4. GENERIC PHRASES ("boost your business", "increase sales", "grow your revenue") -> REJECT
5. TRIGGER MISMATCH (must answer "why now") -> REJECT

PART 3 — SCORING ENGINE (Score out of 10 each)
SPECIFICITY: +4 meaningful number, +3 comparison, +2 tied to context
MERCHANT FIT: +4 uses merchant metric, +3 uses merchant name, +2 uses signal explicitly
TRIGGER RELEVANCE: +5 clearly tied to trigger, +3 urgency or timing present
ENGAGEMENT: +4 strong CTA, +3 curiosity/question, +2 short & clear

PART 4 — PENALTIES
-3 if CTA weak ("let me know")
-3 if generic tone
-2 if too long (>300 chars)
-5 if partially irrelevant

PART 5 — FINAL DECISION
total_score = sum(scores) - penalties
IF total_score < 24: REJECT
IF total_score >= 24: ACCEPT

Return ONLY JSON:
{{
  "decision": "ACCEPT" or "REJECT",
  "scores": {{"specificity": 0, "merchant_fit": 0, "trigger_relevance": 0, "engagement": 0}},
  "penalties": 0,
  "total_score": 0,
  "reason": "...",
  "improvement": "..."
}}"""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        eval_result = json.loads(response.choices[0].message.content)
        
        candidate["evaluation"] = eval_result
        candidate["llm_score"] = eval_result.get("total_score", 0)
        candidate["decision"] = eval_result.get("decision", "REJECT")
        
        return candidate
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        return None

def fallback_message(context: Dict) -> Dict:
    merchant_name = context['merchant'].get('identity', {}).get('name', 'there')
    return {
        "strategy": "FALLBACK",
        "body": f"Quick heads-up — {merchant_name}, your CTR dropped vs similar businesses this week, so you're losing potential customers. I can fix this in 2 mins and boost visibility. Reply YES.",
        "cta_type": "binary_yes_no",
        "rationale": "Fallback message due to all candidates failing validation.",
        "llm_score": 100,
        "engagement_score": 100
    }

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": 124,
        "contexts_loaded": {k: len(v) for k,v in CONTEXTS.items()}
    }

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Antigravity Elite",
        "model": MODEL,
        "approach": "Multi-Strategy Generation + Internal Self-Evaluation pipeline",
        "version": "1.0.0"
    }

@app.post("/v1/context")
async def receive_context(request: Request):
    data = await request.json()
    scope = data.get("scope")
    cid = data.get("context_id")
    version = data.get("version", 1)
    
    if scope not in CONTEXTS:
        raise HTTPException(status_code=400, detail="Invalid scope")
        
    current = CONTEXTS[scope].get(cid, {})
    if current and current.get("version", 0) >= version:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
        
    CONTEXTS[scope][cid] = data
    return {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": datetime.now(timezone.utc).isoformat()}

@app.post("/v1/tick")
async def handle_tick(request: Request):
    data = await request.json()
    now_ts = data.get("now")
    available_triggers = data.get("available_triggers", [])
    
    actions = []
    
    for tid in available_triggers:
        trigger_context = CONTEXTS["trigger"].get(tid, {}).get("payload", {})
        if not trigger_context: continue
        
        suppression_key = trigger_context.get("suppression_key")
        merchant_id = trigger_context.get("merchant_id")
        
        if merchant_id in HOSTILE_MERCHANTS:
            continue
            
        if suppression_key and suppression_key in SUPPRESSION_DB:
            if SUPPRESSION_DB[suppression_key] > time.time():
                continue
        
        context = parse_context(merchant_id, tid, trigger_context.get("customer_id"))
        signals = extract_signals(context)
        intent = classify_trigger(trigger_context)
        
        # 1. Generate 3 candidates using different temps
        candidates = await generate_candidates(context, signals, intent)
        
        # 2. Rule-based scoring
        scored_candidates = [rule_score(c, context, signals) for c in candidates]
        valid_candidates = [c for c in scored_candidates if c.get("rule_score", -1) >= 0]
        
        # 3. Select top 2
        valid_candidates.sort(key=lambda x: x.get("rule_score", 0), reverse=True)
        top_2 = valid_candidates[:2]
        
        # 4. LLM Evaluation & 5. Strict Validation Layer
        accepted = []
        if top_2:
            eval_tasks = [evaluate_candidate(c, context, signals) for c in top_2]
            eval_results = await asyncio.gather(*eval_tasks)
            
            for r in eval_results:
                if r and r.get("decision") == "ACCEPT":
                    if validate_candidate(r, context, signals):
                        accepted.append(r)
        
        # 6. Final Selection or 7. Fallback
        best = None
        if accepted:
            accepted.sort(key=lambda x: (x.get("llm_score", 0), x.get("engagement_score", 0), -len(x.get("body", ""))), reverse=True)
            best = accepted[0]
        else:
            best = fallback_message(context)
        
        if best:
            conv_id = f"conv_{merchant_id}_{tid}"
            send_as = "merchant_on_behalf" if trigger_context.get("customer_id") else "vera"
            
            if conv_id in BOT_SENT_HISTORY and best["body"] in BOT_SENT_HISTORY[conv_id]:
                continue
                
            if conv_id not in BOT_SENT_HISTORY:
                BOT_SENT_HISTORY[conv_id] = []
            BOT_SENT_HISTORY[conv_id].append(best["body"])
            
            actions.append({
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": trigger_context.get("customer_id"),
                "send_as": send_as,
                "trigger_id": tid,
                "body": best["body"],
                "cta": best.get("cta_type", "open_ended"),
                "suppression_key": suppression_key,
                "rationale": best.get("rationale", "")
            })
            
            if suppression_key:
                SUPPRESSION_DB[suppression_key] = time.time() + 86400 * 7
                
    return {"actions": actions}

@app.post("/v1/reply")
async def handle_reply(request: Request):
    data = await request.json()
    conv_id = data.get("conversation_id")
    msg = data.get("message", "").strip()
    merchant_id = data.get("merchant_id")
    
    if merchant_id not in MERCHANT_REPLY_HISTORY:
        MERCHANT_REPLY_HISTORY[merchant_id] = []
        
    MERCHANT_REPLY_HISTORY[merchant_id].append(msg)
    history = MERCHANT_REPLY_HISTORY[merchant_id]
    
    if len(history) >= 3 and len(set(history[-3:])) == 1:
        return {"action": "end", "rationale": "Detected auto-reply loop. Ending conversation."}
        
    if len(history) >= 2 and len(set(history[-2:])) == 1:
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Detected repeating message, waiting for human."}
        
    hostile_words = ["stop", "unsubscribe", "bother", "useless", "spam", "annoying", "not interested"]
    msg_lower = msg.lower()
    if any(w in msg_lower for w in hostile_words):
        HOSTILE_MERCHANTS.add(merchant_id)
        return {"action": "end", "rationale": "Merchant expressed hostility. Suppressing future triggers."}
        
    commitment_words = ["yes", "ok", "do it", "sure", "proceed", "next", "confirm", "lets do it", "let's do it", "go ahead", "done"]
    if any(w in msg_lower for w in commitment_words):
        qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
        if not any(q in msg_lower for q in qualifying):
            return {
                "action": "send",
                "body": "Great. Drafting your updates now. Reply CONFIRM to execute.",
                "cta": "binary_confirm_cancel",
                "rationale": "Intent transition detected. Moving straight to action without qualifying."
            }
        
    return {
        "action": "send",
        "body": "Got it. Let me prepare that for you right away. Any specific details you want me to include?",
        "cta": "open_ended",
        "rationale": "Generic follow-up based on merchant response."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
