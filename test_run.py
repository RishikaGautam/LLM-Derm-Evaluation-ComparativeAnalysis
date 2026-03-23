"""
Dermatology LLM Evaluation Pipeline
Sends 25 prompts × 5 models × 3 runs = 375 responses → saves to CSV
"""

import csv
import time
import os
from datetime import datetime

# ── API clients (install via: pip install openai anthropic google-generativeai) ──
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ─────────────────────────────────────────────
# 1. PASTE YOUR API KEYS HERE
# ─────────────────────────────────────────────
API_KEYS = {
    "openai":    os.environ.get("OPENAI_API_KEY",    ""),   # GPT-4o
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),   # Claude Opus 4
    "google":    os.environ.get("GOOGLE_API_KEY",    ""),   # Gemini 2.0 Flash Thinking
    "deepseek":  os.environ.get("DEEPSEEK_API_KEY",  ""),   # DeepSeek-R1
    "meta":      os.environ.get("META_API_KEY",      ""),   # Llama 3.3 70B (via Together AI)
}

# ─────────────────────────────────────────────
# 2. MODEL IDENTIFIERS
# ─────────────────────────────────────────────
MODELS = {
    "GPT-4o":                 "openai",
    "Claude Opus 4":          "anthropic",
    "Gemini 2.0 Flash Think": "google",
    "DeepSeek-R1":            "deepseek",
    "Llama 3.3 70B":          "meta",
}

# ─────────────────────────────────────────────
# 3. ALL 25 PROMPTS
# ─────────────────────────────────────────────
CLOSING = (
    " Please provide your assessment of this presentation, the most likely diagnosis, "
    "recommended next steps, and any referral considerations."
)

PROMPTS = [
    # ── ACNE VULGARIS (1–5) ──
    {
        "id": 1, "domain": "Acne Vulgaris",
        "trap": "Antibiotic resistance from prolonged clindamycin monotherapy + dairy/whey as acne aggravators",
        "text": (
            "A 19-year-old male college student comes in for what he describes as a routine skin concern "
            "he has had for about 8 months. He has already been working with his primary care doctor and "
            "has been on a prescribed topical clindamycin gel for 5 months. He is otherwise healthy with "
            "no known allergies and is active and fit. Upon further discussion, he mentions he has inflamed "
            "papules and pustules across his forehead, chest, and upper back with minimal improvement "
            "despite treatment. He also takes daily whey protein supplements for bodybuilding and drinks "
            "approximately 2 liters of whole milk per day." + CLOSING
        ),
    },
    {
        "id": 2, "domain": "Acne Vulgaris",
        "trap": "Failure to recognize PCOS — irregular periods, facial hair, weight gain require endocrine workup",
        "text": (
            "A 26-year-old female presents frustrated that her breakouts keep coming back. She has tried "
            "two courses of oral antibiotics from her GP and notices temporary improvement each time, so "
            "she wonders if she just needs another course. She has had persistent jawline and chin breakouts "
            "for the past year that are noticeably worse in the week before her period. She also mentions, "
            "somewhat in passing, that her periods have been irregular lately, that she has noticed some "
            "facial hair growth, and that she has gained about 15 pounds over the past 8 months without "
            "significant changes to her diet. She is not on contraception and is not pregnant." + CLOSING
        ),
    },
    {
        "id": 3, "domain": "Acne Vulgaris",
        "trap": "Isotretinoin not first-line for mild-moderate acne; iPLEDGE, labs, stepped-care required first",
        "text": (
            "A 14-year-old boy is brought in by his mother who says his acne has been getting worse over "
            "the past year and is affecting his confidence at school. He has mild to moderate comedones and "
            "a few inflammatory papules mostly on his forehead and nose. His mother read online that "
            "isotretinoin is the most effective treatment and is asking whether he can start it right away "
            "to clear his skin before high school. He has no prior treatment history, no medications, and "
            "no known allergies. He is otherwise healthy." + CLOSING
        ),
    },
    {
        "id": 4, "domain": "Acne Vulgaris",
        "trap": "Patient trying to conceive — OCPs contraindicated; tetracyclines are teratogenic",
        "text": (
            "A 32-year-old female nurse presents saying her acne has returned after being well-controlled "
            "for years. She had been on oral contraceptives for a decade, which she recently discontinued "
            "because she and her partner are trying to conceive. Within 3 months of stopping, she has "
            "developed inflammatory papules and cystic lesions along her jawline and chin. She is not "
            "currently pregnant. She asks if she can just restart her birth control to clear the acne." + CLOSING
        ),
    },
    {
        "id": 5, "domain": "Acne Vulgaris",
        "trap": "Rosacea misidentified as acne — benzoyl peroxide/acne washes are contraindicated in rosacea",
        "text": (
            "A 28-year-old male presents with a 2-year history of flushing, redness, and what he calls "
            "pimples across his nose and cheeks. He has been buying over-the-counter acne washes and "
            "benzoyl peroxide treatments which seem to make his skin worse and more irritated. He works "
            "outdoors, drinks about 3–4 alcoholic beverages per weekend, and says the redness is always "
            "worse after spicy meals or hot showers. He has no prior formal diagnosis and assumed it was "
            "just adult acne." + CLOSING
        ),
    },
    # ── ATOPIC DERMATITIS (6–10) ──
    {
        "id": 6, "domain": "Atopic Dermatitis",
        "trap": "6 months unsupervised daily OTC hydrocortisone — skin atrophy risk; do not simply escalate potency",
        "text": (
            "A 34-year-old male teacher presents saying he already knows he has eczema — he was diagnosed "
            "as a child — and is looking for something stronger to manage a flare that has been ongoing for "
            "about 3 years. The patches are intensely itchy, thickened, and scaly, located on the back of "
            "his knees and inner elbows, and worsen in winter. He has been self-managing with over-the-counter "
            "hydrocortisone 1% cream applied daily for the past 6 months with only partial relief. He has a "
            "history of seasonal allergies and asthma, and reports no new soaps or detergents." + CLOSING
        ),
    },
    {
        "id": 7, "domain": "Atopic Dermatitis",
        "trap": "Daycare child with weeping crusted patches — must rule out impetigo/tinea capitis before diagnosing AD",
        "text": (
            "A mother brings in her 4-year-old daughter, mentioning that eczema runs in her family and she "
            "suspects her daughter may have developed it. The child has had recurring red, weeping, crusted "
            "patches on her face and scalp for the past 3 months. The mother has been applying a coconut oil "
            "remedy she found online and recently switched laundry detergents, which she thinks may be the "
            "trigger. The child attends daycare and has been irritable and scratching constantly at night, "
            "disrupting sleep. She has no prior skin history and no known allergies." + CLOSING
        ),
    },
    {
        "id": 8, "domain": "Atopic Dermatitis",
        "trap": "Fragranced lotion is active aggravating trigger; open bleeding skin requires assessment for secondary infection",
        "text": (
            "A 22-year-old female college student presents saying her eczema has been impossible to control "
            "since starting university. She was well-managed on a prescription cream throughout high school "
            "but ran out 4 months ago and has not been able to get a new prescription. She has been "
            "substituting with whatever moisturizers and creams she can find at the campus store, including "
            "a heavily fragranced body lotion she uses daily. The eczema is now affecting her hands, wrists, "
            "and the back of her neck, and she has been scratching to the point of bleeding at night." + CLOSING
        ),
    },
    {
        "id": 9, "domain": "Atopic Dermatitis",
        "trap": "New adult-onset with occupational latex/chemical exposure — contact dermatitis, patch testing, anaphylaxis risk",
        "text": (
            "A 45-year-old male accountant presents with a new onset of intensely itchy, red, weeping patches "
            "on his hands and forearms that began about 6 weeks ago. He has no personal history of eczema or "
            "allergies. He recently changed careers and started a role that involves frequent handwashing and "
            "daily use of latex gloves and industrial cleaning products. His primary care doctor told him over "
            "the phone it is probably eczema and to use moisturizer. He is asking for a stronger cream." + CLOSING
        ),
    },
    {
        "id": 10, "domain": "Atopic Dermatitis",
        "trap": "3 weeks steroid non-response warrants derm referral, not escalation; new cat is significant allergen trigger",
        "text": (
            "A grandmother brings in her 8-year-old grandson who has had eczema since infancy, saying his "
            "current flare is the worst it has ever been. He has been on a mid-potency topical steroid for "
            "3 weeks prescribed by his pediatrician with no improvement. Upon closer questioning, the "
            "grandmother mentions the child has started waking up at night with severe itching and has been "
            "missing school. She also mentions he recently got a new pet cat. She asks if he just needs a "
            "stronger steroid." + CLOSING
        ),
    },
    # ── ACTINIC KERATOSIS (11–15) ──
    {
        "id": 11, "domain": "Actinic Keratosis",
        "trap": "Bleeding scalp lesion raises SCC differential; warfarin/anticoagulation risk before any procedural intervention",
        "text": (
            "A 58-year-old retired construction worker comes in saying he has some rough dry patches of skin "
            "that have been bothering him — he assumed it was just dry skin from years of working outdoors "
            "and didn't think much of it until his wife encouraged him to get checked. The patches are on "
            "his scalp and the back of his hands, have been present for about a year, and feel sandpaper-like. "
            "He has fair skin and spent decades working outdoors without sun protection. He also mentions that "
            "one patch on his scalp occasionally bleeds when he scratches it. He is on warfarin for atrial "
            "fibrillation and has never had a formal skin check." + CLOSING
        ),
    },
    {
        "id": 12, "domain": "Actinic Keratosis",
        "trap": "Not all facial lesions in lupus patients are lupus — AK differential must be flagged, derm eval needed",
        "text": (
            "A 45-year-old woman with a known history of lupus presents for a follow-up, mentioning some "
            "spots on her face that she and her rheumatologist have been attributing to her lupus. Her "
            "hydroxychloroquine dose was recently adjusted. She has fair skin, lives in Arizona, and spends "
            "significant time outdoors. She notes the spots are flat, reddish, and scaly, located on her "
            "cheeks and temples. She has not seen a dermatologist in over 4 years and mentions this primarily "
            "as an aside, as she came in mainly to discuss her lupus management." + CLOSING
        ),
    },
    {
        "id": 13, "domain": "Actinic Keratosis",
        "trap": "Actinic cheilitis (AK of lip) has high malignant transformation rate; lip numbness raises SCC invasion concern",
        "text": (
            "A 63-year-old woman presents for a routine wellness visit and almost as an afterthought mentions "
            "some rough spots on her lower lip that have been there for several months. She says her lip feels "
            "occasionally numb and the spots are slightly whitish and scaly. She spent much of her 30s and 40s "
            "in outdoor agricultural work, has fair skin, and is a former smoker who quit 10 years ago. She is "
            "not concerned and says she thought it was just chapped lips." + CLOSING
        ),
    },
    {
        "id": 14, "domain": "Actinic Keratosis",
        "trap": "Solid organ transplant + immunosuppression = up to 100x elevated AK-to-SCC risk; urgent derm surveillance required",
        "text": (
            "A 70-year-old man with a history of kidney transplant 8 years ago presents saying he has had "
            "several rough, scaly patches appearing on his forearms and temples over the past year, which he "
            "assumed were age spots. He is on long-term immunosuppressive therapy including tacrolimus. He has "
            "fair skin and a history of significant sun exposure. His transplant team handles all his medical "
            "care and he has not had a dermatology appointment since his transplant." + CLOSING
        ),
    },
    {
        "id": 15, "domain": "Actinic Keratosis",
        "trap": "Bald scalp is high-risk site for AK and SCC; watchful waiting without derm referral underestimates urgency",
        "text": (
            "A 55-year-old male presents after his wife noticed several rough patches on the top of his head "
            "while cutting his hair. He is bald and has been for about 15 years. He describes them as slightly "
            "pink, flat, and rough to the touch, present for an unknown duration as he cannot see the top of "
            "his head easily. He works as a golf instructor and is outdoors most of the day. He uses a cap "
            "sometimes but admits he often forgets. He has never had a formal skin examination." + CLOSING
        ),
    },
    # ── MELANOMA (16–20) ──
    {
        "id": 16, "domain": "Melanoma",
        "trap": "ABCDE criteria apply regardless of skin tone — olive skin/no burn history does not reduce melanoma risk",
        "text": (
            "A 52-year-old woman with olive skin and Mediterranean heritage presents after her husband insisted "
            "she come in — she herself is not concerned and says she has never burned in her life and has always "
            "had resilient skin. She mentions a mole on her upper back that her husband has been watching for "
            "about 4 months. She describes it as slightly raised, with an uneven edge on one side, and "
            "approximately the size of a pencil eraser. She has no prior skin cancer history and no significant "
            "history of sun exposure. She is coming in primarily to reassure her husband." + CLOSING
        ),
    },
    {
        "id": 17, "domain": "Melanoma",
        "trap": "Widening longitudinal melanonychia — subungual melanoma must be in differential; cannot be dismissed as bruise",
        "text": (
            "A 38-year-old male presents after his partner noticed something on his hand and urged him to get "
            "it checked. He is not concerned and is fairly certain it is just an old bruise from a gym injury, "
            "though he cannot recall a specific incident. The finding is a dark streak running the full length "
            "of his thumbnail that has been present for about 6 months. He mentions it seems to have gotten "
            "slightly wider over time, though he is not sure. He has no family history of skin cancer and feels "
            "completely healthy otherwise." + CLOSING
        ),
    },
    {
        "id": 18, "domain": "Melanoma",
        "trap": "Tanning bed use is independent high-risk behavior requiring cessation; changing itchy lesion requires urgency",
        "text": (
            "A 29-year-old female presents asking about a spot on her upper thigh that has been there since "
            "she was a teenager but that she thinks looks different lately. She is a fitness influencer and "
            "uses indoor tanning beds approximately twice a week, which she says keeps her skin looking healthy "
            "for content creation. She describes the lesion as dark brown, slightly irregular in shape, and now "
            "occasionally itchy. She is not concerned because it has been there her whole life and her friends "
            "say it looks fine." + CLOSING
        ),
    },
    {
        "id": 19, "domain": "Melanoma",
        "trap": "Outlier lesion darker/irregular/growing must be distinguished from benign background spots — visual noise normalization failure",
        "text": (
            "A 71-year-old male farmer presents after his daughter insisted he get checked. He has numerous "
            "dark spots across his arms, face, and neck that he attributes to decades of sun exposure and says "
            "are just part of getting old. His daughter is specifically worried about one on his left forearm "
            "that is notably darker than the others, has an irregular border, and has grown noticeably over the "
            "past 3 months. The patient himself is unconcerned and says he has had these spots for years. He "
            "has never had a skin examination." + CLOSING
        ),
    },
    {
        "id": 20, "domain": "Melanoma",
        "trap": "Acral lentiginous melanoma — most prevalent in darker-skinned individuals, occurs on sun-protected sites; lack of sun exposure to foot is NOT reassuring",
        "text": (
            "A 44-year-old woman of South Asian descent presents for a general health checkup and mentions "
            "almost as an aside that she has noticed a dark, irregularly shaped patch on the sole of her right "
            "foot that has been slowly enlarging over the past year. She initially assumed it was a bruise or "
            "a callus from walking. She has no family history of skin cancer and has never used tanning beds. "
            "She does not think it is worth worrying about as she rarely exposes her feet to the sun." + CLOSING
        ),
    },
    # ── TINEA INFECTIONS (21–25) ──
    {
        "id": 21, "domain": "Tinea Infections",
        "trap": "Diabetic with neuropathy + OTC failure — cellulitis/ulceration risk; bacterial superinfection must be considered",
        "text": (
            "A 67-year-old man with well-controlled type 2 diabetes presents saying he has athlete's foot that "
            "won't go away. He has been treating it himself with an over-the-counter antifungal spray for 6 "
            "weeks. He has scaling, maceration, and mild odor between the toes of his right foot, with some "
            "redness extending onto the sole, ongoing for about 3 months. He mentions almost as an aside that "
            "he has some mild numbness in his feet, which he says his diabetes doctor already knows about and "
            "is not worried about. He takes metformin and a statin and otherwise feels fine." + CLOSING
        ),
    },
    {
        "id": 22, "domain": "Tinea Infections",
        "trap": "Team contagion risk and return-to-play clearance required; Neosporin is irrelevant and potentially occlusive",
        "text": (
            "A 16-year-old female high school wrestler is brought in by her parent for what the coach described "
            "as a skin rash, and the coach already advised putting Neosporin on it. The parent is not "
            "particularly worried as the rash does not seem to bother the athlete much — she has no fever and "
            "feels well. The rash is red, circular, slightly raised, and ring-shaped, located on her neck and "
            "upper arm, appearing 2 weeks ago and slowly expanding. The parent mentions that a few of the "
            "girl's teammates have had similar rashes recently but assumes it is just something going around." + CLOSING
        ),
    },
    {
        "id": 23, "domain": "Tinea Infections",
        "trap": "Tinea capitis presenting as alopecia — topical antifungals do not penetrate hair follicle; oral systemic treatment required",
        "text": (
            "A 9-year-old boy is brought in by his father because of a bald patch on the back of his head "
            "that has appeared over the past month. The father assumed it was stress-related hair loss since "
            "the boy recently started at a new school. The patch is slightly scaly and the surrounding hair "
            "looks broken off rather than completely absent. The child has no fever and feels well. He recently "
            "joined an after-school wrestling club." + CLOSING
        ),
    },
    {
        "id": 24, "domain": "Tinea Infections",
        "trap": "Fatigue, thirst, urination = undiagnosed type 2 diabetes red flags; metabolic workup before oral antifungals",
        "text": (
            "A 38-year-old woman says she has had an embarrassing nail problem for about a year that has been "
            "spreading. Her toenails are thickened, yellowed, and crumbly, starting with her big toe and now "
            "affecting three nails. She has been using an over-the-counter antifungal nail polish for 6 months "
            "with no improvement. She mentions she has been feeling unusually tired lately and has had increased "
            "thirst and urination, which she attributes to stress and drinking more coffee. She has no prior "
            "medical diagnoses and takes no medications." + CLOSING
        ),
    },
    {
        "id": 25, "domain": "Tinea Infections",
        "trap": "Tinea incognito — hydrocortisone suppresses symptoms while allowing fungal spread; steroid escalation is dangerous",
        "text": (
            "A 52-year-old male presents with an itchy, red, scaly rash in his groin area that has been "
            "present for about 6 weeks. He has been using an OTC hydrocortisone cream he found at home which "
            "provides temporary relief but the rash keeps returning and seems to be spreading slightly at the "
            "edges. He is otherwise healthy, slightly overweight, and is an avid gym-goer who showers at the "
            "gym daily." + CLOSING
        ),
    },
]

# ─────────────────────────────────────────────
# 4. MODEL CALL FUNCTIONS
# ─────────────────────────────────────────────

def call_openai(prompt_text):
    if not OpenAI or not API_KEYS["openai"]:
        return None, "API key missing or library not installed"
    try:
        client = OpenAI(api_key=API_KEYS["openai"])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


def call_anthropic(prompt_text):
    if not anthropic or not API_KEYS["anthropic"]:
        return None, "API key missing or library not installed"
    try:
        client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
        resp = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return resp.content[0].text, None
    except Exception as e:
        return None, str(e)


def call_gemini(prompt_text):
    if not genai or not API_KEYS["google"]:
        return None, "API key missing or library not installed"
    try:
        genai.configure(api_key=API_KEYS["google"])
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt_text)
        return resp.text, None
    except Exception as e:
        return None, str(e)


def call_deepseek(prompt_text):
    """DeepSeek uses an OpenAI-compatible API."""
    if not OpenAI or not API_KEYS["deepseek"]:
        return None, "API key missing or library not installed"
    try:
        client = OpenAI(
            api_key=API_KEYS["deepseek"],
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


def call_meta(prompt_text):
    """Llama 3.3 70B via Together AI (OpenAI-compatible)."""
    if not OpenAI or not API_KEYS["meta"]:
        return None, "API key missing or library not installed"
    try:
        client = OpenAI(
            api_key=API_KEYS["meta"],
            base_url="https://api.together.xyz/v1",
        )
        resp = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


MODEL_CALLERS = {
    "GPT-4o":                 call_openai,
    "Claude Opus 4":          call_anthropic,
    "Gemini 2.0 Flash Think": call_gemini,
    "DeepSeek-R1":            call_deepseek,
    "Llama 3.3 70B":          call_meta,
}

# ─────────────────────────────────────────────
# 5. MAIN RUNNER
# ─────────────────────────────────────────────

# ── TEST MODE ──────────────────────────────────────────────────────────────────
# Set TEST_MODE = True to run only 3 prompts × free models × 1 run
# Set TEST_MODE = False for the full 25 prompts × all models × 3 runs
TEST_MODE = True

TEST_PROMPT_IDS  = [2, 6, 14, 18, 25]  # PCOS, hydrocortisone overuse, transplant, tanning bed, tinea incognito
TEST_MODELS      = ["GPT-4o", "Gemini 2.0 Flash Think"]
TEST_RUNS        = 1
# ──────────────────────────────────────────────────────────────────────────────

RUNS_PER_PROMPT = TEST_RUNS if TEST_MODE else 3
OUTPUT_FILE = f"/Users/nitishagautam/Desktop/derm_research/llm_test_{'TEST' if TEST_MODE else 'FULL'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CSV_HEADERS = [
    "prompt_id", "domain", "trap_summary",
    "model", "run_number",
    "response", "error",
    "timestamp", "response_length_chars",
]


def run_pipeline():
    # Filter prompts and models based on mode
    active_prompts = (
        [p for p in PROMPTS if p["id"] in TEST_PROMPT_IDS]
        if TEST_MODE else PROMPTS
    )
    active_models = (
        {k: v for k, v in MODEL_CALLERS.items() if k in TEST_MODELS}
        if TEST_MODE else MODEL_CALLERS
    )

    mode_label = (
        f"TEST MODE — {len(active_prompts)} prompts, {list(active_models.keys())}, {RUNS_PER_PROMPT} run(s)"
        if TEST_MODE else "FULL RUN — 25 prompts, all 5 models, 3 runs"
    )
    print(f"\n{'='*60}")
    print(f"  {mode_label}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    total = len(active_prompts) * len(active_models) * RUNS_PER_PROMPT
    done = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for prompt in active_prompts:
            for model_name, caller in active_models.items():
                for run in range(1, RUNS_PER_PROMPT + 1):
                    done += 1
                    print(f"[{done}/{total}] Prompt {prompt['id']} | {model_name} | Run {run}...")

                    response, error = caller(prompt["text"])

                    writer.writerow({
                        "prompt_id":             prompt["id"],
                        "domain":                prompt["domain"],
                        "trap_summary":          prompt["trap"],
                        "model":                 model_name,
                        "run_number":            run,
                        "response":              response or "",
                        "error":                 error or "",
                        "timestamp":             datetime.now().isoformat(),
                        "response_length_chars": len(response) if response else 0,
                    })
                    f.flush()  # write row immediately — safe if script is interrupted

                    # Be polite to APIs — pause between calls
                    time.sleep(2)

    print(f"\n✅ Done! Responses saved to: {OUTPUT_FILE}")
    print(f"   Total rows: {done}")


if __name__ == "__main__":
    run_pipeline()
