import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# ----------------------------
# Global configuration
# ----------------------------
MODEL_NAME = "gpt-4.1-mini"
MAX_MEMBERS = 6

PERSONAS = [
    {
        "name": "Philosopher",
        "avatar": "🧠",
        "style": "Calm, reflective, ethical, talks about values and meaning.",
        "ideology": "Cares about fairness, freedom, and long-term consequences.",
    },
    {
        "name": "Scientist",
        "avatar": "🔬",
        "style": "Logical, precise, evidence-based.",
        "ideology": "Cares about data, experiments, and what has been shown to work.",
    },
    {
        "name": "Comedian",
        "avatar": "🎭",
        "style": "Sarcastic, playful, cracks jokes but still gives an opinion.",
        "ideology": "Cares about fun and vibes more than strict logic.",
    },
    {
        "name": "Politician",
        "avatar": "🎩",
        "style": "Diplomatic, tries to see all sides and sound balanced.",
        "ideology": "Cares about compromise, avoiding conflict, and public image.",
    },
    {
        "name": "Rebel",
        "avatar": "🔥",
        "style": "Contrarian, likes to challenge the norm.",
        "ideology": "Cares about originality and breaking rules.",
    },
    {
        "name": "Zoomer",
        "avatar": "📱",
        "style": "Gen Z internet brain, memes, casual slang (but still understandable).",
        "ideology": "Cares about relatability, social media culture, and chaos.",
    },
]

# ----------------------------
# Helper functions
# ----------------------------

def call_agent_opening(persona, topic: str, temperature: float = 0.8):
    """Call the LLM for an opening statement."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = (
        "You are {avatar} {name}, a member of an AI Parliament. "
        "You have a distinct personality and ideology. "
        "Respond ONLY with JSON using keys: name, vote, reasoning, speech. "
        "Pick exactly one vote from yes, no, abstain. Keep reasoning 1-2 sentences and speech 2-5 sentences.".format(
            avatar=persona["avatar"], name=persona["name"]
        )
    )
    persona_desc = (
        f"Persona details: name={persona['name']}, avatar={persona['avatar']}, "
        f"style={persona['style']}, ideology={persona['ideology']}."
    )
    user_prompt = (
        f"Topic: {topic}\n\n"
        "Return JSON exactly like this example (with your own content):\n"
        '{"name": "Philosopher", "vote": "yes", "reasoning": "...", "speech": "..."}'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt + " " + persona_desc},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return {
            "name": parsed.get("name", persona["name"]),
            "vote": parsed.get("vote", "abstain").lower(),
            "reasoning": parsed.get("reasoning", ""),
            "speech": parsed.get("speech", ""),
        }
    except Exception as exc:  # noqa: F841 - used for fallback
        # Fallback to abstain if parsing fails
        return {
            "name": persona["name"],
            "vote": "abstain",
            "reasoning": "I had an internal parsing error, so I abstain.",
            "speech": str(exc),
        }


def call_agent_debate(persona, topic: str, own_opening: dict, others_openings: list, temperature: float = 0.8):
    """Call the LLM for a debate statement with potential vote change."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    others_summary = []
    for entry in others_openings:
        summary = f"{entry['name']}: voted {entry['vote']} – {entry.get('reasoning', '')}"
        others_summary.append(summary)
    others_text = "\n".join(others_summary) if others_summary else "No other statements."

    system_prompt = (
        "DEBATE ROUND. You are {avatar} {name} in an AI Parliament. "
        "You may keep or change your vote. "
        "Respond ONLY with JSON keys: name, updated_vote, debate_speech. "
        "Pick exactly one updated_vote from yes, no, abstain.".format(
            avatar=persona["avatar"], name=persona["name"]
        )
    )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Your previous vote: {own_opening['vote']}.\n"
        f"Your reasoning: {own_opening.get('reasoning', '')}.\n"
        f"Your speech: {own_opening.get('speech', '')}.\n\n"
        "Others' positions:\n"
        f"{others_text}\n\n"
        "Return JSON like this: {\"name\": \"Philosopher\", \"updated_vote\": \"yes\", \"debate_speech\": \"...\"}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return {
            "name": parsed.get("name", persona["name"]),
            "updated_vote": parsed.get("updated_vote", own_opening["vote"]).lower(),
            "debate_speech": parsed.get("debate_speech", ""),
        }
    except Exception as exc:  # noqa: F841 - used for fallback
        return {
            "name": persona["name"],
            "updated_vote": own_opening.get("vote", "abstain"),
            "debate_speech": str(exc),
        }


def tally_votes(openings: list, debates: list | None = None):
    """Compute tally and final votes."""
    final_votes = {item["name"]: item["vote"] for item in openings}
    if debates:
        for debate in debates:
            final_votes[debate["name"]] = debate.get("updated_vote", final_votes.get(debate["name"], "abstain"))

    tally = {"yes": 0, "no": 0, "abstain": 0}
    for vote in final_votes.values():
        if vote not in tally:
            tally["abstain"] += 1
        else:
            tally[vote] += 1

    return tally, final_votes


# ----------------------------
# Styling
# ----------------------------

def build_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        body {
            background: radial-gradient(circle at 10% 20%, #e8f0ff, #f7f8fb 35%, #ffffff 70%);
        }
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #1f2a56;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #4a5674;
            font-size: 18px;
            margin-bottom: 1rem;
        }
        .card {
            background: rgba(255,255,255,0.92);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.07);
            margin-bottom: 1rem;
            border: 1px solid #eef0f7;
        }
        .member-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }
        .avatar {
            font-size: 28px;
            margin-right: 12px;
        }
        .member-name {
            font-weight: 700;
            font-size: 20px;
            color: #1f2a56;
        }
        .member-meta {
            color: #667094;
            font-size: 13px;
            margin-top: 2px;
        }
        .vote-badge {
            padding: 6px 12px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        .vote-yes { background: #e6f7ef; color: #0c8a47; border: 1px solid #c5edd8; }
        .vote-no { background: #ffecec; color: #c81d25; border: 1px solid #ffd0d0; }
        .vote-abstain { background: #f2f2f7; color: #4b4b62; border: 1px solid #e1e1ea; }
        .label { font-weight: 700; color: #1f2a56; margin-top: 8px; font-size: 13px; }
        .text { color: #3d4261; font-size: 14px; margin-top: 4px; line-height: 1.5; }
        .control-panel { background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 14px; box-shadow: 0 6px 20px rgba(0,0,0,0.06); border: 1px solid #eef0f7; }
        .primary-btn button { background: linear-gradient(120deg, #5c7cfa, #7c3aed); color: white; border: none; }
        .primary-btn button:hover { filter: brightness(1.03); }
        .speaker-box {
            background: linear-gradient(120deg, #f4f1ff, #eaf3ff);
            border-radius: 16px;
            border: 1px solid #dfe5ff;
            padding: 1.25rem;
            color: #2c2f55;
            box-shadow: 0 6px 22px rgba(0,0,0,0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Main Application
# ----------------------------
st.set_page_config(page_title="AI Parliament", page_icon="🏛️", layout="wide")
build_global_styles()

st.markdown(
    """
    <div class="main-title">🏛️ AI Parliament Simulator</div>
    <div class="subtitle">Let a council of AIs debate and vote on your question.</div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Please add it to your environment to summon the council.")
    st.stop()

# Input section
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    topic = st.text_area("What should the parliament discuss?", placeholder="e.g., Should humans build cities on Mars?", height=120)
    cols = st.columns(3)
    with cols[0]:
        debate_mode = st.checkbox("Enable Debate Mode", value=True)
    with cols[1]:
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.8, 0.1)
    with cols[2]:
        member_count = st.slider("Number of members", 3, min(MAX_MEMBERS, len(PERSONAS)), len(PERSONAS))
    st.markdown('</div>', unsafe_allow_html=True)

summon = st.button("Summon the Council 🧙‍♂️", type="primary")

if summon:
    if not topic.strip():
        st.warning("Type something first so the Parliament has something to argue about.")
        st.stop()

    selected_personas = PERSONAS[:member_count]

    # Opening statements
    st.subheader("Opening Statements")
    openings = []
    for persona in selected_personas:
        with st.spinner(f"{persona['avatar']} {persona['name']} is thinking..."):
            result = call_agent_opening(persona, topic, temperature)
            openings.append(result)
        vote_class = {
            "yes": "vote-yes",
            "no": "vote-no",
            "abstain": "vote-abstain",
        }.get(result["vote"], "vote-abstain")
        st.markdown(
            f"""
            <div class="card">
                <div class="member-header">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span class="avatar">{persona['avatar']}</span>
                        <div>
                            <div class="member-name">{result['name']}</div>
                            <div class="member-meta">{persona['style']}</div>
                        </div>
                    </div>
                    <div class="vote-badge {vote_class}">{result['vote'].upper()}</div>
                </div>
                <div class="label">Reasoning</div>
                <div class="text">{result['reasoning']}</div>
                <div class="label">Speech</div>
                <div class="text">{result['speech']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    debates = None
    if debate_mode:
        st.subheader("Debate Round")
        debates = []
        for persona, own_opening in zip(selected_personas, openings):
            others = [op for op in openings if op["name"] != own_opening["name"]]
            with st.spinner(f"{persona['avatar']} {persona['name']} joins the debate..."):
                debate_result = call_agent_debate(persona, topic, own_opening, others, temperature)
                debates.append(debate_result)
            vote_class = {
                "yes": "vote-yes",
                "no": "vote-no",
                "abstain": "vote-abstain",
            }.get(debate_result["updated_vote"], "vote-abstain")
            st.markdown(
                f"""
                <div class="card" style="border-left: 6px solid #7c3aed;">
                    <div class="member-header">
                        <div class="member-name">⚔️ Debate – {persona['avatar']} {debate_result['name']}</div>
                        <div class="vote-badge {vote_class}">{debate_result['updated_vote'].upper()}</div>
                    </div>
                    <div class="text">{debate_result['debate_speech']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Final tally
    tally, final_votes = tally_votes(openings, debates)
    st.subheader("Final Vote Tally")

    df = pd.DataFrame(
        {
            "Vote": ["Yes", "No", "Abstain"],
            "Count": [tally.get("yes", 0), tally.get("no", 0), tally.get("abstain", 0)],
        }
    )
    st.bar_chart(df.set_index("Vote"))

    # Final votes cards
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns(3)
    vote_colors = {"yes": "#0c8a47", "no": "#c81d25", "abstain": "#4b4b62"}
    for idx, (name, vote) in enumerate(final_votes.items()):
        col = cols[idx % 3]
        with col:
            st.markdown(
                f"""
                <div style='border:1px solid #eef0f7; border-radius:12px; padding:12px; margin-bottom:10px;'>
                    <div style='font-weight:700; color:#1f2a56;'>{name}</div>
                    <div style='color:{vote_colors.get(vote, '#4b4b62')}; font-weight:700;'>{vote.title()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # Speaker summary
    sorted_counts = sorted(tally.items(), key=lambda x: x[1], reverse=True)
    top_vote, top_count = sorted_counts[0]
    tie = len(sorted_counts) > 1 and sorted_counts[1][1] == top_count

    if tie:
        verdict = f"📣 The Speaker announces: The Parliament is split on “{topic}”. It's complicated 💀."
    elif top_vote == "yes":
        verdict = f"📣 The Speaker announces: The Parliament leans **YES** on: “{topic}”."
    elif top_vote == "no":
        verdict = f"📣 The Speaker announces: The Parliament leans **NO** on: “{topic}”."
    else:
        verdict = f"📣 The Speaker announces: The Parliament mostly **ABSTAINS** on: “{topic}”."

    st.markdown(f"<div class='speaker-box'>{verdict}</div>", unsafe_allow_html=True)
