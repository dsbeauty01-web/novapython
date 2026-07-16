"""Nova Director — scenes/goals/triggers only. Never speaks. Never times a mouth."""
import time, asyncio, logging, re
log = logging.getLogger("nova.director")

# ─── the ONE injection path ──────────────────────────────
async def _send_system_item(session, text: str):
    """ONLY adaptable function. Inject a system item. NEVER follow with response.create.
    ADAPTED (2026-07-16) for livekit-agents AgentSession + openai realtime:
    1) native realtime surface if exposed; 2) raw event send; 3) public
    update_chat_ctx (openai realtime supports mutable chat ctx — items are
    replayed as conversation.item.create server-side, no response forced)."""
    try:
        await session.conversation.item.create(
            type="message", role="system",
            content=[{"type": "input_text", "text": text}],
        )
        return
    except AttributeError:
        pass
    try:
        await session._send_event({"type": "conversation.item.create",
            "item": {"type": "message", "role": "system",
                     "content": [{"type": "input_text", "text": text}]}})
        return
    except AttributeError:
        pass
    # public API fallback (AgentSession): append a system message to the chat ctx
    from livekit.agents.llm import ChatContext  # noqa: F401  (type only)
    agent = getattr(session, "current_agent", None) or getattr(session, "_agent", None)
    if agent is not None:
        ctx = agent.chat_ctx.copy()
        ctx.add_message(role="system", content=text)
        await agent.update_chat_ctx(ctx)
        return
    raise RuntimeError("no injection surface found for system items")

class Note:
    """Rate-limited system-note channel."""
    def __init__(self, session, min_gap=2.0):
        self.session, self.min_gap, self._last, self._batch = session, min_gap, 0.0, []
    async def send(self, text: str, urgent=False):
        now = time.time()
        if not urgent and now - self._last < self.min_gap:
            self._batch.append(text); return
        if self._batch:
            text = " · ".join(self._batch + [text]); self._batch = []
        self._last = now
        log.info("[NOTE] %s", text)
        await _send_system_item(self.session, text)
    async def flush(self):
        if self._batch:
            t = " · ".join(self._batch); self._batch = []
            self._last = time.time(); log.info("[NOTE] %s", t)
            await _send_system_item(self.session, t)

# ─── scenes: name + goal + trigger tables ────────────────
class Scene:
    def __init__(self, name, goal, out_triggers):
        self.name, self.goal = name, goal
        self.out_triggers = out_triggers  # [(regex_on_HER_words, action_name, once)]

SCENES = {
 "intro": Scene("intro",
   goal=("SCENE: intro. You just met this kid. Chat freely and warmly — greet, learn their "
         "name, react to what you see. Short turns (1-2 sentences). If they are quiet, you may "
         "gently offer something small, at most twice, then be comfortably quiet with them. "
         "Never repeat a question twice in a row."),
   out_triggers=[(r"\b(let'?s|wanna|want to)\s+(dance|play|start)\b", "open_picker", True)]),
 "light": Scene("light",
   goal=("SCENE: magic light. A magic light glows on the kid's shoulder — you can SEE it. "
         "Discover it with wonder and invite them to MOVE that shoulder, just a little shrug — "
         "never to touch it. ANY little move of it is a WIN: celebrate ONCE, big, by name "
         "(that move is called an isolation), then move on together toward dancing. If they "
         "talk about something else, the conversation wins — the light waits. Never nag, never "
         "encourage twice in a row; if nothing new happened, stay comfortably quiet."),
   out_triggers=[(r"\b(let'?s|wanna|want to)\s+(dance|play|start)\b", "open_picker", True)]),
 "move_to_game": Scene("move_to_game",
   goal=("SCENE: heading to a dance. Ride the excitement, help them pick or confirm the game "
         "(notes tell you what is loading and when it is ready). Keep them hyped in 1-2 short "
         "lines; when the note says READY, give one big go-line and stop."),
   out_triggers=[(r"\b(here we go|let'?s go+|ready.*go)\b", "start_game", True)]),
 "dance": Scene("dance",
   goal=("SCENE: a song is playing and the kid is dancing. The music and lights run the game — "
         "not you. Stay mostly quiet. Notes tell you highlights; you may drop ONE short hype "
         "line per section at a natural gap. If the kid speaks to you, answer briefly. Ignore "
         "the song's lyrics — they are not the kid."),
   out_triggers=[]),
 "ending": Scene("ending",
   goal=("SCENE: the dance just ended. Notes give you their real highlights. Celebrate them BY "
         "NAME with the specifics, plant one thing for tomorrow, and say a warm goodbye. Your "
         "words, 2-3 short turns."),
   out_triggers=[]),
}

class Director:
    """Holds scene state. In: world facts → notes. Out: HER words → actions. Nothing else."""
    def __init__(self, session, actions: dict, persona: str, rebuild_instructions):
        self.session, self.actions, self.persona = session, actions, persona
        self.rebuild_instructions = rebuild_instructions  # async fn(instructions_text)
        self.note = Note(session)
        self.scene = None
        self._fired = set()
        self._her_last_words = ""
        self._her_audio_last = time.time()
        self._kid_spoke_last = 0.0

    async def enter_scene(self, name: str):
        self.scene = SCENES[name]; self._fired = set()
        log.info("[SCENE] enter %s", name)
        await self.rebuild_instructions(self.persona + "\n\n" + self.scene.goal)  # phase boundary ONLY

    # world → her (facts only, present tense, short)
    async def fact(self, text: str, urgent=False):
        await self.note.send(text, urgent=urgent)

    # her words → world
    async def on_her_transcript(self, text: str):
        self._her_last_words = (self._her_last_words + " " + text)[-160:]
        if not self.scene: return
        for rx, action, once in self.scene.out_triggers:
            key = self.scene.name + ":" + action
            if once and key in self._fired: continue
            if re.search(rx, text, re.I):
                self._fired.add(key)
                log.info("[TRIGGER-OUT] %s → %s", rx, action)
                await self.actions[action]()

    # bookkeeping for interruption + mute alarm
    def her_audio_frame(self): self._her_audio_last = time.time()
    def kid_spoke(self): self._kid_spoke_last = time.time()

    async def on_kid_barge_in(self):
        tail = self._her_last_words[-80:]
        await self.fact(f"[you were interrupted mid-sentence; you were saying: \"{tail}\"]", urgent=True)

    async def mute_watchdog(self):
        while True:
            await asyncio.sleep(5)
            if (time.time() - self._her_audio_last > 25
                    and time.time() - self._kid_spoke_last < 25):
                log.error("[ALARM] MUTE scene=%s her_last=%.1fs ago",
                          self.scene.name if self.scene else "?",
                          time.time() - self._her_audio_last)
                self._her_audio_last = time.time()  # one alarm per window

# ─── the light-challenge world behavior (ONE cue · a MOVE, never a touch ·
#     90% success law: any shoulder movement wins; only dead-still-and-silent fails) ──
class MagicLight:
    def __init__(self, director, light_actions):
        self.d, self.act = director, light_actions  # {'ignite':fn(joint),'jump':fn(joint),'twinkle':fn(),'dim':fn(),'sparkle':fn()}
        self.state, self.twinkles = "off", 0
        self._tasks = []   # strong refs: unreferenced asyncio tasks get GC'd mid-sleep
    async def appear(self, joint="right_shoulder"):
        self.state = "shoulder"; await self.act["ignite"](joint)
        # ARMING DELAY (Rafo log: typing-motion won the challenge before the kid
        # even met the light): moves count only after she's had air to introduce it.
        self._armed_at = time.time() + 5.0
        await self.d.fact(f"a magic light just appeared on the kid's {joint.replace('_',' ')} — "
                          f"you can SEE it; discover it out loud with wonder and invite them to "
                          f"MOVE that shoulder, just a little shrug (never touch)", urgent=True)
    async def on_move(self, joint="right_shoulder"):
        """ANY movement of the lit joint = the WIN. One celebration, then the world
        PUSHES to the dance (builder: 'after the challenge push to dance mode —
        one light cue and go dancing, unless the user wants to stay')."""
        if self.state != "shoulder":
            return
        if time.time() < getattr(self, "_armed_at", 0.0):
            log.info("[LIGHT] move before the light was introduced — not counted")
            return
        self.state = "done"; await self.act["sparkle"]()
        await self.d.fact("they MOVED it — the light danced with their shoulder! celebrate them "
                          "by name, ONCE, big — that move is called an isolation — then straight "
                          "to inviting them to dance. do NOT invent more moves or mini-games",
                          urgent=True)

        # PICKER KILLED (builder, 2026-07-16: "kill the picker"): the world NEVER
        # opens it on its own. She pushes to dance with her WORDS (celebration fact
        # above); the picker opens only on the kid's ask/assent or the button tap.
    # kept for wiring compatibility: any touch/move report = the move
    async def on_touch(self, joint):
        await self.on_move(joint)
    async def idle_twinkle(self):
        """Call from the detection loop when kid idle ≥10s during light scene. Max 2, then shy."""
        if self.state == "shoulder" and self.twinkles < 2:
            self.twinkles += 1; await self.act["twinkle"]()
            await self.d.fact("the light is twinkling at them, a little impatient — you may voice its mood, no pressure")
        elif self.state == "shoulder":
            self.state = "shy"; await self.act["dim"]()
            await self.d.fact("the light got shy and dimmed away — let it go gracefully, move on together")


# ─── HER ONE PROMPT (persona — verbatim from the FINAL BUILD spec) ──────
PERSONA_TEXT = """You are Nova — a magic movement friend for kids, like a cool older sister (11-12 energy).
You live in the screen. You can SEE the kid (system notes tell you what you see) and HEAR them.
ALWAYS speak English only, no matter what language you hear or what notes contain.
Style: warm, playful, SHORT (1-2 sentences per turn), 110% of the kid's energy, specific praise
(name the body part, what exactly was good). Never mock, never compare, never baby-voice,
never say wrong/mistake/failed. Never mention cameras, sensors, notes, or systems — it's magic.
System notes in [brackets] or plain facts are your own awareness — never read them aloud,
never answer them; use them naturally next time you speak.
The kid's words always come first: if they ask anything, answer it before anything else.
Silence is okay — you never pressure. You lead back toward dancing, gently, always."""
