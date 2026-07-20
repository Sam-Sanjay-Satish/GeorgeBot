#!/usr/bin/env python3
"""
Extended-thinking orchestrator for GeorgeBot.

A second query path, parallel to the single-shot `GeorgeBot.ask()`/stream and
to default mode's "simple" verify-then-answer path (see `chatbot.py`'s
`answer_verified_stream`). `GeorgeBot.rewrite_and_route(..., mode="default")`
(the "planner") decides per-query whether a question is simple or needs one
of the three plans below; `api.py` calls `ExtendedThinking.run(...)` with that
precomputed route/plan/confidence only for the not-simple case — no manual
toggle. `run()` no longer classifies the query itself (that used to be this
module's own `classify_mode`, now folded into the planner call).

This is **not** one generic loop — each of the three plans below has its own
structure, tool access, and termination logic (see the design doc):

  - scattered_info  — fuzzy, evaluator-driven multi-round retrieval (the only
                      mode that iterates). Decompose → multi-phrasing fan-out →
                      dedup → coverage evaluator → re-query gaps → stop.
  - course_planning — fixed enumeration → deterministic solver. NO fuzzy
                      evaluator. Graph enumerates eligible courses, Banner
                      batches live sections, `scheduler.solve_schedule` returns
                      conflict-free combinations.
  - situational     — triage-first, tightly-scoped fixed checklist over policy
                      docs + offices. Banner/solver excluded. Calm, procedural,
                      no case assessment (enforced in the answer system prompt).

All LLM stages are MiniMax-M3 via the existing `GeorgeBot._call_llm` plumbing,
differentiated only by prompt. Retrieval reuses `bot.vector_retrieve` /
`bot.graph_retrieve` / `bot.gs`; context assembly and source formatting reuse
`bot._assemble_context` / `bot.format_sources`; the answer streams through
`bot.answer_stream(..., system_prompt=<mode prompt>)`.

`run()` yields event dicts the API layer maps straight to SSE frames:
    {"type": "mode",    "data": "<plan label, e.g. 'Course Planner'>"}  # first event, always
    {"type": "status",  "data": "<phase text>"}
    {"type": "clarify", "data": "<question to the user>"}   # terminal — no answer
    {"type": "planning_state", "data": {<see below>}}  # course_planning only
    {"type": "sources", "data": [<source dict>, ...]}
    {"type": "token",   "data": "<answer piece>"}
    {"type": "done"}
    {"type": "error",   "data": "<message>"}

Slot-gathering that needs input (course_planning / situational) emits a single
`clarify` event and stops; the user's reply arrives as an ordinary next turn in
`history`, and re-entry finds the slots there (rebuild-from-history — no
server-side session state, no conversation id).

**Course-planning slot persistence (client-echoed, still no server state).**
Rebuild-from-history alone was not enough for `course_planning`: re-deriving
every slot from the whole conversation on every turn meant a slot that had
already been *resolved* could be collapsed back to its vaguer original form.
The confirmed failure was a program-disambiguation loop — turn 1 "CS major" →
ambiguous → "which one?"; turn 2 the user picks an exact option → the whole
history is re-read → "computer science" again → the identical question,
forever, with no way out through normal conversation.

The fix keeps the module stateless: `_run_planning` accepts a `planning_state`
dict from the caller, emits an updated one as a `planning_state` event, and the
**client echoes the latest one back** on the next turn (exactly how `history`
already works). It carries only STABLE inputs:

    {"program": {"pid","title","credential"} | null,
     "term_season": str|null, "term_year": int|null,
     "completed_courses": [str], "completed_known": bool,
     "constraints": {"include_courses": [], "exclude_courses": [],
                     "soft_preferences": []},
     "non_course_resolved": {"<COURSE>": {"satisfied": bool, "reason": str|null}},
     "pending_clarify": {"kind": str, "options": [...]} | null}

Volatile outputs (eligible-course list, Banner sections/seats, computed
schedules) are deliberately NOT persisted — they're recomputed fresh every turn
from the stable inputs, because seat counts change constantly (see the Banner
TTL design in CLAUDE.md).

It degrades safely in both directions: a missing/None/malformed `planning_state`
falls back to the original full re-derivation from history, and every field is
re-validated on the way in (it arrives from the client, so it is not trusted).
"""

from __future__ import annotations

import json
import re
import sys
import time

from banner import banner_retrieve
from chatbot import DEFAULT_PLAN, VALID_PLANS
from scheduler import solve_schedule

# scattered_info fuzzy loop: initial round + up to (cap-1) gap-filling rounds.
SCATTERED_MAX_ITERATIONS = 3
SCATTERED_QUERY_N = 4          # chunks pulled per sub-query (matches N_CONTEXT)

# course_planning: cap how many eligible courses we throw at Banner in one batch
# (a huge remaining-requirements list would be a slow, pointless fan-out).
PLANNING_MAX_ELIGIBLE = 18

# How many candidate programs to list back in a disambiguation question. Only a
# display cap — every candidate is kept in `pending_clarify.options`, so a
# student who names one that wasn't shown still resolves.
PLANNING_MAX_PROGRAM_OPTIONS = 8

# Short, user-facing labels for each plan — emitted as a "mode" event so the
# frontend can tag which pipeline produced an answer.
PLAN_LABELS: dict[str, str] = {
    "scattered_info": "Deep Research",
    "course_planning": "Course Planner",
    "situational": "Guidance",
}


# ---------------------------------------------------------------------------
# Situation sub-types (situational mode) — each maps to scoped policy search
# seeds and the real office to direct the student to. Data-driven so more types
# can be added without touching the pipeline logic (design §5.2, launch set).
# ---------------------------------------------------------------------------

SITUATION_TYPES: dict[str, dict] = {
    "academic_integrity": {
        "label": "academic integrity allegation",
        "queries": [
            "academic integrity policy plagiarism cheating allegation process",
            "how to respond to an academic integrity allegation student",
            "academic integrity appeal rights meeting outcome",
        ],
        "office": "the Office of the Ombudsperson (uvicombudsperson.ca) and UVSS "
                  "Student Advocacy — both give free, confidential help responding "
                  "to an allegation",
    },
    "financial_hold": {
        "label": "financial hold on your account",
        "queries": [
            "financial hold account tuition fees outstanding balance registration",
            "how to remove a hold on my UVic account pay balance",
            "tuition payment deadline late fees consequences",
        ],
        "office": "Accounting Services / Student Financial Services (tuition and "
                  "account holds)",
    },
    "medical_or_compassionate_withdrawal": {
        "label": "medical or compassionate withdrawal",
        "queries": [
            "medical withdrawal request compassionate withdrawal process deadline",
            "withdraw from courses for medical reasons documentation required",
            "retroactive withdrawal appeal medical grounds tuition refund",
        ],
        "office": "an Academic Adviser in your faculty and the Centre for "
                  "Accessible Learning (CAL), plus Student Wellness for support",
    },
    "academic_standing": {
        "label": "academic standing (probation / required to withdraw)",
        "queries": [
            "academic probation required to withdraw academic standing GPA",
            "academic concession reinstatement after required to withdraw",
            "how to improve academic standing conditions probation",
        ],
        "office": "an Academic Adviser in your faculty (they handle standing, "
                  "probation conditions, and reinstatement)",
    },
    "grade_appeal_or_review": {
        "label": "grade appeal or review",
        "queries": [
            "grade review appeal process deadline final grade dispute",
            "how to request a formal review of an assigned grade",
            "grade appeal steps instructor chair dean deadline",
        ],
        "office": "your course instructor first, then the department Chair; the "
                  "Ombudsperson can advise on the formal review process",
    },
    "admission_or_registration_issue": {
        "label": "admission or registration issue",
        "queries": [
            "registration error course add drop deadline permission override",
            "admission condition requirement deferral registration hold",
            "how to resolve a registration problem waitlist prerequisite error",
        ],
        "office": "Undergraduate Records / the Registrar's office, and an "
                  "Academic Adviser in your faculty",
    },
    "other_policy": {
        "label": "your situation",
        "queries": [
            "student policy procedure rights process deadline",
            "who to contact for help with a student academic issue",
        ],
        "office": "an Academic Adviser in your faculty and the Office of the "
                  "Ombudsperson (free, confidential, independent)",
    },
}


# ---------------------------------------------------------------------------
# Mode-specific answer system prompts
# ---------------------------------------------------------------------------

def _scattered_system_prompt(base: str) -> str:
    return base + (
        "\n\nThis question is broad and was researched across multiple searches. "
        "Make sure you address each distinct part of what was asked that you have "
        "evidence for, organize the answer clearly, and don't pad it with parts "
        "you have no support for — just answer what you can, well."
    )


def _planning_system_prompt(base: str) -> str:
    return base + (
        "\n\nYou are helping a student plan their courses. The reference material "
        "includes computed schedule options (conflict-free combinations of live "
        "sections) and live Banner availability, or — when no schedule could be "
        "computed — the program's actual remaining requirement groups. Only name "
        "requirement group labels, courses, or program structure that literally "
        "appear in the reference material; if it doesn't cover something (e.g. "
        "why a course isn't eligible yet), say that plainly instead of guessing "
        "at program structure. When you present options:\n"
        "- Show 2-3 concrete schedule options with course codes, section codes "
        "(e.g. A01), and meeting days/times; name the term.\n"
        "- These are LECTURE sections — remind the student to add any required "
        "lab/tutorial section, which isn't included in the combination.\n"
        "- Present seat counts as current-as-of-now (they change), and flag "
        "full/waitlist-only sections.\n"
        "- Flag any listed non-course requirements (year standing, GPA, "
        "permission) since those can't be verified from a course list.\n"
        "- If availability couldn't be checked for a course, say so plainly "
        "rather than omitting it silently.\n"
        "- If no conflict-free schedule was found, say that and explain the "
        "conflict rather than inventing one."
    )


def _situational_system_prompt(base: str, situation: dict, deadline: str | None,
                               actions_taken: str | None) -> str:
    deadline_line = (
        f"\n- IMPORTANT: the student mentioned a deadline: {deadline}. Lead with "
        f"it — state it prominently near the top of your answer."
        if deadline else ""
    )
    actions_line = (
        f"\n- The student has already: {actions_taken}. Take that into account; "
        f"don't tell them to do something they've done."
        if actions_taken else ""
    )
    return base + (
        f"\n\nSITUATIONAL MODE — the student is dealing with {situation['label']}. "
        "This is a procedural/policy situation, and your job is narrow and "
        "important:\n"
        "- Tone: calm, plain, and non-alarmist. Do not catastrophize.\n"
        "- Provide ACCURATE procedural information only: what the policy/process "
        "says, what the concrete next steps are, and the exact deadlines and "
        "contacts. Ground procedural facts in the reference material; do not "
        "invent a policy detail or a deadline.\n"
        f"- Direct the student to real human support: {situation['office']}. Name "
        "it explicitly — you are not a substitute for it.\n"
        "- Do NOT assess the student's individual case: do not predict outcomes, "
        "do not say whether they're likely to be found responsible or to succeed, "
        "and do not reassure them that 'it'll be fine'. Stick to what the process "
        "is, not how it will turn out for them.\n"
        "- If you cannot find the specific applicable policy in the reference "
        "material, say so plainly and point them to the office above rather than "
        "guessing at the policy content."
        + deadline_line + actions_line
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ExtendedThinking:
    def __init__(self, bot) -> None:
        self.bot = bot

    # -- LLM JSON helper (mirrors rewrite_and_route's defensive parse) --------

    def _json_llm(self, prompt: str, thinking: str = "disabled",
                  max_tokens: int = 2000) -> dict | None:
        """Call MiniMax-M3, strip fences, parse JSON. None on any failure."""
        try:
            out = self.bot._call_llm([{"role": "user", "content": prompt}],
                                     max_tokens=max_tokens, thinking=thinking)
            if not out:
                return None
            out = out.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(out)
        except Exception as e:
            print(f"  [thinking._json_llm] parse/LLM failure: {e}", file=sys.stderr)
            return None

    def _history_text(self, history: list[dict]) -> str:
        if not history:
            return ""
        from chatbot import MAX_HISTORY_TURNS
        lines = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('content', '')}")
        return "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    # -- Public entry ---------------------------------------------------------

    def run(self, question: str, history: list[dict], audience: str,
            route: dict, plan: str, confidence: float,
            planning_state: dict | None = None):
        """Dispatch into one of the three plans. Yields event dicts (see module
        docstring). `route`/`plan`/`confidence` come from
        `GeorgeBot.rewrite_and_route(..., mode="default")` — the planner
        already classified the query and applied the confidence downgrade, so
        this no longer classifies anything itself (that used to be this
        module's own `classify_mode`).

        `planning_state` is the client-echoed course-planning state from the
        previous turn (see the module docstring). It is optional and untrusted:
        None/malformed means "derive everything from history", i.e. exactly the
        pre-persistence behavior. Ignored by the other two plans."""
        log: dict = {"question": question, "audience": audience,
                     "t_start": time.monotonic()}
        try:
            # Defensive guard in case an invalid plan value ever reaches here
            # (e.g. a future caller bypassing the planner) — mirrors the
            # planner's own fallback.
            if plan not in VALID_PLANS:
                plan = DEFAULT_PLAN

            log.update(mode_classified=plan, router_confidence=confidence)
            yield {"type": "mode", "data": PLAN_LABELS[plan]}

            if plan == "course_planning":
                yield from self._run_planning(question, history, audience, log,
                                              planning_state)
            elif plan == "situational":
                yield from self._run_situational(question, history, audience, log)
            else:
                yield from self._run_scattered(question, history, audience, log, route)
        except Exception as e:  # never let the extended path hard-fail the request
            print(f"  [thinking.run] error: {e}", file=sys.stderr)
            yield {"type": "error", "data": str(e)}
        finally:
            self._emit_log(log)

    def _emit_log(self, log: dict) -> None:
        log["elapsed_s"] = round(time.monotonic() - log.pop("t_start", time.monotonic()), 2)
        print(f"[thinking] {json.dumps(log, default=str)}", file=sys.stderr)

    # -- Shared answer tail ---------------------------------------------------

    def _stream_answer(self, question: str, context: str, history: list[dict],
                       sources: list[dict], system_prompt: str):
        """Emit sources, then stream the mode-specific answer, then done."""
        yield {"type": "sources", "data": sources}
        for piece in self.bot.answer_stream(question, context, history,
                                            system_prompt=system_prompt):
            yield {"type": "token", "data": piece}
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Mode: scattered_info (fuzzy, evaluator-driven)
    # ------------------------------------------------------------------

    def _run_scattered(self, question: str, history: list[dict], audience: str,
                       log: dict, route: dict):
        yield {"type": "status", "data": "Breaking the question into parts…"}

        # `route` (course codes, program, topic families, department) is the
        # planner's own rewrite/route output — computed once by the caller,
        # not re-derived here.

        # Planner: decompose into info-needs, each with 1+ (multi-phrased) queries.
        plan = self._json_llm(
            f"{self._history_text(history)}"
            "Break this UVic question into its distinct information needs for a "
            "retrieval system. For EACH need, give a short description and 1-3 "
            "search queries. When the need is abstract/policy/experiential (where "
            "the right document might use different words than the question), give "
            "2-3 differently-worded queries; for a concrete factual need, one is "
            "fine.\n\n"
            f"Question: {question}\n\n"
            'Return ONLY JSON: {"items":[{"id":"i1","description":"...",'
            '"queries":["...","..."]}]}',
            thinking="disabled", max_tokens=1500,
        ) or {}
        items = plan.get("items") or [{"id": "i1", "description": question,
                                       "queries": [route["search_query"]]}]
        # Normalize + track issued queries so retries rephrase, never repeat.
        for n, it in enumerate(items, 1):
            it.setdefault("id", f"i{n}")
            it["status"] = "not_covered"
            it["queries"] = [q for q in (it.get("queries") or []) if q][:3] or [question]

        # Graph arm (deterministic, one bulk fetch).
        graph_facts: dict = {}
        if route["course_codes"] or route["program_query"]:
            yield {"type": "status", "data": "Looking up course facts…"}
            graph_facts = self.bot.graph_retrieve(
                route["course_codes"], route["program_query"],
                route["wants_outline"], route["completed_courses"])

        seen_chunk_ids: set[str] = set()
        chunks: list[dict] = []
        issued: set[str] = set()

        def _run_queries(queries: list[str]) -> int:
            added = 0
            for q in queries:
                key = q.strip().lower()
                if not key or key in issued:
                    continue
                issued.add(key)
                for ch in self.bot.vector_retrieve(
                        q, audience=audience, n=SCATTERED_QUERY_N,
                        topic_families=route["topic_families"],
                        department=route["department"]):
                    cid = ch.get("chunk_id")
                    if cid and cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        chunks.append(ch)
                        added += 1
            return added

        termination = "evaluator_driven"
        iterations = 0
        for iteration in range(SCATTERED_MAX_ITERATIONS):
            iterations = iteration + 1
            yield {"type": "status",
                   "data": ("Searching…" if iteration == 0 else "Filling in the gaps…")}
            pending = [it for it in items if it["status"] != "covered"]
            queries = [q for it in pending for q in it["queries"]]
            added = _run_queries(queries)

            # Diminishing-returns backstop: a round that adds no new chunks stops.
            if iteration > 0 and added == 0:
                termination = "diminishing_returns"
                break
            if iteration == SCATTERED_MAX_ITERATIONS - 1:
                termination = "hard_cap_forced"
                break

            # Evaluator — the one place we turn MiniMax reasoning on. MiniMax-M3
            # only accepts "adaptive" or "disabled" for thinking.type (there is
            # no "enabled"); "adaptive" lets the model reason when the coverage
            # judgment is non-trivial. See chatbot.py's provider notes.
            evidence = "\n".join(
                f"- [{i+1}] {c['metadata'].get('title', '')} "
                f"(source={c['metadata'].get('source', '')})"
                for i, c in enumerate(chunks)) or "(nothing retrieved yet)"
            checklist = "\n".join(f'- {it["id"]}: {it["description"]}' for it in items)
            verdict = self._json_llm(
                "You are judging whether retrieved evidence covers each part of a "
                "question. For each checklist item say covered / partial / "
                "not_covered, and for anything not fully covered give 1-2 NEW, "
                "differently-worded search queries (do not repeat earlier "
                "phrasings).\n\n"
                f"Question: {question}\n\nChecklist:\n{checklist}\n\n"
                f"Evidence gathered (titles):\n{evidence}\n\n"
                f"Queries already tried: {sorted(issued)}\n\n"
                'Return ONLY JSON: {"items":[{"id":"i1","status":"covered|partial|'
                'not_covered","next_queries":["..."]}],"stop":false}',
                thinking="adaptive", max_tokens=1500,
            ) or {}

            by_id = {v.get("id"): v for v in (verdict.get("items") or [])}
            for it in items:
                v = by_id.get(it["id"])
                if not v:
                    continue
                it["status"] = v.get("status", it["status"])
                nq = [q for q in (v.get("next_queries") or []) if q][:2]
                if nq:
                    it["queries"] = nq
            if verdict.get("stop") or all(it["status"] == "covered" for it in items):
                termination = "evaluator_driven"
                break

        log.update(iteration_depth=iterations, termination_reason=termination,
                   n_chunks=len(chunks),
                   checklist_coverage={it["id"]: it["status"] for it in items})

        yield {"type": "status", "data": "Writing the answer…"}
        context, _ = self.bot._assemble_context(chunks, graph_facts, {}, {})
        sources = self.bot.format_sources(chunks, graph_facts, {}, {})
        system_prompt = _scattered_system_prompt(self.bot.SYSTEM_PROMPT)
        yield from self._stream_answer(question, context, history, sources, system_prompt)

    # ------------------------------------------------------------------
    # Mode: course_planning (fixed enumeration -> solver; no fuzzy evaluator)
    # ------------------------------------------------------------------

    def _planning_slots(self, question: str, history: list[dict],
                        state: dict | None = None) -> dict:
        """Extract course-planning slots. Returns a *delta*: every field is
        nullable, and null means "the extraction says nothing about this",
        which `_merge_planning_slots` reads as "keep what's already there".

        Two framings:
          - `state is None` — full derivation from the whole conversation (the
            original, pre-persistence behavior; the baseline it merges into is
            empty, so the result is identical to what this returned before).
          - `state` given — DELTA extraction: the already-established slots are
            shown to the model and it is asked to report only what the LATEST
            message states or changes. This is what stops a resolved slot being
            re-derived from scratch — and collapsed back to a vaguer value —
            on every single turn.
        """
        if state is None:
            scope = ("Extract course-planning slots from the whole conversation "
                     "for a UVic student. Use null for anything never stated.")
            known = ""
        else:
            prog = state["program"]
            prog_line = (f"{prog['title']} ({prog['credential']})" if prog
                         else "(not resolved yet)")
            completed = ", ".join(state["completed_courses"]) or "(none recorded)"
            known = (
                "ALREADY ESTABLISHED for this student (from earlier turns — "
                "treat as settled):\n"
                f"  program: {prog_line}\n"
                f"  term: {state['term_season'] or '?'} {state['term_year'] or '?'}\n"
                f"  completed courses: {completed} "
                f"(known={state['completed_known']})\n"
                f"  constraints: {json.dumps(state['constraints'])}\n\n"
            )
            scope = ("Report ONLY what the student's LATEST message states or "
                     "changes about these slots. Use null for every slot the "
                     "latest message doesn't speak to — it's already recorded "
                     "above and must not be restated or re-guessed.")
        data = self._json_llm(
            f"{self._history_text(history)}"
            f"Today's date is {time.strftime('%Y-%m-%d')}.\n"
            f"{known}{scope} Return ONLY JSON with:\n"
            '  "program": the student\'s program/major as stated (e.g. "computer '
            'science", "mechanical engineering"). If the assistant just offered a '
            "list of programs to choose from and the student picked one, return "
            "that option's full text VERBATIM. null if not stated here.\n"
            '  "program_changed": true ONLY if the latest message explicitly '
            "corrects or switches an already-established program (e.g. "
            '"actually I\'m in Honours, not the Major"); false otherwise.\n'
            '  "completed_courses": the student\'s FULL list of completed course '
            'codes as of now (normalized like "CSC110"), including any already '
            "recorded above, or null if the latest message says nothing about "
            "completed courses.\n"
            '  "completed_known": true if the student has indicated what they have '
            "taken (or explicitly said none/first-year), else null.\n"
            '  "term_season": "spring"|"summer"|"fall"|null for the term they want '
            "to plan (resolve relative phrases using today's date).\n"
            '  "term_year": 4-digit year or null.\n'
            '  "include_courses": array of course codes they specifically want '
            "included, or null if not mentioned.\n"
            '  "exclude_courses": array of course codes they want to avoid, or '
            "null if not mentioned.\n"
            '  "soft_preferences": array of {"type","value"} using ONLY these '
            'types: no_time_before (HHMM), no_time_after (HHMM), no_days (array of '
            'day codes Mo/Tu/We/Th/Fr), delivery ("online"|"in_person"), '
            'max_courses (int). Parse phrases like "no morning classes" -> '
            '{"type":"no_time_before","value":"1000"}, "no Friday classes" -> '
            '{"type":"no_days","value":["Fr"]}, "5 courses" -> {"type":'
            '"max_courses","value":5}. Drop anything that doesn\'t fit these '
            "types. null if not mentioned.\n\n"
            f"Latest message: {question}",
            thinking="disabled", max_tokens=1200,
        ) or {}
        return {
            "program": (data.get("program") or "").strip() or None,
            "program_changed": bool(data.get("program_changed")),
            "completed_courses": (_norm_codes(data.get("completed_courses"))
                                  if data.get("completed_courses") is not None else None),
            "completed_known": bool(data.get("completed_known")),
            "term_season": (data.get("term_season") or "").strip().lower() or None,
            "term_year": _coerce_year(data.get("term_year")),
            "include_courses": (_norm_codes(data.get("include_courses"))
                                if data.get("include_courses") is not None else None),
            "exclude_courses": (_norm_codes(data.get("exclude_courses"))
                                if data.get("exclude_courses") is not None else None),
            "soft_preferences": (data.get("soft_preferences")
                                 if isinstance(data.get("soft_preferences"), list) else None),
        }

    @staticmethod
    def _merge_planning_slots(state: dict, ex: dict) -> dict:
        """Fold an extraction delta into the persisted state, in place.

        Merge policy: explicitly-stated new info wins (so corrections work);
        silence about an already-resolved slot leaves it exactly as it was
        (so nothing gets re-derived and degraded). `program` is deliberately
        NOT merged here — it needs graph resolution, handled in
        `_resolve_program`.
        """
        if ex.get("term_season") in ("spring", "summer", "fall"):
            state["term_season"] = ex["term_season"]
        if ex.get("term_year"):
            state["term_year"] = ex["term_year"]
        if ex.get("completed_courses") is not None:
            codes = ex["completed_courses"]
            # An empty list only overwrites when the student actually said
            # "none / just starting" — otherwise a null-ish extraction would
            # silently wipe a list they gave us three turns ago.
            if codes or ex.get("completed_known"):
                state["completed_courses"] = codes
            if codes:
                state["completed_known"] = True
        if ex.get("completed_known"):
            state["completed_known"] = True
        cons = state["constraints"]
        for key in ("include_courses", "exclude_courses"):
            if ex.get(key) is not None:
                cons[key] = ex[key]
        if ex.get("soft_preferences") is not None:
            cons["soft_preferences"] = [p for p in ex["soft_preferences"]
                                        if isinstance(p, dict) and p.get("type")]
        return state

    def _resolve_non_course_requirements(
        self, blocked: dict[str, list[dict]], history: list[dict],
        question: str = "",
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Resolve non-course requirements (year standing, declared program,
        GPA, permission, AWR, units-from-list, ...) that `prereq_satisfied`
        can't evaluate itself, by inferring from the conversation whether the
        student has said something indicating they meet each one.

        `blocked` — {course_code: [unknown_requirement, ...]} for candidate
        courses whose only remaining gap is non-course (course prereqs already
        satisfied).

        `question` is the CURRENT turn, which `history` does not contain. It
        must be included: when the previous turn asked "do you meet these?",
        the student's answer *is* the current turn, and reading only `history`
        made that answer invisible for one full round — so the identical
        question got asked again immediately after being answered.

        Returns:
          resolved   — course codes where every non-course requirement was
                      inferred satisfied from history -> eligible.
          unresolved — distinct requirement descriptions never addressed in
                      the conversation -> the caller should ask the user.
          notes      — {course_code: reason} for courses where history
                      indicates a requirement is NOT met -> excluded, but
                      surfaced with an explanation rather than silently
                      dropped.
        """
        by_key: dict[tuple, str] = {}
        for reqs in blocked.values():
            for r in reqs:
                desc = r.get("description") or r.get("type") or "requirement"
                by_key[(r.get("type"), desc)] = desc
        if not by_key:
            return list(blocked.keys()), [], {}

        req_list = list(by_key.values())
        verdict = self._json_llm(
            f"{self._history_text(history)}"
            "A UVic student is asking about course eligibility. For EACH of the "
            "following program/course requirements, decide from the "
            "conversation whether the student has said something indicating "
            "they meet it, said something indicating they do NOT meet it, or "
            "never addressed it at all. Only count an explicit statement — "
            "don't guess or assume. A requirement phrased with \"OR\" is a "
            "single requirement satisfied by meeting ANY ONE of its listed "
            "alternatives — mark it satisfied if the student's statements "
            "cover at least one alternative, even if the others were never "
            "addressed.\n\n"
            + "\n".join(f'- "{r}"' for r in req_list) +
            f"\n\nThe student's latest message (the most likely place an answer "
            f"to these appears): {question}\n\n"
            'Return ONLY JSON: {"results":[{"requirement":"<exact text '
            'above>","status":"satisfied|not_satisfied|unknown"}]}',
            thinking="disabled", max_tokens=1000,
        ) or {}
        status_by_desc = {v.get("requirement"): v.get("status")
                          for v in (verdict.get("results") or [])}

        resolved: list[str] = []
        unresolved: set[str] = set()
        notes: dict[str, str] = {}
        for code, reqs in blocked.items():
            descs = [r.get("description") or r.get("type") or "requirement" for r in reqs]
            statuses = [status_by_desc.get(d, "unknown") for d in descs]
            if all(s == "satisfied" for s in statuses):
                resolved.append(code)
            elif any(s == "not_satisfied" for s in statuses):
                failed = [d for d, s in zip(descs, statuses) if s == "not_satisfied"]
                notes[code] = "requires " + "; ".join(failed) + ", which you don't meet"
            else:
                for d, s in zip(descs, statuses):
                    if s != "satisfied":
                        unresolved.add(d)
        return resolved, sorted(unresolved), notes

    def _resolve_program(self, text: str | None, question: str, state: dict,
                         log: dict) -> str | None:
        """Resolve the student's program to a pid, storing it on `state`.

        Returns a clarify question to ask, or None when `state["program"]` is
        now resolved (or there was nothing to resolve).

        **This is the loop-breaker.** Once `state["program"]` holds a pid, the
        fuzzy `search_programs` path is never entered again for the rest of the
        conversation unless the student explicitly corrects their program — so
        a vague phrase like "CS major" can no longer be re-derived from history
        and collapse an already-picked program back into an ambiguous match.
        """
        pending = state.get("pending_clarify")
        reply = text or question

        # (a) The previous turn asked "which of these programs?" — match the
        # reply against the options WE offered, deterministically. No fuzzy
        # re-search: the options are already exact program records, and
        # re-searching the reply text is precisely what used to loop.
        if pending and pending.get("kind") == "program_disambiguation":
            picked = _match_program_option(reply, pending.get("options") or [])
            if picked:
                state["program"] = picked
                state["pending_clarify"] = None
                log["program_resolved_via"] = "disambiguation_reply"
                return None

        # (b) Nothing new to resolve from -> keep whatever is pinned (the
        # caller only passes `text` when the program is unpinned or the student
        # explicitly corrected it, so this is the common steady-state path and
        # the one that keeps `search_programs` out of the loop entirely).
        if not text:
            if state["program"]:
                log["program_resolved_via"] = "persisted"
            return None

        # (c) Fresh resolution (first time, or an explicit correction). Drop any
        # existing pin first: until the new text resolves, the state must not
        # claim the program the student just corrected away from.
        state["program"] = None
        # Mirror chatbot.py's _program_facts: rank by
        # surplus tokens and auto-pick a clear winner instead of asking on any
        # raw multi-match — search_programs only guarantees token-subset
        # containment, so e.g. "computer science honours" legitimately matches
        # the standalone Honours program AND both combined-honours programs;
        # ranking is what tells them apart.
        matches = self.bot.gs.search_programs(text)
        if not matches:
            return (f"I couldn't find a program matching \"{text}\" in the "
                    f"calendar. Could you give the exact program/degree name?")
        if len(matches) > 1:
            ranked = self.bot._rank_program_matches(text, matches)
            best, runner_up = ranked[0], ranked[1]
            if (best["_extra"], best["_size"]) == (runner_up["_extra"], runner_up["_size"]):
                # Genuine tie -> ask, and remember every candidate so the
                # student's reply next turn resolves against this exact list
                # (path (a)) instead of being fuzzy-searched all over again.
                options = [{"pid": m["pid"], "title": m["title"],
                            "credential": m.get("credential") or ""} for m in ranked]
                state["pending_clarify"] = {"kind": "program_disambiguation",
                                            "options": options}
                shown = options[:PLANNING_MAX_PROGRAM_OPTIONS]
                names = "; ".join(f"{o['title']} ({o['credential']})" for o in shown)
                more = (f" (and {len(options) - len(shown)} more)"
                        if len(options) > len(shown) else "")
                return (f"A few programs match \"{text}\": {names}{more}. Which one? "
                        f"Copy the one you're in.")
            matches = ranked
        m = matches[0]
        state["program"] = {"pid": m["pid"], "title": m["title"],
                            "credential": m.get("credential") or ""}
        state["pending_clarify"] = None
        log["program_resolved_via"] = "search"
        return None

    def _run_planning(self, question: str, history: list[dict], audience: str,
                      log: dict, planning_state: dict | None = None):
        yield {"type": "status", "data": "Working out your course options…"}

        # Client-echoed state is untrusted input: sanitize, and on anything
        # missing/malformed fall back to full derivation from history (exactly
        # the pre-persistence behavior).
        state = _sanitize_planning_state(planning_state)
        had_state = state is not None
        log["planning_state_in"] = had_state
        incoming = _state_fingerprint(state)
        if state is None:
            state = _empty_planning_state()

        ex = self._planning_slots(question, history, state if had_state else None)
        self._merge_planning_slots(state, ex)

        def _state_event():
            """Emit the updated state iff it actually changed this turn."""
            if _state_fingerprint(state) != incoming:
                return [{"type": "planning_state", "data": state}]
            return []

        # Program: resolve (or keep the pin) before the missing-slot check, so
        # a turn that only answers the disambiguation question counts as having
        # a program.
        prog_text = ex["program"] if (ex["program"] and
                                      (ex["program_changed"] or not state["program"] or
                                       state.get("pending_clarify"))) else None
        clarify = self._resolve_program(prog_text, question, state, log)
        if clarify:
            log["termination_reason"] = "clarify_pending"
            yield from _state_event()
            yield {"type": "clarify", "data": clarify}
            return

        # Pre-check: required slots. Missing -> clarify and STOP (design §4.1).
        missing = []
        if not state["program"]:
            missing.append("your program or major")
        if not (state["term_season"] and state["term_year"]):
            missing.append("which term you're planning for (e.g. Spring 2026)")
        if not state["completed_known"]:
            missing.append("which courses you've already completed (or say you're "
                           "just starting)")
        if missing:
            log["termination_reason"] = "clarify_pending"
            log["missing_slots"] = missing
            yield from _state_event()
            yield {"type": "clarify",
                   "data": ("To plan your courses I need a bit more: "
                            + "; ".join(missing) + ".")}
            return

        pid = state["program"]["pid"]
        completed = state["completed_courses"]

        # Step 1: enumerate eligible courses (graph, deterministic).
        yield {"type": "status", "data": "Finding courses you're eligible for…"}
        remaining_groups = self.bot.gs.requirements_remaining(pid, completed)
        candidate_codes: list[str] = []
        seen: set[str] = set()
        for grp in remaining_groups:
            for code in grp.get("remaining", []):
                if code not in seen:
                    seen.add(code)
                    candidate_codes.append(code)
        # Keep courses whose COURSE prerequisites are satisfied by `completed`.
        # Non-course requirements (year standing, declared program, GPA,
        # permission, ...) are NOT auto-excluded here — they can't be read off
        # the completed-courses list, so they're resolved separately below by
        # inferring from the conversation, or asking, rather than silently
        # dropping the course.
        eligible: list[str] = []
        non_course_blocked: dict[str, list[dict]] = {}
        for code in candidate_codes:
            try:
                result = self.bot.gs.prereq_satisfied(code, completed)
            except Exception:
                continue
            if result.get("missing"):
                continue  # real course prereqs unmet -> not eligible, unambiguous
            unknowns = result.get("unknown_requirements") or []
            if unknowns:
                non_course_blocked[code] = unknowns
            else:
                eligible.append(code)

        non_course_notes: dict[str, str] = {}
        if non_course_blocked:
            # Same pinning rule as `program`: a course whose non-course
            # requirements were already settled in an earlier turn is never
            # re-resolved (and so never re-asked about) — only genuinely new
            # blocked courses go to the LLM.
            settled = state["non_course_resolved"]
            for code in non_course_blocked:
                verdict = settled.get(code)
                if not verdict:
                    continue
                if verdict["satisfied"]:
                    eligible.append(code)
                else:
                    non_course_notes[code] = (verdict.get("reason")
                                              or "requirements you haven't confirmed")
            fresh = {c: reqs for c, reqs in non_course_blocked.items() if c not in settled}
            log["n_non_course_persisted"] = len(non_course_blocked) - len(fresh)
            if fresh:
                resolved_codes, unresolved_reqs, fresh_notes = \
                    self._resolve_non_course_requirements(fresh, history, question)
                eligible.extend(resolved_codes)
                non_course_notes.update(fresh_notes)
                for code in resolved_codes:
                    settled[code] = {"satisfied": True, "reason": None}
                for code, note in fresh_notes.items():
                    settled[code] = {"satisfied": False, "reason": note}
                if unresolved_reqs:
                    log["termination_reason"] = "clarify_pending"
                    log["unresolved_non_course_reqs"] = unresolved_reqs
                    yield from _state_event()
                    yield {"type": "clarify",
                           "data": ("A few of your remaining courses depend on things "
                                    "I can't tell from our conversation — do you meet "
                                    "these: " + "; ".join(unresolved_reqs) + "?")}
                    return

        eligible = eligible[:PLANNING_MAX_ELIGIBLE]
        log["n_remaining_groups"] = len(remaining_groups)
        log["n_eligible"] = len(eligible)
        log["n_non_course_resolved"] = len(non_course_blocked) - len(non_course_notes)
        log["n_non_course_contradicted"] = len(non_course_notes)

        if not eligible:
            log["termination_reason"] = "enumeration_complete"
            blocked = {c: "prerequisites not yet met with your completed courses"
                      for c in candidate_codes if c not in eligible and c not in non_course_notes}
            blocked.update(non_course_notes)
            context = _render_remaining_groups(remaining_groups, blocked)
            sources = self.bot.format_sources([], {}, {}, {})
            yield from _state_event()
            yield from self._stream_answer(
                question, context, history, sources,
                _planning_system_prompt(self.bot.SYSTEM_PROMPT))
            return

        # Step 2: batch Banner for live sections (distinguish error vs not-offered).
        yield {"type": "status", "data": "Checking live class times and seats…"}
        season, year = state["term_season"], state["term_year"]
        banner_facts = banner_retrieve(eligible, season, year)
        banner_error = False
        offered_codes: set[str] = set()
        if not banner_facts:
            # Ambiguous whole-batch {} -> retry per course to tell apart a real
            # failure from "none offered" (design §4.4).
            per_course_courses = []
            term_label = ""
            term = ""
            any_success = False
            for code in eligible:
                bf = banner_retrieve([code], season, year)
                if bf:
                    any_success = True
                    term_label = bf.get("term_label", term_label)
                    term = bf.get("term", term)
                    for c in bf.get("courses", []):
                        offered_codes.add(c["code"])
                        per_course_courses.append(c)
            if any_success:
                banner_facts = {"kind": "availability", "term": term,
                                "term_label": term_label, "courses": per_course_courses}
            else:
                banner_error = True
                banner_facts = {}
        else:
            offered_codes = {c["code"] for c in banner_facts.get("courses", [])}

        not_offered = [c for c in eligible if c not in offered_codes]
        log.update(banner_error=banner_error, n_offered=len(offered_codes),
                   n_not_offered=len(not_offered))

        # Step 3 + 4: hard-filter via constraints (handled inside the solver:
        # include/exclude enforced structurally) then solve.
        yield {"type": "status", "data": "Building conflict-free schedules…"}
        sections = [s for c in banner_facts.get("courses", []) for s in c.get("sections", [])]
        schedules = solve_schedule(sections, state["constraints"])
        log["termination_reason"] = "solver_returned"
        log["n_schedules"] = len(schedules)

        # Build the answer context: a synthesized schedule-options block + the
        # live Banner blocks (for [n] citation).
        yield {"type": "status", "data": "Writing up your options…"}
        banner_context, _ = self.bot._assemble_context([], {}, banner_facts, {})
        sched_text = _render_schedules(schedules, banner_facts.get("term_label", ""))
        gaps = []
        if banner_error:
            gaps.append("Live availability could not be checked (Banner was "
                        "unreachable) — seat/time data is unavailable this run.")
        if not_offered:
            gaps.append("Not offered in this term (no sections found): "
                        + ", ".join(not_offered) + ".")
        non_course = [g["label"] for g in remaining_groups if g.get("has_non_course_reqs")]
        if non_course:
            gaps.append("These requirement groups also have non-course "
                        "requirements (year standing / GPA / permission) that a "
                        "course list can't verify: " + "; ".join(g or "(unlabeled)"
                                                                 for g in non_course) + ".")
        if non_course_notes:
            gaps.append("Not included in the schedule options because you haven't "
                        "confirmed you meet their requirements: " + "; ".join(
                            f"{c} ({note})" for c, note in non_course_notes.items()) + ".")
        gap_text = ("\n\nPLANNING NOTES:\n- " + "\n- ".join(gaps)) if gaps else ""
        context = f"{sched_text}{gap_text}\n\n{banner_context}".strip()

        sources = self.bot.format_sources([], {}, banner_facts, {})
        yield from _state_event()
        yield from self._stream_answer(
            question, context, history, sources,
            _planning_system_prompt(self.bot.SYSTEM_PROMPT))

    # ------------------------------------------------------------------
    # Mode: situational (fixed, scoped)
    # ------------------------------------------------------------------

    def _situational_slots(self, question: str, history: list[dict]) -> dict:
        data = self._json_llm(
            f"{self._history_text(history)}"
            "A UVic student is describing a procedural/policy situation. Extract "
            "these slots. Return ONLY JSON:\n"
            '  "situation_type": one of '
            f"{list(SITUATION_TYPES.keys())} (use \"other_policy\" if none clearly "
            "fits).\n"
            '  "situation_type_confident": true/false — false if you are guessing.\n'
            '  "what_happened": one-sentence summary of the situation, or null.\n'
            '  "deadline": any deadline the student mentioned (as written), or '
            "null.\n"
            '  "actions_taken": what the student says they have already done, or '
            "null.\n\n"
            f"Latest message: {question}",
            thinking="disabled", max_tokens=800,
        ) or {}
        stype = data.get("situation_type")
        if stype not in SITUATION_TYPES:
            stype = "other_policy"
        return {
            "situation_type": stype,
            "confident": bool(data.get("situation_type_confident")),
            "what_happened": data.get("what_happened") or None,
            "deadline": data.get("deadline") or None,
            "actions_taken": data.get("actions_taken") or None,
        }

    def _run_situational(self, question: str, history: list[dict], audience: str,
                         log: dict):
        yield {"type": "status", "data": "Understanding your situation…"}
        slots = self._situational_slots(question, history)

        # If we can't tell what kind of situation this is, ask (design §5.1).
        if not slots["confident"] and slots["situation_type"] == "other_policy" \
                and not slots["what_happened"]:
            log["termination_reason"] = "clarify_pending"
            yield {"type": "clarify",
                   "data": ("I want to point you to the right policy and office. "
                            "Could you say a bit more about what's happened — for "
                            "example an academic-integrity notice, a hold on your "
                            "account, a withdrawal, a grade issue, or something "
                            "else?")}
            return

        situation = SITUATION_TYPES[slots["situation_type"]]
        log.update(situation_type=slots["situation_type"],
                   situation_confident=slots["confident"],
                   deadline=slots["deadline"])

        # Fixed, scoped retrieval — pre-baked policy query seeds for this type.
        yield {"type": "status", "data": "Looking up the relevant policy…"}
        seen_chunk_ids: set[str] = set()
        chunks: list[dict] = []
        for q in situation["queries"]:
            for ch in self.bot.vector_retrieve(q, audience=audience, n=SCATTERED_QUERY_N):
                cid = ch.get("chunk_id")
                if cid and cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    chunks.append(ch)

        log.update(termination_reason="fixed_checklist_complete", n_chunks=len(chunks),
                   policy_found=bool(chunks))

        yield {"type": "status", "data": "Writing a plan…"}
        context, _ = self.bot._assemble_context(chunks, {}, {}, {})
        sources = self.bot.format_sources(chunks, {}, {}, {})
        system_prompt = _situational_system_prompt(
            self.bot.SYSTEM_PROMPT, situation, slots["deadline"], slots["actions_taken"])
        yield from self._stream_answer(question, context, history, sources, system_prompt)


# ---------------------------------------------------------------------------
# Rendering / small helpers
# ---------------------------------------------------------------------------

def _coerce_year(value) -> int | None:
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _norm_codes(value) -> list[str]:
    """Normalize a course-code list to the graph's spaceless upper form."""
    if not isinstance(value, list):
        return []
    return [c.replace(" ", "").upper() for c in value if isinstance(c, str) and c.strip()]


# ---------------------------------------------------------------------------
# course_planning persisted state (client-echoed — see the module docstring)
# ---------------------------------------------------------------------------

def _empty_planning_state() -> dict:
    return {
        "program": None,
        "term_season": None,
        "term_year": None,
        "completed_courses": [],
        "completed_known": False,
        "constraints": {"include_courses": [], "exclude_courses": [],
                        "soft_preferences": []},
        "non_course_resolved": {},
        "pending_clarify": None,
    }


def _sanitize_planning_state(raw) -> dict | None:
    """Validate client-echoed planning state into the canonical shape.

    Returns None when there's nothing usable — the caller then derives
    everything from history, i.e. the original behavior. This runs on data
    that came back through the browser, so every field is re-checked rather
    than trusted; a partially-bad payload keeps whatever fields survive.
    """
    if not isinstance(raw, dict):
        return None
    try:
        state = _empty_planning_state()

        prog = raw.get("program")
        if isinstance(prog, dict) and prog.get("pid"):
            state["program"] = {"pid": str(prog["pid"]),
                                "title": str(prog.get("title") or ""),
                                "credential": str(prog.get("credential") or "")}

        season = raw.get("term_season")
        if isinstance(season, str) and season.strip().lower() in ("spring", "summer", "fall"):
            state["term_season"] = season.strip().lower()
        state["term_year"] = _coerce_year(raw.get("term_year"))

        state["completed_courses"] = _norm_codes(raw.get("completed_courses"))
        state["completed_known"] = bool(raw.get("completed_known"))

        cons = raw.get("constraints")
        if isinstance(cons, dict):
            state["constraints"] = {
                "include_courses": _norm_codes(cons.get("include_courses")),
                "exclude_courses": _norm_codes(cons.get("exclude_courses")),
                "soft_preferences": [p for p in (cons.get("soft_preferences") or [])
                                     if isinstance(p, dict) and p.get("type")],
            }

        ncr = raw.get("non_course_resolved")
        if isinstance(ncr, dict):
            state["non_course_resolved"] = {
                str(code).replace(" ", "").upper(): {
                    "satisfied": bool(v.get("satisfied")),
                    "reason": (str(v["reason"]) if v.get("reason") else None),
                }
                for code, v in ncr.items() if isinstance(v, dict)
            }

        pc = raw.get("pending_clarify")
        if isinstance(pc, dict) and pc.get("kind"):
            options = [{"pid": str(o["pid"]), "title": str(o.get("title") or ""),
                        "credential": str(o.get("credential") or "")}
                       for o in (pc.get("options") or [])
                       if isinstance(o, dict) and o.get("pid")]
            state["pending_clarify"] = {"kind": str(pc["kind"]), "options": options}

        # Nothing actually survived validation (e.g. a client that sent `{}`).
        # Report it as absent so the caller uses full derivation from history —
        # delta extraction with an empty baseline would ignore anything the
        # student stated in an earlier turn.
        if _state_fingerprint(state) == _state_fingerprint(_empty_planning_state()):
            return None
        return state
    except Exception as e:
        print(f"  [thinking._sanitize_planning_state] discarding bad state: {e}",
              file=sys.stderr)
        return None


def _state_fingerprint(state: dict | None) -> str:
    """Stable serialization, so we only emit a `planning_state` event when the
    state actually changed this turn."""
    return json.dumps(state, sort_keys=True, default=str)


def _program_tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def _match_program_option(reply: str, options: list[dict]) -> dict | None:
    """Match a student's reply to a disambiguation question against the exact
    options we offered. Deterministic (no graph search, no LLM) — this is what
    the fuzzy re-search used to get wrong.

    Exact normalized match on "title (credential)" wins; otherwise the reply's
    tokens must be a subset of exactly ONE option's tokens. Ambiguity returns
    None (ask again) rather than guessing the wrong requirements list.
    """
    tokens = _program_tokens(reply)
    if not tokens or not options:
        return None
    exact = [o for o in options
             if _program_tokens(f"{o['title']} {o['credential']}") == tokens]
    if len(exact) == 1:
        return exact[0]
    subset = [o for o in options
              if set(tokens) <= set(_program_tokens(f"{o['title']} {o['credential']}"))]
    return subset[0] if len(subset) == 1 else None


def _hhmm(t: str | None) -> str:
    return f"{t[:2]}:{t[2:]}" if t and len(t) == 4 else "?"


def _render_remaining_groups(remaining_groups: list[dict],
                             blocked: dict[str, str] | None = None) -> str:
    """Render `GraphStore.requirements_remaining` output into a plain-text
    block for the answer model. Used on the no-eligible-courses path, where
    there is no solver output to narrate — without this, the model was left
    with only a one-line status sentence and no grounded requirement detail,
    and would fabricate plausible-looking group labels/courses to fill the
    gap (real incident: invented "Year 4 (CSC)" / "Honours project" style
    groups for a program whose actual remaining groups were never shown to
    it). This is the actual program structure, so it's the only thing the
    model should describe.
    """
    if not remaining_groups:
        return ("PROGRAM REQUIREMENTS: No outstanding requirement groups were "
                "found for this program given the completed courses — the "
                "student may have finished the enumerable requirements.")
    lines = ["PROGRAM REQUIREMENTS REMAINING (from the official program "
             "structure — this is the ONLY source of group/requirement "
             "detail; do not invent, rename, or assume any group, label, or "
             "course beyond what's listed here):"]
    for grp in remaining_groups:
        label = grp.get("label") or "(unlabeled group)"
        remaining = grp.get("remaining") or []
        lines.append(f"\n{label}:")
        if remaining:
            for code in remaining:
                note = f" — {blocked[code]}" if blocked and code in blocked else ""
                lines.append(f"  - {code}{note}")
        if grp.get("has_non_course_reqs"):
            lines.append("  - (plus non-course requirements in this group: year "
                         "standing / GPA / permission / units-from-list — not "
                         "verifiable from a course list)")
    return "\n".join(lines)


def _render_schedules(schedules: list[dict], term_label: str) -> str:
    """Render solver output into a plain text block for the answer model."""
    if not schedules:
        return ("SCHEDULE OPTIONS: No conflict-free combination of the eligible, "
                "offered sections was found under the given constraints. Explain "
                "the conflict to the student rather than inventing a schedule.")
    lines = [f"SCHEDULE OPTIONS ({term_label}) — computed conflict-free "
             f"combinations of live lecture sections:"]
    for n, sched in enumerate(schedules, 1):
        lines.append(f"\nOption {n} ({len(sched['courses'])} courses"
                     + (", all preferences met" if sched["prefs_met"] else "") + "):")
        for s in sched["sections"]:
            m = s.get("meeting") or {}
            when = (" ".join(m.get("days") or []) + f" {_hhmm(m.get('begin'))}-"
                    f"{_hhmm(m.get('end'))}") if (m.get("days") or m.get("begin")) else "async/TBA"
            seat = ("FULL" if not s.get("open") else f"{s.get('seats_available', '?')} seats")
            lines.append(f"  - {s['subject_course']} {s.get('section', '')} "
                         f"({s.get('schedule_type', '')}): {when} [{seat}]")
        if sched["notes"]:
            lines.append("  Trade-offs: " + "; ".join(sched["notes"]))
    return "\n".join(lines)
