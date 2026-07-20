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
    {"type": "status",  "data": "<phase text>"}
    {"type": "clarify", "data": "<question to the user>"}   # terminal — no answer
    {"type": "sources", "data": [<source dict>, ...]}
    {"type": "token",   "data": "<answer piece>"}
    {"type": "done"}
    {"type": "error",   "data": "<message>"}

Slot-gathering that needs input (course_planning / situational) emits a single
`clarify` event and stops; the user's reply arrives as an ordinary next turn in
`history`, and re-entry finds the slots there (rebuild-from-history — no
server-side session state, no conversation id).
"""

from __future__ import annotations

import json
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
        "sections) and live Banner availability. When you present options:\n"
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
            route: dict, plan: str, confidence: float):
        """Dispatch into one of the three plans. Yields event dicts (see module
        docstring). `route`/`plan`/`confidence` come from
        `GeorgeBot.rewrite_and_route(..., mode="default")` — the planner
        already classified the query and applied the confidence downgrade, so
        this no longer classifies anything itself (that used to be this
        module's own `classify_mode`)."""
        log: dict = {"question": question, "audience": audience,
                     "t_start": time.monotonic()}
        try:
            # Defensive guard in case an invalid plan value ever reaches here
            # (e.g. a future caller bypassing the planner) — mirrors the
            # planner's own fallback.
            if plan not in VALID_PLANS:
                plan = DEFAULT_PLAN

            log.update(mode_classified=plan, router_confidence=confidence)

            if plan == "course_planning":
                yield from self._run_planning(question, history, audience, log)
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

    def _planning_slots(self, question: str, history: list[dict]) -> dict:
        """Extract completed courses, program, target term, and constraints."""
        data = self._json_llm(
            f"{self._history_text(history)}"
            f"Today's date is {time.strftime('%Y-%m-%d')}.\n"
            "Extract course-planning slots from the conversation for a UVic "
            "student. Return ONLY JSON with:\n"
            '  "program": the student\'s program/major as stated (e.g. "computer '
            'science", "mechanical engineering"), or null if never stated.\n'
            '  "completed_courses": array of course codes they said they have '
            'completed (normalized like "CSC110"), or [] if none stated.\n'
            '  "completed_known": true if the student indicated what they have '
            "taken (or explicitly said none/first-year), false if it's simply "
            "never been mentioned.\n"
            '  "term_season": "spring"|"summer"|"fall"|null for the term they want '
            "to plan (resolve relative phrases using today's date).\n"
            '  "term_year": 4-digit year or null.\n'
            '  "include_courses": array of course codes they specifically want '
            "included, or [].\n"
            '  "exclude_courses": array of course codes they want to avoid, or [].\n'
            '  "soft_preferences": array of {"type","value"} using ONLY these '
            'types: no_time_before (HHMM), no_time_after (HHMM), no_days (array of '
            'day codes Mo/Tu/We/Th/Fr), delivery ("online"|"in_person"), '
            'max_courses (int). Parse phrases like "no morning classes" -> '
            '{"type":"no_time_before","value":"1000"}, "no Friday classes" -> '
            '{"type":"no_days","value":["Fr"]}, "5 courses" -> {"type":'
            '"max_courses","value":5}. Drop anything that doesn\'t fit these '
            "types.\n\n"
            f"Latest question: {question}",
            thinking="disabled", max_tokens=1200,
        ) or {}
        return {
            "program": data.get("program") or None,
            "completed_courses": [c.replace(" ", "").upper()
                                  for c in (data.get("completed_courses") or [])],
            "completed_known": bool(data.get("completed_known")),
            "term_season": (data.get("term_season") or "").strip().lower() or None,
            "term_year": data.get("term_year"),
            "constraints": {
                "include_courses": [c.replace(" ", "").upper()
                                    for c in (data.get("include_courses") or [])],
                "exclude_courses": [c.replace(" ", "").upper()
                                    for c in (data.get("exclude_courses") or [])],
                "soft_preferences": data.get("soft_preferences") or [],
            },
        }

    def _run_planning(self, question: str, history: list[dict], audience: str,
                      log: dict):
        yield {"type": "status", "data": "Working out your course options…"}
        slots = self._planning_slots(question, history)

        # Pre-check: required slots. Missing -> clarify and STOP (design §4.1).
        missing = []
        if not slots["program"]:
            missing.append("your program or major")
        if not (slots["term_season"] and slots["term_year"]):
            missing.append("which term you're planning for (e.g. Spring 2026)")
        if not slots["completed_known"]:
            missing.append("which courses you've already completed (or say you're "
                           "just starting)")
        if missing:
            log["termination_reason"] = "clarify_pending"
            log["missing_slots"] = missing
            yield {"type": "clarify",
                   "data": ("To plan your courses I need a bit more: "
                            + "; ".join(missing) + ".")}
            return

        # Resolve program. Mirror chatbot.py's _program_facts: rank by surplus
        # tokens and auto-pick a clear winner instead of asking on any raw
        # multi-match — search_programs only guarantees token-subset
        # containment, so e.g. "computer science honours" legitimately
        # matches the standalone Honours program AND both combined-honours
        # programs; ranking is what tells them apart.
        matches = self.bot.gs.search_programs(slots["program"])
        if not matches:
            log["termination_reason"] = "clarify_pending"
            yield {"type": "clarify",
                   "data": (f"I couldn't find a program matching "
                            f"\"{slots['program']}\" in the calendar. Could you give "
                            f"the exact program/degree name?")}
            return
        if len(matches) > 1:
            ranked = self.bot._rank_program_matches(slots["program"], matches)
            best, runner_up = ranked[0], ranked[1]
            if (best["_extra"], best["_size"]) == (runner_up["_extra"], runner_up["_size"]):
                names = "; ".join(f"{m['title']} ({m.get('credential', '')})"
                                  for m in matches[:6])
                log["termination_reason"] = "clarify_pending"
                yield {"type": "clarify",
                       "data": (f"A few programs match \"{slots['program']}\": {names}. "
                                f"Which one?")}
                return
            matches = ranked
        pid = matches[0]["pid"]
        completed = slots["completed_courses"]

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
        # Keep only courses whose prerequisites are satisfied by what's completed.
        eligible: list[str] = []
        for code in candidate_codes:
            try:
                if self.bot.gs.prereq_satisfied(code, completed).get("satisfied"):
                    eligible.append(code)
            except Exception:
                continue
        eligible = eligible[:PLANNING_MAX_ELIGIBLE]
        log["n_remaining_groups"] = len(remaining_groups)
        log["n_eligible"] = len(eligible)

        if not eligible:
            log["termination_reason"] = "enumeration_complete"
            context = ("No outstanding courses with satisfied prerequisites were "
                       "found for this program given the completed courses. The "
                       "student may have finished the enumerable requirements, or "
                       "the remaining ones are non-course requirements.")
            sources = self.bot.format_sources([], {}, {}, {})
            yield from self._stream_answer(
                question, context, history, sources,
                _planning_system_prompt(self.bot.SYSTEM_PROMPT))
            return

        # Step 2: batch Banner for live sections (distinguish error vs not-offered).
        yield {"type": "status", "data": "Checking live class times and seats…"}
        season, year = slots["term_season"], _coerce_year(slots["term_year"])
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
        schedules = solve_schedule(sections, slots["constraints"])
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
        gap_text = ("\n\nPLANNING NOTES:\n- " + "\n- ".join(gaps)) if gaps else ""
        context = f"{sched_text}{gap_text}\n\n{banner_context}".strip()

        sources = self.bot.format_sources([], {}, banner_facts, {})
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


def _hhmm(t: str | None) -> str:
    return f"{t[:2]}:{t[2:]}" if t and len(t) == 4 else "?"


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
