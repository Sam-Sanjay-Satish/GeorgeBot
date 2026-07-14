"""
graph_queries.py — Cross-graph query helpers over the v2 course + program graphs.

Loads both pickled DiGraphs (course_graph.pkl, program_graph.pkl) once and
exposes a thin accessor layer for the structured queries Haiku routes here:
prerequisites, unlocks, cross-listings, program requirements, and the join
between them. course_ref ids in the program graph are course codes in the same
namespace as the course graph, which is what makes the cross-graph join trivial.

All accessors guard against missing / stub / dangling nodes and return empty
collections rather than raising.

Usage as a module:
    from graph_queries import GraphStore
    gs = GraphStore.load()
    gs.get_prereqs("CSC110")          # {"tree": <prereq_parsed>, "courses": [...]}
    gs.get_unlocks("MATH100")         # courses that list MATH100 as a prereq
    gs.programs_requiring("STAT260")  # programs whose trees reach STAT260

Quick CLI smoke test:
    python graph_queries.py CSC110
    python graph_queries.py --program BSC-ANSH
"""

import json
import os
import pathlib
import pickle

import networkx as nx

_HERE = pathlib.Path(__file__).parent

# Graph-artifact dir: BASE_DIR-relative by default (local dev), or a Railway
# Volume mount via env. GRAPH_DATA_DIR wins; else DATA_DIR/graph_data; else
# ./graph_data. Mirrors the CHROMA_DIR/TAXONOMY_FILE scheme in chatbot.py.
if os.getenv("GRAPH_DATA_DIR"):
    _GRAPH_DIR = pathlib.Path(os.environ["GRAPH_DATA_DIR"])
elif os.getenv("DATA_DIR"):
    _GRAPH_DIR = pathlib.Path(os.environ["DATA_DIR"]) / "graph_data"
else:
    _GRAPH_DIR = _HERE / "graph_data"

_COURSE_PKL = _GRAPH_DIR / "course_graph.pkl"
_PROGRAM_PKL = _GRAPH_DIR / "program_graph.pkl"
_OUTLINES_JSON = _GRAPH_DIR / "heat_outlines.json"


def _norm(code: str) -> str:
    return (code or "").replace(" ", "").upper()


# ---------------------------------------------------------------------------
# Tree-walking helpers used by the new accessors
# ---------------------------------------------------------------------------

_NON_COURSE_LEAF_TYPES = frozenset({
    "year_standing", "declared", "permission", "awr", "gpa",
    "high_school", "text", "conditional", "admission", "units",
})


def _describe_non_course(node: dict) -> dict:
    """Human-readable dict for a non-course requirement leaf node."""
    t = node.get("type")
    _ord = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
    if t == "year_standing":
        yr = node.get("min_year", "?")
        return {"type": t, "description": f"Minimum {_ord.get(yr, str(yr) + 'th')}-year standing"}
    if t == "declared":
        return {"type": t, "description": f"Declared: {node.get('program', '')}"}
    if t == "permission":
        return {"type": t, "description": f"Permission of {node.get('of', 'department')}"}
    if t == "awr":
        return {"type": t, "description": "Academic Writing Requirement (AWR) satisfied"}
    if t == "gpa":
        scope = node.get("scope_desc") or ""
        return {"type": t, "description": f"GPA ≥ {node.get('min_gpa')} {scope}".strip()}
    if t == "high_school":
        return {"type": t, "description": f"High school: {node.get('value', '')}"}
    if t in ("text", "conditional"):
        return {"type": t, "description": node.get("value", "")}
    if t == "admission":
        return {"type": t, "description": f"Admission to {node.get('to', '')}"}
    if t == "units":
        return {"type": t, "description": f"Minimum {node.get('min_units')} units of {node.get('description', '')}"}
    if t == "units_from":
        items = node.get("items") or []
        codes = [i.get("code", "") for i in items if i.get("type") == "course"]
        return {"type": t, "description": f"Minimum {node.get('min_units')} units from: {', '.join(codes)}"}
    return {"type": t, "raw": node}


def _tree_course_titles(node: dict, out: dict) -> None:
    """Collect {code: title} for every course node in a requirement tree."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "course":
        out[node.get("code", "")] = node.get("title", "")
    for item in (node.get("items") or []):
        _tree_course_titles(item, out)


def _collect_course_codes_from_tree(node: dict, out: set) -> None:
    """Recursively collect all course-leaf codes from a requirement tree."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "course":
        code = node.get("code")
        if code:
            out.add(code)
    for item in (node.get("items") or []):
        _collect_course_codes_from_tree(item, out)


def _eval_prereq_tree(node: dict, completed: set, titles: dict) -> tuple:
    """
    Evaluate a prereq tree against a set of completed course codes.

    Returns (missing, unknowns):
      missing  — list of course gaps; each item is either
                 {"code": ..., "title": ...} for a single course, or
                 {"one_of": n, "options": [{"code": ..., "title": ...}, ...]}
                 for an OR-group.  OR-groups flatten all leaf codes from
                 unsatisfied branches so callers never see nested one_of dicts.
      unknowns — list of non-course requirements that cannot be evaluated
                 programmatically (year standing, declared program, GPA, etc.)
    """
    if not isinstance(node, dict):
        return [], []
    t = node.get("type")
    items = node.get("items") or []

    if t == "course":
        code = node.get("code", "")
        if code in completed:
            return [], []
        return [{"code": code, "title": node.get("title", "")}], []

    if t == "concurrent":
        return [], []  # corequisite — skip here, handled by get_corequisites()

    if t in _NON_COURSE_LEAF_TYPES:
        return [], [_describe_non_course(node)]

    if t == "units_from":
        required = node.get("min_units", 0)
        course_items = [i for i in items if i.get("type") == "course"]
        done_units = sum(
            float(i.get("credits") or 0)
            for i in course_items if i.get("code", "") in completed
        )
        if done_units >= required:
            return [], []
        remaining = [i for i in course_items if i.get("code", "") not in completed]
        opts = [{"code": i["code"], "title": i.get("title", "")} for i in remaining]
        note = f"need {required - done_units:.4g} more units from this list"
        if len(opts) == 1:
            return [{**opts[0], "note": note}], []
        return [{"one_of": int(required - done_units), "options": opts, "note": note}], []

    if t == "grade":
        scope = node.get("scope", 1)
        course_items = [i for i in items if i.get("type") == "course"]
        done = sum(1 for i in course_items if i.get("code", "") in completed)
        if done >= scope:
            return [], []
        remaining = [i for i in course_items if i.get("code", "") not in completed]
        need = scope - done
        note = f"min grade {node.get('min_grade')} required"
        opts = [{"code": i["code"], "title": i.get("title", "")} for i in remaining]
        if need == 1 and len(opts) == 1:
            return [{**opts[0], "note": note}], []
        return [{"one_of": need, "options": opts, "note": note}], []

    if t == "gpa_grade":
        course_items = [i for i in items if i.get("type") == "course"]
        if any(i.get("code", "") in completed for i in course_items):
            return [], []
        opts = [{"code": i["code"], "title": i.get("title", "")} for i in course_items]
        note = f"min GPA {node.get('min_gpa')} required"
        return [{"one_of": 1, "options": opts, "note": note}], []

    if t == "all":
        all_m, all_u = [], []
        for item in items:
            m, u = _eval_prereq_tree(item, completed, titles)
            all_m.extend(m)
            all_u.extend(u)
        return all_m, all_u

    if t == "any":
        n = node.get("n", 1)
        sat_count = 0
        unsat_codes: list = []   # flat leaf codes from all unsatisfied branches
        unsat_unknowns: list = []
        for item in items:
            m, u = _eval_prereq_tree(item, completed, titles)
            if not m and not u:
                sat_count += 1
            else:
                # Flatten: collect all leaf codes from this unsatisfied branch
                branch_codes: set = set()
                _collect_course_codes_from_tree(item, branch_codes)
                for code in sorted(branch_codes - completed):
                    unsat_codes.append({"code": code, "title": titles.get(code, "")})
                unsat_unknowns.extend(u)
        if sat_count >= n:
            return [], []
        still_need = n - sat_count
        if unsat_codes:
            if still_need == 1 and len(unsat_codes) == 1:
                return unsat_codes, []
            return [{"one_of": still_need, "options": unsat_codes}], []
        return [], unsat_unknowns

    return [], [{"type": t, "raw": node}]


def _collect_non_course_reqs(node: dict, out: list) -> None:
    """Walk a tree, appending every non-course requirement to `out`."""
    if not isinstance(node, dict):
        return
    t = node.get("type")
    if t in _NON_COURSE_LEAF_TYPES or t == "units_from":
        out.append(_describe_non_course(node))
        return
    if t == "grade":
        codes = [i.get("code", "") for i in (node.get("items") or []) if i.get("type") == "course"]
        out.append({"type": "grade", "description": f"Min grade {node.get('min_grade')} required in: {', '.join(codes)}"})
        return
    if t == "gpa_grade":
        codes = [i.get("code", "") for i in (node.get("items") or []) if i.get("type") == "course"]
        out.append({"type": "gpa_grade", "description": f"Min GPA {node.get('min_gpa')} in one of: {', '.join(codes)}"})
        return
    for item in (node.get("items") or []):
        _collect_non_course_reqs(item, out)


def _collect_concurrent_nodes(node: dict, out: list) -> None:
    """Walk a tree, collecting every `concurrent` node into `out`."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "concurrent":
        out.append(node)
        return  # concurrent nodes don't nest
    for item in (node.get("items") or []):
        _collect_concurrent_nodes(item, out)


def _find_alternatives_in_tree(node: dict, target: str, out: set) -> None:
    """
    Walk a prereq tree; whenever `target` appears directly as a course-leaf
    inside an `any` node, add all sibling course codes to `out`.
    """
    if not isinstance(node, dict):
        return
    t = node.get("type")
    items = node.get("items") or []
    if t == "any":
        direct = {i.get("code") for i in items if i.get("type") == "course"}
        if target in direct:
            out.update(direct - {target})
    for item in items:
        _find_alternatives_in_tree(item, target, out)


def _load_outlines(path=_OUTLINES_JSON) -> dict:
    """Load v2.2's heat_outlines.json — a flat {code: {course, term, url, text}}
    dict (HEAT-only, i.e. eng/CS courses; other courses have no outline) — and
    re-key with _norm() for lookup safety."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {_norm(code): rec for code, rec in raw.items()}


class GraphStore:
    def __init__(self, course_graph: nx.DiGraph, program_graph: nx.DiGraph, outlines: dict):
        self.cg = course_graph
        self.pg = program_graph
        self.outlines = outlines  # dict: course_code -> outline dict

    @classmethod
    def load(cls, course_pkl=_COURSE_PKL, program_pkl=_PROGRAM_PKL) -> "GraphStore":
        with open(course_pkl, "rb") as f:
            cg = pickle.load(f)
        with open(program_pkl, "rb") as f:
            pg = pickle.load(f)
        outlines = _load_outlines()
        return cls(cg, pg, outlines)

    # ----- course side -------------------------------------------------------

    def get_course(self, code: str) -> dict | None:
        code = _norm(code)
        if code in self.cg and self.cg.nodes[code].get("kind") == "course":
            return {"code": code, **self.cg.nodes[code]}
        return None

    def get_outline(self, code: str) -> dict | None:
        """Return the course outline for `code`, or None if no outline exists."""
        return self.outlines.get(_norm(code))

    def get_prereqs(self, code: str) -> dict:
        """Structured prereq tree (AND/OR preserved) + flat set of prereq codes."""
        code = _norm(code)
        if code not in self.cg:
            return {"tree": None, "courses": []}
        tree = self.cg.nodes[code].get("prereq_tree")
        courses = sorted(
            v for _u, v, d in self.cg.out_edges(code, data=True)
            if d.get("edge_type") == "prereq"
        )
        return {"tree": tree, "courses": courses}

    def get_unlocks(self, code: str) -> list:
        """Courses that list `code` as a prerequisite."""
        code = _norm(code)
        if code not in self.cg:
            return []
        return sorted(
            u for u, _v, d in self.cg.in_edges(code, data=True)
            if d.get("edge_type") == "prereq"
        )

    def prereq_chain(self, code: str) -> set:
        """Transitive prerequisite closure (all courses required, directly or not)."""
        code = _norm(code)
        if code not in self.cg:
            return set()
        seen: set = set()
        stack = [code]
        while stack:
            cur = stack.pop()
            for _u, v, d in self.cg.out_edges(cur, data=True):
                if d.get("edge_type") == "prereq" and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def cross_listings(self, code: str) -> list:
        code = _norm(code)
        if code not in self.cg:
            return []
        return sorted(
            v for _u, v, d in self.cg.out_edges(code, data=True)
            if d.get("edge_type") == "cross_listed"
        )

    def prereq_satisfied(self, code: str, completed: list) -> dict:
        """
        Evaluate whether `code`'s prerequisites are met by the completed course list.

        Returns:
          satisfied          — True only when all course requirements are met and no
                               unknown (non-course) requirements remain.
          missing            — course gaps respecting AND/OR structure.  Each item is
                               {"code": ..., "title": ...} for a single required course,
                               or {"one_of": n, "options": [...]} for an OR-group where
                               n more options still need to be satisfied.
          unknown_requirements — non-course requirements (year standing, GPA, declared
                                 program, permission, etc.) that cannot be evaluated
                                 programmatically; the caller must surface these to the user.
        """
        code = _norm(code)
        completed_set = {_norm(c) for c in completed}
        if code not in self.cg or self.cg.nodes[code].get("kind") != "course":
            return {"satisfied": False, "missing": [], "unknown_requirements": []}
        tree = self.cg.nodes[code].get("prereq_tree")
        if not tree:
            return {"satisfied": True, "missing": [], "unknown_requirements": []}
        titles: dict = {}
        _tree_course_titles(tree, titles)
        missing, unknowns = _eval_prereq_tree(tree, completed_set, titles)
        return {
            "satisfied": not missing and not unknowns,
            "missing": missing,
            "unknown_requirements": unknowns,
        }

    def get_eligibility(self, code: str) -> dict:
        """
        Like get_prereqs(), but also surfaces non-course requirements that have no
        course-code leaf and therefore produce no prereq edges — year standing, GPA,
        declared program, permission, AWR, high-school prerequisites, etc.

        A course like CSC499 has zero course prerequisites but two non-course
        requirements (4th-year standing + Honours declaration); get_prereqs() returns
        an empty courses list for it, which looks like "no prerequisites" and is wrong.

        Returns:
          tree       — the full prereq_parsed tree (same as get_prereqs)
          courses    — sorted list of direct course prereq codes (same as get_prereqs)
          non_course — list of non-course requirements with human-readable descriptions
        """
        code = _norm(code)
        if code not in self.cg:
            return {"tree": None, "courses": [], "non_course": []}
        tree = self.cg.nodes[code].get("prereq_tree")
        courses = sorted(
            v for _u, v, d in self.cg.out_edges(code, data=True)
            if d.get("edge_type") == "prereq"
        )
        non_course: list = []
        _collect_non_course_reqs(tree, non_course)
        return {"tree": tree, "courses": courses, "non_course": non_course}

    def get_corequisites(self, code: str) -> dict:
        """
        Return corequisite requirements for `code` — courses that must be enrolled
        in concurrently (same term), parsed from `concurrent` nodes in the prereq tree.

        Returns:
          courses — list of {"code": ..., "title": ..., "need_one_of": bool}; when
                    need_one_of=True only one of the courses in that concurrent group
                    is required, not all of them.
          note    — non-None if "concurrent"/"coreq" language is detected in
                    prereq_text but was not parsed into a structured concurrent node.
        """
        code = _norm(code)
        if code not in self.cg or self.cg.nodes[code].get("kind") != "course":
            return {"courses": [], "note": None}
        tree = self.cg.nodes[code].get("prereq_tree")
        concurrent_nodes: list = []
        _collect_concurrent_nodes(tree, concurrent_nodes)
        courses = []
        for cn in concurrent_nodes:
            course_items = [i for i in (cn.get("items") or []) if i.get("type") == "course"]
            n = cn.get("n")
            need_one_of = n == 1 and len(course_items) > 1
            for item in course_items:
                courses.append({
                    "code": item.get("code", ""),
                    "title": item.get("title", ""),
                    "need_one_of": need_one_of,
                })
        note = None
        if not concurrent_nodes:
            pt = (self.cg.nodes[code].get("prereq_text") or "").lower()
            if any(kw in pt for kw in ("coreq", "concurrent", "concurrently", "simultaneously")):
                note = "corequisite language detected in prereq_text; check prereq_text for details"
        return {"courses": courses, "note": note}

    # ----- program side ------------------------------------------------------

    def search_programs(self, query: str) -> list:
        """
        Fuzzy-match programs by title, code, or credential, returning every candidate
        rather than just the best one. Two programs can share a title and differ only
        by credential (e.g. "Computer Science and Mathematics" is both a Combined
        Major and a Combined Honours program) — picking just one silently produces a
        real but wrong requirements list for the other. Callers (or the student)
        should disambiguate among the returned candidates before calling
        program-scoped accessors like requirements_remaining().

        Matching is token-based, not whole-string substring: `query` is split on
        whitespace and every token must appear somewhere in the combined
        "title code credential" text (case-insensitive). This is what lets a query
        like "computer science honours" match a program whose title is just
        "Computer Science" — "honours" lives in its credential ("Bachelor of Science
        - Honours"), not its title, so a literal substring match on the full query
        string would silently return nothing.

        Returns a list of {"pid", "code", "title", "credential"} dicts, sorted by
        title then code. Empty list if nothing matches or `query` is blank.
        """
        tokens = (query or "").strip().lower().split()
        if not tokens:
            return []
        matches = []
        for pid, d in self.pg.nodes(data=True):
            if d.get("kind") != "program":
                continue
            title = d.get("title") or ""
            code = d.get("code") or ""
            credential = d.get("credential") or ""
            haystack = f"{title} {code} {credential}".lower()
            if all(tok in haystack for tok in tokens):
                matches.append({
                    "pid": pid,
                    "code": code,
                    "title": title,
                    "credential": d.get("credential"),
                })
        return sorted(matches, key=lambda m: (m["title"], m["code"]))

    def _find_program(self, q: str):
        if q in self.pg and self.pg.nodes[q].get("kind") == "program":
            return q
        q_up = q.upper()
        for n, d in self.pg.nodes(data=True):
            if d.get("kind") == "program" and (d.get("code") or "").upper() == q_up:
                return n
        return None

    def get_program(self, q: str) -> dict | None:
        pid = self._find_program(q)
        if not pid:
            return None
        return {"pid": pid, **self.pg.nodes[pid]}

    def program_requirement_groups(self, q: str) -> list:
        """Top-level requirement groups (label + raw tree). Excludes specializations."""
        pid = self._find_program(q)
        if not pid:
            return []
        groups = []
        for _u, gid, d in self.pg.out_edges(pid, data=True):
            if d.get("edge_type") == "has_group":
                gd = self.pg.nodes[gid]
                groups.append({"label": gd.get("label"), "tree": gd.get("requirement_tree")})
        return groups

    def program_specializations(self, q: str) -> list:
        pid = self._find_program(q)
        if not pid:
            return []
        specs = []
        for _u, sid, d in self.pg.out_edges(pid, data=True):
            if d.get("edge_type") == "has_specialization":
                sd = self.pg.nodes[sid]
                specs.append({"title": sd.get("title"), "notes": sd.get("notes")})
        return specs

    def program_courses(self, q: str) -> list:
        """All course codes referenced anywhere in a program (groups + specializations)."""
        pid = self._find_program(q)
        if not pid:
            return []
        codes: set = set()
        # BFS over has_group / has_specialization / spec_has_group, collect requires.
        frontier = [pid]
        visited = set()
        while frontier:
            cur = frontier.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for _u, v, d in self.pg.out_edges(cur, data=True):
                et = d.get("edge_type")
                if et == "requires":
                    codes.add(v)
                elif et in ("has_group", "has_specialization", "spec_has_group"):
                    frontier.append(v)
        return sorted(codes)

    def programs_requiring(self, code: str) -> list:
        """Programs whose requirement trees reach `code` (via any group/specialization)."""
        code = _norm(code)
        if code not in self.pg:
            return []
        progs: set = set()
        for gid in self.pg.predecessors(code):
            # gid is a requirement_group; walk up to the owning program.
            for parent in self.pg.predecessors(gid):
                kind = self.pg.nodes[parent].get("kind")
                if kind == "program":
                    progs.add(parent)
                elif kind == "specialization":
                    for pp in self.pg.predecessors(parent):
                        progs.add(pp)
        return sorted(
            self.pg.nodes[p].get("code") or p for p in progs
        )

    def requirements_remaining(self, program_query: str, completed: list) -> list:
        """
        Return outstanding course requirements for a program, organized by
        requirement group so the result is useful for a 'what do I still need
        for my degree?' question rather than just a flat list.

        Returns a list of dicts, one per requirement group (including groups
        inside specializations):
          label              — year/section label (e.g. "Years 1 and 2"; may be "")
          remaining          — sorted list of course codes not in `completed`
          has_non_course_reqs — True when the group tree contains units_from, text,
                                or other non-enumerable requirements not captured in
                                the `remaining` list; the caller should note these gaps.
        Groups whose remaining list is empty and has_non_course_reqs is False are
        omitted from the result.
        """
        pid = self._find_program(program_query)
        if not pid:
            return []
        completed_set = {_norm(c) for c in completed}
        result: list = []

        def _walk(parent_id: str, spec_label: str = "") -> None:
            for _u, child_id, d in self.pg.out_edges(parent_id, data=True):
                et = d.get("edge_type")
                if et in ("has_group", "spec_has_group"):
                    gd = self.pg.nodes[child_id]
                    tree = gd.get("requirement_tree")
                    codes: set = set()
                    _collect_course_codes_from_tree(tree, codes)
                    remaining = sorted(c for c in codes if c not in completed_set)
                    non_course: list = []
                    _collect_non_course_reqs(tree, non_course)
                    label = (gd.get("label") or "").strip()
                    if spec_label:
                        label = f"{spec_label}: {label}".strip(": ")
                    if remaining or non_course:
                        result.append({
                            "label": label,
                            "remaining": remaining,
                            "has_non_course_reqs": bool(non_course),
                        })
                elif et == "has_specialization":
                    spec_title = (self.pg.nodes[child_id].get("title") or child_id).strip()
                    _walk(child_id, spec_label=spec_title)

        _walk(pid)
        return result

    # ----- cross-graph -------------------------------------------------------

    def course_with_programs(self, code: str) -> dict:
        """Join a course record with the programs that require it."""
        return {
            "course": self.get_course(code),
            "prereqs": self.get_prereqs(code),
            "unlocks": self.get_unlocks(code),
            "cross_listed": self.cross_listings(code),
            "required_by_programs": self.programs_requiring(code),
        }

    def get_alternatives(self, code: str) -> list:
        """
        Return course codes that can substitute for `code` as a prerequisite —
        i.e. courses that appear alongside `code` in the same OR-group (any-node)
        in some other course's prereq tree, aggregated across the full catalog.

        Example: if CSC115 and CSC116 are both listed as options in a '1 of:'
        group for CSC225, then get_alternatives("CSC115") includes CSC116
        (and vice versa).

        Returns a sorted list of course codes.  Only considers direct course
        leaves inside any-nodes, not compound sub-expressions.
        """
        code = _norm(code)
        if code not in self.cg:
            return []
        alternatives: set = set()
        for upstream in self.get_unlocks(code):
            tree = self.cg.nodes[upstream].get("prereq_tree")
            _find_alternatives_in_tree(tree, code, alternatives)
        return sorted(alternatives)


def _cli() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Smoke-test the graph query helpers.")
    ap.add_argument("course", nargs="?", help="course code to inspect")
    ap.add_argument("--program", help="program code/pid to inspect")
    ap.add_argument(
        "--completed", nargs="*", default=[],
        help="completed course codes for prereq_satisfied / requirements_remaining",
    )
    args = ap.parse_args()

    gs = GraphStore.load()
    if args.program:
        prog = gs.get_program(args.program)
        print(json.dumps({
            "program": {k: prog.get(k) for k in ("pid", "code", "title", "credential")} if prog else None,
            "requirement_groups": [g["label"] for g in gs.program_requirement_groups(args.program)],
            "specializations": [s["title"] for s in gs.program_specializations(args.program)],
            "n_courses": len(gs.program_courses(args.program)),
            "requirements_remaining": gs.requirements_remaining(args.program, args.completed),
        }, indent=2, ensure_ascii=False))
    elif args.course:
        result = gs.course_with_programs(args.course)
        result["prereq_chain"] = sorted(gs.prereq_chain(args.course))
        result["eligibility"] = gs.get_eligibility(args.course)
        result["corequisites"] = gs.get_corequisites(args.course)
        result["alternatives"] = gs.get_alternatives(args.course)
        result["prereq_satisfied"] = gs.prereq_satisfied(args.course, args.completed)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
