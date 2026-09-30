"""End-to-end exercise of PRISMGraphPipeline with stubbed agents.

Runs the whole graph against fake Agent 1/2/3 implementations - no PDF,
no model, no network - and asserts the properties that distinguish the
graph pipeline from the sequential one: parallel fan-out,
replace-by-category reduction, remediation of only the failed
categories, framework ordering, and resume-after-crash.

    python tests/test_pipeline_graph.py
"""
import json, os, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WORK = Path(tempfile.mkdtemp(prefix="prism_graph_test_"))
CATEGORIES = [f"Category {i}" for i in range(1, 9)]

# --- synthetic framework file ---
fw = WORK / "framework.json"
fw.write_text(json.dumps([
    {"category": c, "definition": f"def {c}", "scoring_definitions": ["0: no", "5: yes"],
     "indicators": [f"ind {c}"]} for c in CATEGORIES
]))

from PRISM.config import PRISMConfig
from PRISM.pipeline_graph import PRISMGraphPipeline
from PRISM.agents.analysis import SubElementAnalysis
from PRISM.agents.quality import QualityReport, QualityScore

cfg = PRISMConfig(
    domain="cancer",
    framework_path=str(fw),
    output_dir=str(WORK / "out"),
    checkpoint_dir=str(WORK / "ckpt"),
    log_dir=str(WORK / "logs"),
    enable_page_references=False,
)
cfg.agent3.max_remediation_attempts = 3

pipe = PRISMGraphPipeline(cfg, max_concurrency=4)

# ---------------- stubs ----------------
class FakeIngestion:
    def process_document_lightweight(self, path):
        class R:
            total_pages = 3
            full_text = "PAGE1 text\nPAGE2 text\nPAGE3 text"
            page_texts = {1: "one", 2: "two", 3: "three"}
        return R()
    process_document = process_document_lightweight

call_log = []
log_lock = threading.Lock()
active = [0]
peak = [0]

class FakeAgent2:
    def initialize(self):
        with log_lock:
            call_log.append("INIT")
    def _analyze_sub_element(self, category, doc_text, page_texts, domain):
        with log_lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            call_log.append(category)
        time.sleep(0.05)                      # simulate an LLM call
        with log_lock:
            active[0] -= 1
        assert doc_text.startswith("PAGE1"), "document text not delivered to node"
        assert page_texts[2] == "two", "page_texts keys must be ints"
        # Category 3 fails twice, then passes -> exercises the remediation loop.
        n = sum(1 for c in call_log if c == category)
        score = 0 if (category == "Category 3" and n <= 2) else 4
        return SubElementAnalysis(
            category=category, response="r" * 60, score=score,
            scoring_reasoning="j" * 40,
            response_page_citations=[], scoring_reasoning_page_citations=[])

class FakeAgent3:
    def validate_analysis(self, analyses):
        scores, failed = [], []
        for a in analyses:
            passed = a["score"] > 0
            if not passed:
                failed.append(a["category"])
            scores.append(QualityScore(a["category"], .9, .9, .9, .9,
                                       .9 if passed else .1, [], passed))
        return QualityReport(scores, sum(s.composite for s in scores) / len(scores),
                             sum(1 for s in scores if s.passed) / len(scores),
                             failed, 0)

pipe.agent1 = FakeIngestion()
pipe.agent2 = FakeAgent2()
pipe.agent3 = FakeAgent3()

# ---------------- run ----------------
res = pipe.process_document(str(WORK / "fake.pdf"), "Testland", "2024",
                            lightweight_ingestion=True, run_id="RUN1")

print("\n" + "=" * 60)
ok = True
def check(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

check("all categories analyzed", res["total_elements"] == len(CATEGORIES),
      f"({res['total_elements']}/{len(CATEGORIES)})")
check("framework order preserved",
      [a["category"] for a in res["analyses"]] == CATEGORIES)
check("remediation loop ran twice", res["remediation_attempts"] == 2,
      f"(got {res['remediation_attempts']})")
check("no duplicate categories (reducer replaces)",
      len({a["category"] for a in res["analyses"]}) == len(CATEGORIES))
check("Category 3 healed to final score",
      next(a for a in res["analyses"] if a["category"] == "Category 3")["score"] == 4)
check("remediation re-ran ONLY the failure",
      [c for c in call_log if c != "INIT"].count("Category 1") == 1)
check("fan-out was parallel", peak[0] > 1, f"(peak concurrency {peak[0]})")
check("max_concurrency respected", peak[0] <= 4, f"(peak {peak[0]})")
check("agent2 initialized exactly once", call_log.count("INIT") == 1)
check("quality pass rate 100%", res["quality_pass_rate"] == 1.0)
check("output JSON written", Path(res["output_path"]).exists())
check("doc side store on disk",
      (WORK / "ckpt" / "documents" / "RUN1.json").exists())
check("sqlite checkpoint db created",
      (WORK / "ckpt" / "prism_graph.db").exists())

written = json.loads(Path(res["output_path"]).read_text())
check("written JSON matches returned analyses", written == res["analyses"])

# ---------------- resume ----------------
before = len([c for c in call_log if c != "INIT"])
res2 = pipe.process_document(str(WORK / "fake.pdf"), "Testland", "2024",
                             lightweight_ingestion=True, run_id="RUN1")
after = len([c for c in call_log if c != "INIT"])
check("resume replays no LLM calls", after == before, f"({after - before} new calls)")
check("resumed result identical", res2["analyses"] == res["analyses"])

# ---------------- crash mid-run, then resume ----------------
# Cross-stage checkpointing is the headline benefit: a failure in Stage 3
# must not cost the Stage 2 LLM calls.
crashed = {"done": False}
real_validate = pipe.agent3.validate_analysis
def flaky_validate(analyses):
    if not crashed["done"]:
        crashed["done"] = True
        raise RuntimeError("simulated Stage 3 crash")
    return real_validate(analyses)
pipe.agent3.validate_analysis = flaky_validate

base = len([c for c in call_log if c != "INIT"])
try:
    pipe.process_document(str(WORK / "fake.pdf"), "Testland", "2024",
                          lightweight_ingestion=True, run_id="RUN2")
    check("crash propagated", False)
except RuntimeError:
    check("crash propagated", True)

mid = len([c for c in call_log if c != "INIT"])
check("stage 2 ran before crash", mid - base == len(CATEGORIES), f"({mid-base} calls)")

res3 = pipe.process_document(str(WORK / "fake.pdf"), "Testland", "2024",
                             lightweight_ingestion=True, run_id="RUN2")
end = len([c for c in call_log if c != "INIT"])
check("resume after crash replays no Stage 2 work",
      end - mid == 0, f"({end - mid} new analyze calls after resume)")
check("resumed run produced full output",
      res3["total_elements"] == len(CATEGORIES))

print("=" * 60)
print("\nMermaid:\n" + pipe.draw_mermaid())
print(f"\nworkdir: {WORK}")
sys.exit(0 if ok else 1)
