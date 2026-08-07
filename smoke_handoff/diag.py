import json, sys
from pathlib import Path
REPO = Path(r"C:\Users\maxim\Projects\marag-precision"); SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from src import agents, models, pipeline, prompts, retrieval
from src.metrics import exact_match, f1_score
from datasets import load_dataset
spec = json.loads((SCRATCH/"smoke100.json").read_text()); wanted=set(spec["question_ids"])
ds = load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",revision="1908d6afbbead072334abe2965f91bd2709910ab")
qs = pipeline.load_questions(len(wanted), seed=20260807, exclude={r["id"] for r in ds}-wanted,
      name="hotpotqa/hotpot_qa", revision="1908d6afbbead072334abe2965f91bd2709910ab")[:12]
corpus = retrieval.build_corpus(name="hotpotqa/hotpot_qa", split="validation",
      revision="1908d6afbbead072334abe2965f91bd2709910ab", configs=("distractor","fullwiki"))
r = retrieval.RetrievalContext(corpus, k=10)
m, t = models.load_model("Qwen/Qwen2.5-1.5B-Instruct","4bit")
idx={}
for st in [s for s in prompts.PIPELINE_STAGES if s != prompts.SOLO_ROLE]:
    c = pipeline.build_stage_calls(st, qs, idx, retriever=r)
    if not c: continue
    for rec in agents.run_calls(m,t,st,c,"4bit",run_id="diag",batch_size=8):
        idx[(rec["question_id"],rec["stage"],rec["call_index"])]=rec
multi = pipeline.build_answer_records(qs, idx, run_id="diag", retriever=r)
sidx={}
for rec in agents.run_calls(m,t,"solo",pipeline.build_stage_calls("solo",qs,sidx,retriever=r),
                            "4bit",run_id="diag",batch_size=4):
    sidx[(rec["question_id"],rec["stage"],rec["call_index"])]=rec
single = pipeline.build_answer_records(qs, sidx, run_id="diag", retriever=r)
models.unload(m)
sing={a["question_id"]:a for a in single}
for a in multi:
    s=sing[a["question_id"]]
    print(f"\nQ  : {a['question'][:95]}")
    print(f"GOLD  : {a['gold_answer']!r}   [{a['retrieval_stratum']}]")
    print(f"MULTI : {a['predicted_answer'][:160]!r}  f1={a['f1']:.2f}  stage={a['answer_stage']}")
    print(f"SINGLE: {s['predicted_answer'][:160]!r}  f1={s['f1']:.2f}")
    qa=[idx.get((a['question_id'],st,0)) for st in ('qa','qa_step2')]
    for i,rec in enumerate(qa,1):
        if rec: print(f"   qa_step{i}: {(rec.get('parsed') or rec.get('salvaged') or {})}")
