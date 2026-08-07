import json, sys
from collections import Counter
from pathlib import Path
REPO=Path(r"C:\Users\maxim\Projects\marag-precision"); SC=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO))
from src import agents, models, pipeline, prompts, retrieval
from datasets import load_dataset
N=40
spec=json.loads((SC/"smoke100.json").read_text()); wanted=set(spec["question_ids"])
ds=load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",revision="1908d6afbbead072334abe2965f91bd2709910ab")
qs=pipeline.load_questions(len(wanted),seed=20260807,exclude={r["id"] for r in ds}-wanted,
    name="hotpotqa/hotpot_qa",revision="1908d6afbbead072334abe2965f91bd2709910ab")[:N]
corpus=retrieval.build_corpus(name="hotpotqa/hotpot_qa",split="validation",
    revision="1908d6afbbead072334abe2965f91bd2709910ab",configs=("distractor","fullwiki"))
r=retrieval.RetrievalContext(corpus,k=10)
m,t=models.load_model("Qwen/Qwen2.5-1.5B-Instruct","4bit")
idx={}
for st in [s for s in prompts.PIPELINE_STAGES if s!=prompts.SOLO_ROLE]:
    c=pipeline.build_stage_calls(st,qs,idx,retriever=r)
    if not c: continue
    for rec in agents.run_calls(m,t,st,c,"4bit",run_id="q",batch_size=8):
        idx[(rec["question_id"],rec["stage"],rec["call_index"])]=rec
multi=pipeline.build_answer_records(qs,idx,run_id="q",retriever=r)
sidx={}
for rec in agents.run_calls(m,t,"solo",pipeline.build_stage_calls("solo",qs,sidx,retriever=r),
        "4bit",run_id="q",batch_size=4):
    sidx[(rec["question_id"],rec["stage"],rec["call_index"])]=rec
single=pipeline.build_answer_records(qs,sidx,run_id="q",retriever=r); models.unload(m)
sing={a["question_id"]:a for a in single}
empty=[a for a in multi if not (a["predicted_answer"] or "").strip()]
nonempty=[a for a in multi if (a["predicted_answer"] or "").strip()]
succ=Counter()
for a in multi:
    for st in ("qa","qa_step2","qa_step3"):
        rec=idx.get((a["question_id"],st,0))
        if rec: succ[(st,((rec.get("parsed") or rec.get("salvaged") or {}).get("success")))]+=1
print(f"\nn={len(multi)}")
print(f"MULTI empty answers      : {len(empty)}/{len(multi)} = {len(empty)/len(multi):.0%}")
print(f"  of those, SINGLE scored : mean f1 {sum(sing[a['question_id']]['f1'] for a in empty)/max(len(empty),1):.3f}")
print(f"\nF1  multi(all)           : {sum(a['f1'] for a in multi)/len(multi):.3f}")
print(f"F1  multi(non-empty only): {sum(a['f1'] for a in nonempty)/max(len(nonempty),1):.3f}  (n={len(nonempty)})")
print(f"F1  single(all)          : {sum(a['f1'] for a in single)/len(single):.3f}")
print(f"F1  single on same non-empty subset: {sum(sing[a['question_id']]['f1'] for a in nonempty)/max(len(nonempty),1):.3f}")
print("\nper-step QA success flags:", dict(succ))
