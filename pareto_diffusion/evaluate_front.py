import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
import numpy as np
from pareto_diffusion.core import OBJECTIVE_NAMES, summarize_front
from pareto_diffusion.data import load_table


def objectives_from_records(path):
    grouped=defaultdict(list)
    with open(path,encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            record=json.loads(line)
            if record.get("status")!="ok": continue
            m=record.get("normalized_metrics",{})
            grouped[record["candidate_id"]].append([
                m["disturbance_resistance_score"],m["contact_coverage_norm"],m["angular_span_norm"]])
    if not grouped: raise ValueError("No successful benchmark records found")
    return np.stack([np.mean(values,axis=0) for values in grouped.values()])


def main():
    p=argparse.ArgumentParser(description="Evaluate a three-objective generated Pareto front")
    source=p.add_mutually_exclusive_group(required=True); source.add_argument("--records"); source.add_argument("--objectives_npz")
    p.add_argument("--reference_table"); p.add_argument("--reference",type=float,nargs=3,default=(0,0,0)); p.add_argument("--output",required=True)
    args=p.parse_args()
    if args.records: objectives=objectives_from_records(args.records)
    else:
        with np.load(args.objectives_npz,allow_pickle=False) as z: objectives=np.asarray(z["objectives"],dtype=float)
    reference_front=None
    if args.reference_table:
        table=load_table(args.reference_table); mask=(table.splits=="test")&table.feasible
        reference_front=table.objectives[mask] if mask.sum() else table.objectives[table.feasible]
    summary,ids=summarize_front(objectives,args.reference,reference_front)
    result={"objective_names":OBJECTIVE_NAMES,"summary":summary,"non_dominated_ids":ids.tolist()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
