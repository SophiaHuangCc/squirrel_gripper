import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pareto_diffusion.core import crowding_distance, non_dominated_sort
from pareto_diffusion.data import build_table, save_table


def main():
    p=argparse.ArgumentParser(description="Build leakage-safe three-objective Pareto table")
    p.add_argument("--data_dir",required=True); p.add_argument("--output",required=True)
    p.add_argument("--val_fraction",type=float,default=.1); p.add_argument("--test_fraction",type=float,default=.1)
    p.add_argument("--split_seed",type=int,default=0); p.add_argument("--min_scenarios",type=int,default=1)
    p.add_argument("--max_failure_rate",type=float,default=0.0); args=p.parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction+args.test_fraction >= 1:
        p.error("split fractions must be nonnegative and sum to less than one")
    table=build_table(args.data_dir,args.val_fraction,args.test_fraction,args.split_seed,args.min_scenarios,args.max_failure_rate)
    save_table(args.output,table)
    summary={split:int((table.splits==split).sum()) for split in ("train","val","test")}
    train=table.splits=="train"; ranks=non_dominated_sort(table.objectives[train],table.feasible[train],table.violation[train])
    crowding_distance(table.objectives[train],ranks)
    print(json.dumps({"output":os.path.abspath(args.output),"designs":len(table.designs),"splits":summary,
                      "train_pareto_designs":int((ranks==0).sum())},indent=2))

if __name__=="__main__": main()
