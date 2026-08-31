import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from benchmarks.candidates import save_candidates
from dgdm.sampler import DGDMDesignDiffusion
from generator.dataloader import DesignBounds, physical_to_diffusion
from generator.diffusion_utils import ConditionalUnet1D
from pareto_diffusion.core import crowding_distance, non_dominated_sort
from pareto_diffusion.data import load_table
from pareto_diffusion.model import PreferenceClassifier
from pareto_diffusion.sampler import preference_scores, sample_pareto_guided


def load_prior(path,steps,device):
    checkpoint=torch.load(path,map_location=device); bounds_path=os.path.join(os.path.dirname(path),"design_bounds.npz")
    bounds=DesignBounds.from_npz(bounds_path) if os.path.exists(bounds_path) else DesignBounds.defaults()
    timesteps=int(checkpoint.get("args",{}).get("timesteps",100))
    scheduler=DDIMScheduler(num_train_timesteps=timesteps,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    net=ConditionalUnet1D(1,0,[128,256],32); prior=DGDMDesignDiffusion(net,scheduler,bounds,steps).to(device)
    prior.load_state_dict(checkpoint["model"],strict=False)
    if "ema" in checkpoint: prior.ema.load_state_dict(checkpoint["ema"],device); prior.ema.copy_to(net)
    prior.eval(); prior.requires_grad_(False); return prior,bounds


def main():
    p=argparse.ArgumentParser(description="Generate Pareto preference-guided gripper designs")
    p.add_argument("--prior",required=True); p.add_argument("--preference",required=True); p.add_argument("--table",required=True)
    p.add_argument("--output",required=True); p.add_argument("--num_samples",type=int,default=256)
    p.add_argument("--batch_size",type=int,default=256); p.add_argument("--steps",type=int,default=20)
    p.add_argument("--guidance_scale",type=float,default=.1); p.add_argument("--references",type=int,default=16)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda"); args=p.parse_args()
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    prior,bounds=load_prior(args.prior,args.steps,device); checkpoint=torch.load(args.preference,map_location=device)
    if int(checkpoint["design_dim"]) != len(bounds.lo): raise ValueError("Preference and prior design dimensions differ")
    classifier=PreferenceClassifier(checkpoint["design_dim"],checkpoint["width"],checkpoint["time_dim"]).to(device)
    classifier.load_state_dict(checkpoint["model"]); classifier.eval(); classifier.requires_grad_(False)
    table=load_table(args.table); train=np.flatnonzero(table.splits=="train")
    ranks=non_dominated_sort(table.objectives[train],table.feasible[train],table.violation[train])
    reference_ids=train[(ranks==0)&table.feasible[train]]
    if not len(reference_ids): raise ValueError("Training table has no feasible Pareto references")
    reference_units=physical_to_diffusion(torch.tensor(table.designs[reference_ids],dtype=torch.float32,device=device),bounds).clamp(-1,1)
    generator=torch.Generator(device=device).manual_seed(args.seed); outputs=[]; started=time.time()
    remaining=args.num_samples
    while remaining:
        count=min(args.batch_size,remaining)
        outputs.append(sample_pareto_guided(prior,classifier,reference_units,count,args.guidance_scale,args.references,generator,device))
        remaining-=count
    physical=torch.cat([x["design_physical"] for x in outputs]); units=torch.cat([x["design_unit"] for x in outputs])
    with torch.no_grad():
        rank_refs=reference_units[:min(args.references,len(reference_units))]
        scores=preference_scores(classifier,units,rank_refs,torch.tensor(0,device=device)).cpu().numpy()
    order=np.argsort(scores)[::-1]; physical_np=physical.cpu().numpy()[order]; scores=scores[order]
    metadata={"prior":os.path.abspath(args.prior),"preference":os.path.abspath(args.preference),"table":os.path.abspath(args.table),
              "guidance_scale":args.guidance_scale,"inference_steps":args.steps,"references_per_step":args.references,
              "pareto_reference_count":len(reference_ids),"proposal_seconds":time.time()-started,
              "objective_names":["disturbance_resistance","contact_coverage","angular_span"]}
    save_candidates(args.output,physical_np,"pareto_guided_diffusion",args.seed,scores=scores,metadata=metadata)
    print(json.dumps({"output":os.path.abspath(args.output),"num_samples":len(physical_np),**metadata},indent=2))

if __name__=="__main__": main()
