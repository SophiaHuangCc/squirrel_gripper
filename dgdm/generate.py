import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dgdm.data import PROFILE_CHANNELS
from dgdm.guidance import ProfileTarget, ScenarioBatch, aggregate_profile_score
from dgdm.models import InteractionProfileModel
from dgdm.sampler import DGDMDesignDiffusion
from generator.dataloader import DesignBounds
from generator.diffusion_utils import ConditionalUnet1D


def main():
    p=argparse.ArgumentParser(description="Generate designs with inference-time DGDM objectives")
    p.add_argument("--prior",required=True); p.add_argument("--dynamics",required=True); p.add_argument("--task",required=True)
    p.add_argument("--output",default="dgdm/runs/generated.npz"); p.add_argument("--num_samples",type=int,default=256)
    p.add_argument("--steps",type=int,default=20); p.add_argument("--guidance_scale",type=float,default=.1)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda"); args=p.parse_args()
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    prior_ckpt=torch.load(args.prior,map_location=device); dyn_ckpt=torch.load(args.dynamics,map_location=device)
    bounds_path=os.path.join(os.path.dirname(args.prior),"design_bounds.npz"); bounds=DesignBounds.from_npz(bounds_path)
    train_steps=int(prior_ckpt.get("args",{}).get("timesteps",100)); scheduler=DDIMScheduler(num_train_timesteps=train_steps,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    net=ConditionalUnet1D(1,0,[128,256],32); prior=DGDMDesignDiffusion(net,scheduler,bounds,args.steps).to(device)
    prior.load_state_dict(prior_ckpt["model"],strict=False); prior.ema.load_state_dict(prior_ckpt["ema"],device); prior.ema.copy_to(net); prior.eval()
    profile_steps=int(dyn_ckpt["profile_steps"]); dynamics=InteractionProfileModel(profile_steps=profile_steps,channels=len(PROFILE_CHANNELS)).to(device)
    dynamics.load_state_dict(dyn_ckpt["model"]); dynamics.eval(); dynamics.requires_grad_(False)
    with open(args.task,encoding="utf-8") as f: spec=json.load(f)
    scenarios=ScenarioBatch(torch.tensor(spec["scenarios"],dtype=torch.float32,device=device), torch.tensor(spec.get("scenario_weights"),device=device) if "scenario_weights" in spec else None)
    target=ProfileTarget.from_dict(spec["target"],profile_steps,device)
    generator=torch.Generator(device=device).manual_seed(args.seed)
    out=prior.sample(args.num_samples,dynamics,scenarios,target,args.guidance_scale,generator,device)
    with torch.no_grad(): scores=aggregate_profile_score(dynamics,out["design_norm"],scenarios,target)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)
    np.savez_compressed(args.output,design_params=out["design_physical"].cpu().numpy(),design_params_norm=out["design_norm"].cpu().numpy(),scores=scores.cpu().numpy(),task_json=np.asarray([json.dumps(spec)]))
    print("[SAVED]",os.path.abspath(args.output))

if __name__ == "__main__": main()
