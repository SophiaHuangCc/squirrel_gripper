import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from generator.dataloader import DesignBounds
from pareto_diffusion.data import PreferencePairDataset, load_table
from pareto_diffusion.model import PreferenceClassifier


def evaluate(model,loader,scheduler,device,generator):
    model.eval(); total=correct=count=0
    with torch.no_grad():
        for b in loader:
            a,bdesign,label=b["design_a"].to(device),b["design_b"].to(device),b["label"].to(device)
            t=torch.randint(0,scheduler.config.num_train_timesteps,(len(a),),device=device,generator=generator)
            na=torch.randn(a.shape,device=device,generator=generator); nb=torch.randn(bdesign.shape,device=device,generator=generator)
            logits=model(scheduler.add_noise(a,na,t),scheduler.add_noise(bdesign,nb,t),t)
            total += torch.nn.functional.binary_cross_entropy_with_logits(logits,label,reduction="sum").item()
            correct += ((logits>=0)==(label>=.5)).sum().item(); count += len(a)
    return total/max(count,1),correct/max(count,1)


def main():
    p=argparse.ArgumentParser(description="Train diffusion-time Pareto preference classifier")
    p.add_argument("--table",required=True); p.add_argument("--prior",required=True); p.add_argument("--save_dir",required=True)
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch_size",type=int,default=512)
    p.add_argument("--max_pairs",type=int,default=200000); p.add_argument("--workers",type=int,default=4)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--width",type=int,default=256)
    p.add_argument("--time_dim",type=int,default=64); p.add_argument("--seed",type=int,default=0); p.add_argument("--device",default="cuda")
    args=p.parse_args(); torch.manual_seed(args.seed)
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    prior_ckpt=torch.load(args.prior,map_location="cpu"); timesteps=int(prior_ckpt.get("args",{}).get("timesteps",100))
    bounds_path=os.path.join(os.path.dirname(args.prior),"design_bounds.npz")
    bounds=DesignBounds.from_npz(bounds_path) if os.path.exists(bounds_path) else DesignBounds.defaults()
    table=load_table(args.table); train=PreferencePairDataset(table,"train",bounds,args.max_pairs,args.seed)
    if (table.splits=="val").sum()<2:
        raise ValueError("Validation split has fewer than two designs; rebuild the table with more data or a larger val fraction")
    val=PreferencePairDataset(table,"val",bounds,min(args.max_pairs//5,50000),args.seed+1)
    train_loader=DataLoader(train,args.batch_size,shuffle=True,num_workers=args.workers)
    val_loader=DataLoader(val,args.batch_size,shuffle=False,num_workers=args.workers)
    scheduler=DDIMScheduler(num_train_timesteps=timesteps,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    model=PreferenceClassifier(len(bounds.lo),args.width,args.time_dim).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    generator=torch.Generator(device=device).manual_seed(args.seed); os.makedirs(args.save_dir,exist_ok=True); best=float("inf")
    for epoch in range(1,args.epochs+1):
        model.train(); total=count=0
        for batch in train_loader:
            a,b,label=batch["design_a"].to(device),batch["design_b"].to(device),batch["label"].to(device)
            t=torch.randint(0,timesteps,(len(a),),device=device,generator=generator)
            na=torch.randn(a.shape,device=device,generator=generator); nb=torch.randn(b.shape,device=device,generator=generator)
            logits=model(scheduler.add_noise(a,na,t),scheduler.add_noise(b,nb,t),t)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,label)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total += loss.item()*len(a); count += len(a)
        val_loss,val_acc=evaluate(model,val_loader,scheduler,device,generator)
        print(f"epoch={epoch} train_loss={total/max(count,1):.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        state={"model":model.state_dict(),"epoch":epoch,"design_dim":len(bounds.lo),"width":args.width,"time_dim":args.time_dim,
               "num_train_timesteps":timesteps,"objective_names":list(("disturbance_resistance","contact_coverage","angular_span")),
               "table":os.path.abspath(args.table),"args":vars(args)}
        torch.save(state,os.path.join(args.save_dir,"last.pt"))
        if val_loss<best: best=val_loss; torch.save(state,os.path.join(args.save_dir,"best.pt"))
    print(json.dumps({"save_dir":os.path.abspath(args.save_dir),"train_pairs":len(train),"val_pairs":len(val)},indent=2))

if __name__=="__main__": main()
