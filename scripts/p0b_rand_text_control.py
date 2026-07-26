"""#13 text encoder swap: replace CLIP embeddings with random Gaussian noise."""
import json, torch, numpy as np
from pathlib import Path; from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms; from scipy.signal import find_peaks

DEVICE="cuda"; DTYPE=torch.float16
IMAGES=sorted(Path("/home/hiaskc/Talant/graduation/data/coco_val").glob("*.jpg"))[:19]

pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",local_files_only=True,torch_dtype=DTYPE).to(DEVICE)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)

def disc(unet):
    t=[]
    for n,m in unet.named_modules():
        p=n.split(".")
        if "resnets" in p:
            idx=p.index("resnets")
            if len(p)==idx+2 and p[-1].isdigit():t.append(n)
        if "transformer_blocks" in p:
            idx=p.index("transformer_blocks")
            if len(p)==idx+2 and p[-1]=="0":t.append(n)
    return sorted(t)

class H:
    def __init__(s,u):s.u=u;s.f={};s.h=[]
    def _fn(s,n):
        def fn(m,i,o):s.f[n]=o.detach().float().cpu()
        return fn
    def reg(s,t):
        s.rm()
        for n,m in s.u.named_modules():
            if n in t:s.h.append(m.register_forward_hook(s._fn(n)))
    def rm(s):
        for h in s.h:h.remove()
        s.h.clear();s.f.clear()

def ddim_inv(pipe,lat,pe,N):
    s=pipe.scheduler;s.set_timesteps(N,device=DEVICE)
    ts=s.timesteps;z=lat.clone();ext=ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1];npred=pipe.unet(z,tc,encoder_hidden_states=pe).sample
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            c1=(an/ac).sqrt();sc=(1-ac).sqrt();sn=(1-an).sqrt()
            z=c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon(pipe,noise,pe,N):
    s=pipe.scheduler;s.set_timesteps(N,device=DEVICE);z=noise.clone()
    with torch.no_grad():
        for t in s.timesteps:npred=pipe.unet(z,t,encoder_hidden_states=pe).sample;z=s.step(npred,t,z).prev_sample
    return z

def load_enc(pipe,path,sz=512):
    img=Image.open(path).convert("RGB").resize((sz,sz))
    t=transforms.ToTensor()(img).unsqueeze(0).to(DEVICE,dtype=DTYPE);t=2*t-1
    with torch.no_grad():l=pipe.vae.encode(t).latent_dist.sample()
    return l*pipe.vae.config.scaling_factor

def gini(x):
    x=np.sort(np.asarray(x,dtype=np.float64));n=len(x);s=np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s>0 else 0.0

def feat_v2(profile_raw):
    p=np.asarray(profile_raw,dtype=np.float64);L=len(p)
    dmin,dmax=p.min(),p.max()
    pn=(p-dmin)/(dmax-dmin) if dmax>dmin else p.copy()
    pp=float(np.argmax(pn))/L
    k=max(1,int(np.ceil(0.2*L)));top=np.argsort(pn)[-k:]
    conc=float(np.sum(pn[top])/np.sum(pn));sp=float(gini(pn))
    return {"peak_position":pp,"concentration":conc,"spread":sp}

def dist_v2(fa,fb):
    d_pp=abs(fa["peak_position"]-fb["peak_position"])
    d_mag=np.linalg.norm([fa["concentration"]-fb["concentration"],fa["spread"]-fb["spread"]])
    return {"D_total":float(np.linalg.norm([d_pp,d_mag])),"D_peak_pos":d_pp,"D_mag":d_mag}

targets=disc(pipe.unet);hooker=H(pipe.unet);hooker.reg(targets)
# Random text embeddings: same shape as CLIP output (77 tokens, 768 dim)
rand_pe = torch.randn(1, 77, 768, device=DEVICE, dtype=DTYPE)
print("Rand embed shape:", rand_pe.shape, "norm:", torch.norm(rand_pe).item())
print("layers=%d imgs=%d"%(len(targets),len(IMAGES)))
per_img=[]
for imp in IMAGES:
    latent=load_enc(pipe,imp)
    z_inv=ddim_inv(pipe,latent,rand_pe,50)
    hooker.f.clear()
    with torch.no_grad():pipe.unet(z_inv,pipe.scheduler.timesteps[0],encoder_hidden_states=rand_pe).sample
    inv_f={k:v.clone() for k,v in hooker.f.items()}
    z_recon=ddim_recon(pipe,z_inv,rand_pe,50)
    hooker.f.clear()
    with torch.no_grad():pipe.unet(z_recon,pipe.scheduler.timesteps[0],encoder_hidden_states=rand_pe).sample
    recon_f={k:v.clone() for k,v in hooker.f.items()}
    ld={}
    for ln in targets:
        if ln in inv_f and ln in recon_f:ld[ln]=float(torch.norm(inv_f[ln]-recon_f[ln],p=2).item())
    per_img.append(ld)
hooker.rm();del pipe;torch.cuda.empty_cache()

pi_feats=[]
for ld in per_img:
    ordered=sorted(ld.items(),key=lambda x:targets.index(x[0]))
    profile=np.array([v for _,v in ordered]);pi_feats.append(feat_v2(profile))
feat_rand={k:float(np.mean([pi[k] for pi in pi_feats])) for k in pi_feats[0]}
feat_base={"peak_position":0.7632,"concentration":0.5583,"spread":0.5110}
dd=dist_v2(feat_base,feat_rand)
noise_p95=0.0163;min_inter=0.092
tag="BELOW_NOISE" if dd["D_total"]<noise_p95 else ("WEAK_VAR" if dd["D_total"]<min_inter else "CROSS_ARCH")
print("RAND_TEXT: pp=%.4f conc=%.4f sp=%.4f"%(feat_rand["peak_position"],feat_rand["concentration"],feat_rand["spread"]))
print("D_s(base,RAND): D_total=%.6f D_pp=%.6f D_mag=%.6f -> %s"%(dd["D_total"],dd["D_peak_pos"],dd["D_mag"],tag))
json.dump({"RAND_TEXT":feat_rand,"baseline":feat_base,"D_s":dd,"verdict":tag},open("/home/hiaskc/Talant/graduation/outputs/p0b_cross_checkpoint/rand_text_fingerprint.json","w"),indent=2)
print("Saved.")
