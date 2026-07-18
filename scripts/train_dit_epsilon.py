"""
Train DiT-S/2 with eps-prediction DDPM objective.
Variant A: predict epsilon (noise) given noised image.
"""
from __future__ import annotations

import os, sys, json, time, copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dit_controlled_shared import (
    DEVICE, DTYPE, OUTPUT_DIR, BATCH_SIZE, TOTAL_STEPS, WARMUP_STEPS,
    LEARNING_RATE, EMA_DECAY, GRAD_CLIP, LOG_EVERY, SAMPLE_EVERY,
    CHECKPOINT_EVERY, DIT_CONFIG,
    set_seed, get_dit_s2_model, get_train_loader,
    NoiseScheduleDDPM, EMA,
)

OUT_DIR = OUTPUT_DIR / "epsilon"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    model = get_dit_s2_model()
    model.train()
    ema = EMA(model)

    noise_sched = NoiseScheduleDDPM()
    loader = get_train_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scaler = GradScaler()

    loss_log = []
    start_step = 0

    # Resume from checkpoint if available
    ckpt_files = sorted(OUT_DIR.glob("checkpoint_*.pt"))
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"[eps] Resuming from {latest}")
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        loss_log = ckpt.get("loss_log", [])
        print(f"[eps] Resumed at step {start_step}")

    t0 = time.time()

    print(f"[eps] Starting training: {TOTAL_STEPS} steps, batch {BATCH_SIZE}")
    print(f"[eps] Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    for step in range(start_step, TOTAL_STEPS):
        # LR warmup
        if step < WARMUP_STEPS:
            lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        try:
            x0 = next(data_iter).to(DEVICE)
        except (NameError, StopIteration):
            data_iter = iter(loader)
            x0 = next(data_iter).to(DEVICE)

        B = x0.shape[0]

        optimizer.zero_grad()

        with autocast(dtype=DTYPE):
            noise = torch.randn_like(x0)
            t = torch.randint(0, noise_sched.T, (B,), device=DEVICE)
            x_t = noise_sched.add_noise(x0, noise, t)
            cl = torch.zeros(B, dtype=torch.long, device=DEVICE)
            pred = model(x_t, timestep=t, class_labels=cl).sample
            loss = F.mse_loss(pred, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        loss_log.append(loss.item())

        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            avg_loss = sum(loss_log[-100:]) / len(loss_log[-100:])
            print(f"[eps] step {step+1}/{TOTAL_STEPS} | loss={avg_loss:.6f} | "
                  f"lr={lr_now:.2e} | elapsed={elapsed:.0f}s")

        if (step + 1) % SAMPLE_EVERY == 0:
            # Generate a sample from random noise for visual check
            ema.shadow.eval()
            with torch.no_grad():
                from dit_controlled_shared import ddim_reconstruction_eps
                noise_sample = torch.randn(1, 3, 64, 64, device=DEVICE)
                recon, _ = ddim_reconstruction_eps(ema.shadow, noise_sample, num_steps=50)
                # Save as PNG
                from torchvision.utils import save_image
                save_image((recon * 0.5 + 0.5).clamp(0, 1),
                           str(OUT_DIR / f"sample_step{step+1:06d}.png"))
            ema.shadow.train()

        if (step + 1) % CHECKPOINT_EVERY == 0:
            ckpt = {"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "step": step + 1, "loss_log": loss_log}
            torch.save(ckpt, OUT_DIR / f"checkpoint_{step+1:06d}.pt")

    # Save final
    torch.save({"model": ema.shadow.state_dict(), "loss_log": loss_log},
               OUT_DIR / "model_ema.pt")
    with open(OUT_DIR / "loss_log.json", "w") as f:
        json.dump(loss_log, f)

    total_t = time.time() - t0
    print(f"[eps] Done. Total time: {total_t:.0f}s ({total_t/3600:.1f}h)")
    print(f"[eps] Final loss: {sum(loss_log[-100:])/len(loss_log[-100:]):.6f}")


if __name__ == "__main__":
    main()
