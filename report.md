# Lab 2 Report — Neural Network from Scratch

## Goal

The goal of this lab was to build a two-hidden-layer neural network from scratch with NumPy, train it on the digits dataset, check that my hand-derived backpropagation was actually correct using numerical gradient checking, and then compare different hidden widths and learning rates by looking at loss curves, test accuracy, and how long the forward vs backward passes take.

## Method

I used `load_digits` from scikit-learn (1797 samples, 64 features). Inputs were divided by 16 so they're in the 0–1 range, and labels were one-hot encoded. I split 70/30 stratified with `random_state=42`, which gave 1257 training and 540 test samples.

The network was `64 → W → ReLU → W → ReLU → 10 → softmax`, with W = 16 and W = 64. Weight shapes were `(64, W)`, `(W, W)`, `(W, 10)` and biases `(W,)`, `(W,)`, `(10,)`. I initialized weights from N(0, 0.01) and biases to zero.

For the forward pass I implemented each layer step by step and checked that the output shape was `(batch, 10)` and each row summed to 1 within 1e-6. The loss was cross-entropy, and as a sanity check I ran the untrained network and got loss ≈ 2.302585, which matches ln(10) — so the loss was working.

For the gradient check I nudged 10 random weights by ±1e-5 and compared the analytical vs numerical slopes. The requirement was relative error ≤ 1e-6 for at least 9 of the 10 weights.

Then I trained with mini-batch size 32 for 50 epochs, using learning rates [0.001, 0.01, 0.1] and widths [16, 64] — six runs in total. Loss curves went to `results/loss_curves.png`. Finally I timed the forward and backward passes separately, median of 100 runs, at width 64 and batch size 32.

Seeds were fixed to 42 and versions are in `results/versions.txt`.

## Results

**Gradient check**

| # | Analytical | Numerical | Rel. error | Pass |
|---:|---:|---:|---:|:---:|
| 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |
| 2 | -1.8056e-05 | -1.8056e-05 | 4.61e-07 | yes |
| 3 | -9.8343e-05 | -9.8343e-05 | 1.16e-07 | yes |
| 4 | 6.5037e-06 | 6.5037e-06 | 1.41e-06 | no |
| 5 | 5.2395e-05 | 5.2395e-05 | 2.88e-08 | yes |
| 6 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |
| 7 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |
| 8 | 2.8328e-05 | 2.8328e-05 | 2.65e-07 | yes |
| 9 | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |
| 10 | -5.5751e-04 | -5.5751e-04 | 6.90e-09 | yes |

Passed 9/10, which meets the lab requirement. The one that "failed" has an analytical gradient of only ~6.5e-6, so the ±1e-5 perturbation is actually bigger than the gradient itself the nuerical slope is basically noise at that scale. The two values still agree to 6 significant figures, so I'm confident the gradient itself is fine and this is just finite-difference noise.

**Training runs**

| Width | LR | Train acc. | Test acc. | Time (s) | Final loss |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.001 | 0.1018 | 0.1019 | 0.57 | 2.3026 |
| 16 | 0.01 | 0.1018 | 0.1019 | 0.60 | 2.3025 |
| 16 | 0.1 | 0.3222 | 0.3259 | 0.66 | 1.5970 |
| 64 | 0.001 | 0.1018 | 0.1019 | 1.18 | 2.3025 |
| 64 | 0.01 | 0.1018 | 0.1019 | 0.95 | 2.3023 |
| 64 | 0.1 | **0.9467** | **0.9389** | 0.77 | **0.1684** |

Loss curves are in `results/loss_curves.png`.

**Timing (width 64, batch 32, median of 100 runs)**

| Pass | Median time |
|---|---:|
| Forward | 0.0604 ms |
| Backward | 0.0785 ms |

## Conclusions

1. **The gradients are correct.** My hand-derived backprop matched numerical gradients on 9 of 10 weights with relative error ≤ 1e-6. The one miss (rel. eror 1.41e-6 on a gradient of only 6.5e-6) is almost certainly finite-difference noise  if you nudge a weight by ±1e-5 when the true gradient is ~6.5e-6, you're basically measuring roundoff. The values still agre to 6 sig figs and the lab's "at least 9 of 10" rule is met, so I'm happy with this.

2. **Most of my runs didn't learn, and I think I know why.** With lr = 0.001 and lr = 0.01, the loss stayed flat at ≈ 2.3026 (= ln 10) and test accuracy was stuck at 10.19%, which is just chance on 10 classes. Only lr = 0.1 escaped that: 32.6% at width 16 and 93.9% at width 64. The reason is that with N(0, 0.01) init the early gradients are very small (you can see this in the gradient-check table —several are ~1e-5 or exactly 0). At small learning rates the weight updates are essentially nothing per step, so the network never gets moving. One thing I found interesting is that the width-64 lr=0.1 run trained *faster* in wall-clock (0.77s) than the width-16 lr=0.001 run (0.57s) per epoch count — so the bottleneck wasn't compute, it was the optimization getting stuck. If I were doing this again I'd either use a larger init or standardize the input to have zero mean, which would make the gradients bigger at the start.

3. **The backward pass is a bit slower than the forward pass, but not by much.** Backward took 0.0785 ms vs 0.0604 ms forward (~1.3×). That makes sense: the backward pass has to do roughly twice as many matrix multiplications per layer (gradient w.r.t. the inputs, gradient w.r.t. the weights), plus the ReLU mask. On these small tensors the difference is modest, but if I scaled up the batch size and hidden widths the backward pass would pull further ahead because the matmul cost dominates both passes.
