# Mixture-of-Experts Antibody Language Models

Code for *Evaluating Expert Specialization in Mixture-of-Experts Antibody Language Models*, published in the [ICLR 2026 FM4Science Workshop](https://openreview.net/forum?id=laLWlka9kN).

We explore the use of sparse mixture-of-experts (MoE) architectures for antibody language models (AbLMs). Through a systematic comparison of established MoE routing strategies (Top-K, Top-P, and Expert Choice) we observe that token-choice routing consistently outperforms expert-choice routing. Building on this finding, we refine token-choice routing by omitting padding tokens from routing decisions, enabling more effective variable-length pre-training. Our large-scale Top-2 MoE model, BALM-MoE, is trained on both paired and unpaired antibody sequences and surpasses a parameter-matched dense model in performance.

BALM-MoE model weights are available on [Hugging Face](https://huggingface.co/collections/brineylab/moe-paper).

## Dependencies

```bash
pip install git+https://github.com/brineylab/BALM.git
pip install git+https://github.com/smburbach/AbLM-Eval.git
```

## Code

Model architecture code is available in the [BALM repo](https://github.com/brineylab/BALM).

**`model-training/`**
- `01_pilot/` — initial router comparison experiments (45M active params, unpaired only)
- `02_pad-ablations/` — router masking ablations (45M active params, unpaired only)
- `03_final/` — large-scale models (200M active params, mixed paired + unpaired data)

**`model-eval/`** — eval scripts organized by manuscript figure

## Citation
```
Burbach, S.M., Spandau, S., Hurtado, J., & Briney, B. (2026). Evaluating Expert Specialization in Mixture-of-Experts Antibody Language Models. ICLR 2026 Workshop on Foundation Models for Science: Real-World Impact and Science-First Design. https://openreview.net/forum?id=laLWlka9kN
```
