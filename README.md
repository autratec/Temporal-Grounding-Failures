# Temporal Grounding Failures in Reinforcement Learning

This repository contains the implementation and experiments for the paper:

**"Why Agents Don't Watch the Clock: Diagnosing Temporal Grounding Failures in Reinforcement Learning"**

## Repository Structure

```
Temporal-Grounding-Failures/
├── README.md                  # This file
├── arxiv/                     # arXiv submission materials
│   ├── paper.tex             # LaTeX source
│   ├── paper.pdf             # Compiled PDF
│   ├── references.bib        # BibTeX references
│   └── figures/              # Figures (PDF/PNG)
├── code/                      # Environment and training code
│   ├── chrono_env_v2.py      # ChronoEnv implementation
│   ├── train_complete.py     # Training scripts
│   ├── run_sensitivity.py    # Query cost sensitivity analysis
│   └── temporal_debugger.py  # Agent visualization tool
└── results/                   # Pre-computed results
```

## Abstract

We observe that autonomous agents (including LLM-based and standard RL agents) often fail to track physical time across multi-turn interactions, completing tasks at strange times (e.g., submitting at 3:30 when deadline is 4:00). To study this *temporal grounding* failure, we introduce **ChronoEnv**, a time-critical RL benchmark that forces agents to balance task completion against the cost of querying external time. While motivated by observations of LLM agents, we use standard RL agents (PPO/PRM) in ChronoEnv as a controlled proxy to isolate temporal grounding mechanisms from language modeling complexities. Our experiments reveal a pervasive *reward hacking* phenomenon: both Proximal Policy Optimization (PPO) and PPO+Psychological Regret Modeling (PRM) agents discover time-agnostic shortcuts and achieve 0% success. We analyze root causes (sparse rewards, hard exploration, misaligned incentives) and demonstrate that Psychological Regret Modeling provides some guidance but is insufficient.

**Keywords:** Temporal Grounding, Reward Hacking, Psychological Regret Model, RL Benchmark, POMDP, Curriculum Learning

## Paper Information

- **Paper:** [arXiv:2602.xxxxx](https://arxiv.org/abs/2602.xxxxx) (TBA)
- **PRM Foundation:** [arXiv:2602.03171](https://arxiv.org/abs/2602.03171) (Xu et al., 2026)

## Authors

- **Xu, Zhe** - Individual researcher from Singapore
  - Email: jeff_z_xu@yahoo.com

## Citation

If you use ChronoEnv or find this work useful, please cite:

```bibtex
@article{temporal-grounding-failures,
  title={Why Agents Don't Watch the Clock: Diagnosing Temporal Grounding Failures in Reinforcement Learning},
  author={Xu, Zhe},
  journal={arXiv preprint arXiv:2602.xxxxx},
  year={2026}
}
```

## License

This project is released under the MIT License.
