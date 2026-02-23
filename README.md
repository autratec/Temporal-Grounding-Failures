# Why Agents Don't Watch the Clock

**Diagnosing Temporal Grounding Failures in Reinforcement Learning**

## 📄 Paper

**arXiv:** [2602.xxxxx](https://arxiv.org/abs/2602.xxxxx)

**Citation:**
```bibtex
@article{temporal-grounding-failures,
  title={Why Agents Don't Watch the Clock: Diagnosing Temporal Grounding Failures in Reinforcement Learning},
  author={Xu, Zhe},
  journal={arXiv preprint arXiv:2602.xxxxx},
  year={2026}
}
```

## 🔬 Core Finding

We diagnose a fundamental challenge in training time-aware agents: standard RL reward structures create incentives for agents to discover *time-agnostic shortcuts* that avoid the difficult cognitive task of time estimation.

## 📊 Key Results

- **ChronoEnv:** New benchmark forcing time awareness
- **PPO Success:** 0% (1000 episodes)
- **PPO+PRM Success:** 0% (1000 episodes)

## 📁 Repository

- `arxiv/` - Paper LaTeX, PDF, references, figures
- `code/` - ChronoEnv implementation, training scripts
- `results/` - Pre-computed results

## 👤 Author

**Xu, Zhe**  
Email: jeff_z_xu@yahoo.com

## 📜 License

MIT License
