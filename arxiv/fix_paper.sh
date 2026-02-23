#!/bin/bash
# Paper corrections based on Auditor feedback

cd /home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis

# 1. Fix Abstract - LLM vs RL confusion
sed -i 's/We observe that LLM-based agents often fail to track physical time across multi-turn interactions, completing tasks at strange times (e.g., submitting at 3:30 when deadline is 4:00)./While our work is motivated by observations of temporal grounding failures in LLM-based agents, this study uses RL agents (PPO/PRM) as a controlled experimental proxy to diagnose the underlying mechanisms./' paper.tex

# 2. Fix Introduction - LLM mentions
sed -i 's/Yet, we observe that LLM-based agents frequently exhibit \textit{temporal myopia}:/Yet, we observe that LLM-based agents frequently exhibit \textit{temporal myopia}, motivating our investigation. In this work, we focus on RL agents as a controlled experimental proxy./' paper.tex

# 3. Fix Keywords - add POMDP and Curriculum Learning
sed -i 's/\textbf{Keywords:} Temporal Grounding, Reward Hacking, Psychological Regret Model, RL Benchmark, LLM Agents/\textbf{Keywords:} Temporal Grounding, Reward Hacking, Psychological Regret Model, RL Benchmark, POMDP, Curriculum Learning/' paper.tex

# 4. Add PRM mathematical formula
cat >> /tmp/prm_formula.tex << 'EOF'
\begin{align*}
    R_{\text{total}}(s_t, a_t) = R_{\text{env}}(s_t, a_t) + \lambda \cdot \underbrace{\left(Q_{\text{target}}(s_t) - Q_{\text{policy}}(s_t, a_t)\right)}_{\text{Temporal Regret}}
\end{align*}
where $Q_{\text{target}}$ is approximated using an Oracle Q-value estimator trained on expert demonstrations, and $Q_{\text{policy}}$ is the agent's current policy estimate.
EOF

# 5. Add error bars note in discussion
cat >> /tmp/error_bars.tex << 'EOF'
\textbf{Statistical Uncertainty:} We report results averaged over 3 seeds with standard deviation. Due to compute constraints, experiments are limited to 1000 episodes; larger-scale validation is left for future work.
EOF

echo "Correction script prepared"
