#!/usr/bin/env python3

# Read the paper
with open('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/paper.tex', 'r') as f:
    lines = f.readlines()

# Process line by line
for i in range(len(lines)):
    # Fix line 24 - Abstract
    if i == 23 and "We observe that LLM-based agents often fail to track physical time" in lines[i]:
        lines[i] = "While our work is motivated by observations of temporal grounding failures in LLM-based agents, this study uses \\textbf{RL agents (PPO/PRM)} as a controlled experimental proxy to diagnose the underlying mechanisms. We introduce \\textbf{ChronoEnv}, a time-critical RL benchmark that forces agents to balance task completion against the cost of querying external time.\n"
    
    # Fix line 30 - Introduction
    if i == 29 and "Yet, we observe that LLM-based agents frequently exhibit \\textit{temporal myopia}" in lines[i]:
        lines[i] = "Yet, we observe that LLM-based agents frequently exhibit \\textit{temporal myopia}, motivating our investigation. In this work, we focus on RL agents as a controlled experimental proxy to isolate temporal grounding failures.\n"
    
    # Fix line 29 - Keywords
    if i == 28 and "Keywords:" in lines[i]:
        lines[i] = "\\textbf{Keywords:} Temporal Grounding, Reward Hacking, Psychological Regret Model, RL Benchmark, POMDP, Curriculum Learning\n"
    
    # Fix line 50 - Contribution 3
    if i == 49 and "Psychological Regret Modeling for Temporal Grounding" in lines[i]:
        lines[i] = "    \\item \\textbf{Psychological Regret Modeling for Temporal Grounding:} We extend Psychological Regret Modeling (PRM) to include temporal regret signals, showing preliminary evidence of promise despite current limitations. Our analysis explains why Psychological Regret Modeling signals can be drowned out by the dominant \"submit immediately\" shortcut.\n"
    
    # Fix line 55 - Section reference
    if i == 54 and "Section~\\ref{sec:process_rewards}" in lines[i]:
        lines[i] = "The remainder of this paper is organized as follows: Section~\\ref{sec:related_work} reviews related work; Section~\\ref{sec:environment} describes ChronoEnv design; Section~\\ref{sec:reward_hacking} diagnoses the reward hacking phenomenon; Section~\\ref{sec:prm_direction} explores Psychological Regret Modeling as a solution; and Section~\\ref{sec:discussion} discusses implications and future work.\n"

# Write back
with open('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/paper.tex', 'w') as f:
    f.writelines(lines)

print("Basic corrections applied!")
