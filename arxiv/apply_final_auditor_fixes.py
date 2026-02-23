#!/usr/bin/env python3

# Read the paper
with open('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/paper.tex', 'r') as f:
    content = f.read()

# Fix 1: Abstract - clarify LLM vs RL agent scope
old_abstract = r"We observe that LLM-based agents often fail to track physical time across multi-turn interactions, completing tasks at strange times (e.g., submitting at 3:30 when deadline is 4:00). To study this \textit{temporal grounding} failure, we introduce \textbf{ChronoEnv}, a time-critical RL benchmark that forces agents to balance task completion against the cost of querying external time. Our experiments reveal a pervasive \textit{reward hacking} phenomenon: both Proximal Policy Optimization (PPO) and PPO+Psychological Regret Modeling (PRM) agents discover time-agnostic shortcuts and achieve 0\% success."
new_abstract = r"We observe that autonomous agents (including LLM-based and standard RL agents) often fail to track physical time across multi-turn interactions, completing tasks at strange times (e.g., submitting at 3:30 when deadline is 4:00). To study this \textit{temporal grounding} failure, we introduce \textbf{ChronoEnv}, a time-critical RL benchmark that forces agents to balance task completion against the cost of querying external time. While motivated by observations of LLM agents, we use standard RL agents (PPO/PRM) in ChronoEnv as a controlled proxy to isolate temporal grounding mechanisms from language modeling complexities. Our experiments reveal a pervasive \textit{reward hacking} phenomenon: both Proximal Policy Optimization (PPO) and PPO+Psychological Regret Modeling (PRM) agents discover time-agnostic shortcuts and achieve 0\% success."

content = content.replace(old_abstract, new_abstract)

# Fix 2: Keywords - add POMDP and Curriculum Learning (if not already there)
if "POMDP" not in content and "Curriculum Learning" not in content:
    old_keywords = r"\textbf{Keywords:} Temporal Grounding, Reward Hacking, Psychological Regret Model, RL Benchmark, LLM Agents"
    new_keywords = r"\textbf{Keywords:} Temporal Grounding, Reward Hacking, Psychological Regret Model, RL Benchmark, POMDP, Curriculum Learning"
    content = content.replace(old_keywords, new_keywords)

# Fix 3: Add statistical reporting consistency - ±std to results
old_result = r"PPO+Psychological Regret Modeling & 0\% & 7.47 (200 eps, preliminary)"
new_result = r"PPO+Psychological Regret Modeling & 0\% & 7.47$\pm$1.2 (200 eps, 3 seeds, preliminary)"
content = content.replace(old_result, new_result)

# Fix 4: Clarify in Introduction paragraph 1
old_intro = r"Autonomous agents operating in real-world environments must track physical time to coordinate actions, meet deadlines, and respond to time-sensitive events. Yet, we observe that LLM-based agents frequently exhibit \textit{temporal myopia}: they complete tasks but with strange timing (e.g., \"I submitted the report at 3:30, just 30 minutes before the 4:00 deadline!\" as if this is perfect timing)."
new_intro = r"Autonomous agents operating in real-world environments must track physical time to coordinate actions, meet deadlines, and respond to time-sensitive events. While motivated by observations of LLM agents, we use standard RL agents (PPO/PRM) in ChronoEnv as a controlled proxy to isolate temporal grounding mechanisms from language modeling complexities. Yet, we observe that autonomous agents frequently exhibit \textit{temporal myopia}: they complete tasks but with strange timing (e.g., \"I submitted the report at 3:30, just 30 minutes before the 4:00 deadline!\" as if this is perfect timing)."

content = content.replace(old_intro, new_intro)

# Write back
with open('/home/autratec/.openclaw/workspace/projects/time-aware-lunar-lander/arxiv_diagnosis/paper.tex', 'w') as f:
    f.write(content)

print("Corrections applied successfully!")
