> GRPO
>
> Shreeyam Bangera and Ankur Dahiya
>
> April 2025
>
> 1 Introduction
>
> LLMs or Large Language Models are the talk of the hour, with everyone
> using ChatGPT, Copilot, Claude, Gemini and other famous LLMs for
> generating human like text, generating code, research and many other
> things. One may ask as to how they are able to get access and
> knowledge of all this worldly knowledge? This is done by a process
> called ”pretraining” where they are fed huge corpus of text from the
> internet, various books, articles, forums and any other publicly
> available data source with the sole purpose of predicting the next
> probable word while generating.
>
> But pretraining on its own is not enough for getting a model that
> gives optimized outputs aligning itself to the user’s prompts. This is
> because pretraining is like giving the model a lot of general
> knowledge, i.e. not providing it with specific skills for a particular
> task. Fine-tuning solves this particular problem as it helps the model
> to learn various patterns by training it a bit more on the particular
> task related dataset and tuning the model’s weights to the task
> allowing the model to apply the knowledge learnt(during pretraining)
> to the task given by the user.
>
> But even after all this, the model might generate texts or information
> that might not align with human values, for example: if any student
> asks the way to deal with increasing competition in college exams,
> then instead of helping him/her to improve the studies, it may suggest
> to sabotage other students. This is the problem of AI disalignment.
> Text generated like this may be toxic, biased on various premises
> cause harm and even at times provide faulty and incorrect responses.
>
> AI alignment can be done by various methods but the most famous and
> widely used method is that of Reinforcement Learning (RL). It usually
> uses a reward model for teaching the AI to align itself to human
> measures and optimize it correctly, giving it reward for each correct
> generation and a negative reward for each incorrect one. Other than
> using the reward model, Reinforcement Learning with Human Feedback
> (RLHF) has been an impactful technique for training modern language
> models such as ChatGPT (like when you are asked your preferred
> response by ChatGPT while is generates 2 responses). In RLHF, the
> model is fine-tuned based on scores/labels provided by humans via
> various
>
> 1
>
> approaches like PPO, DPO and GRPO which would be expanded on in this
> blog.
>
> 2 Prerequisites Crash Course (LLMs and RL)
>
> A\) Large Language Models
>
> Word2Vec learns word representations by maximizing the probability of
> context words given a target word using the Skip-Gram model:
>
> X
>
> max logP(c \| w) where
>
> (w,c)∈D

⃗vc·⃗vw P(c \| w) = c′∈V e⃗vc′·⃗vw

> LLMs model the probability of the next word in a sequence, given the
> previous ones. Mathematically:
>
> n
>
> P(w1,w2,...,wn) = P(wt \| w1,...,wt−1) t=1
>
> The model’s parameters are optimized by minimizing the negative
> log-likelihood loss over a large corpus using stochastic gradient
> descent (SGD) or its variants:
>
> n
>
> L = − logP(wt \| w\<t) t=1
>
> This is pretty much all one needs to know about LLMs to understand
> RLHF.
>
> B\) Reinforcement Learning Crash Course
>
> State: The state st represents the environment’s configuration at time
> t.
>
> st ∈ S
>
> Agent: The agent observes the state and selects actions to maximize
> cumulative reward.
>
> Reward: The reward rt is the feedback received after taking action at
> in state st.
>
> rt = R(st,at)
>
> Policy: The policy π defines a probability distribution over actions
> given a state.
>
> π(a \| s) = P(at = a \| st = s)
>
> 2
>
> 3 An overview of the Classic Re-inforcement Learn-ing from Human
> Feedback
>
> The agent is the language model, the state is the query and the token
> output of the model, the reward is the ranking of the output to the
> queries given by the human, and the policy is the parameters of the
> model.
>
> Lemma 1 (Stochastic Transitions): We model the next state as
> stochastic, i.e.,
>
> st+1 ∼ P(st+1 \| st,at)
>
> Trajectory Probability: The probability of a trajectory τ under policy
> π is given by:
>
> T−1
>
> P(τ \| π) = ρ0(s0) P(st+1 \| st,at)π(at \| st) t=0
>
> Lemma 2 (Discounted Rewards): We discount rewards since immediate
> rewards are preferred:
>
> ∞
>
> Gt = γkrt+k, where γ ∈ \[0,1)
>
> k=0
>
> Trajectories are basically a series of states and actions. The goal is
> to select a policy that maximizes the expected return:
>
> π∗ = argmaxJ(π)
>
> The function J(π) represents this expected return. It is calculated by
> averaging the total rewards R(τ) received over all possible
> trajectories τ, weighted by how likely each trajectory is under the
> policy π. In other words, the better the policy, the more likely it is
> to generate high-reward trajectories:
>
> Z
>
> J(π) = P(τ \| π)R(τ) = Eτ∼π\[R(τ)\]. τ
>
> To maximize the expected return in LLMs where the policy is
> parameterized by θ, we use gradient ascent as follows:
>
> θk+1 = θk +α∇θJ(πθ)θk
>
> Now the goal is to find an expression of ’J’ and compute it. Of course
> it is computationally impossible to calculate the return over all
> possible trajectories. Therefore we approximate it as:
>
> X X
>
> gˆ = ∇θ logπθ(at\|st)R(τ)
>
> τ∈D t=0
>
> 3
>
> The original gradient estimator has pretty high variance because it
> dumps the entire return R(τ) on every action taken during the episode,
> even if that action had little to do with the final reward. This ends
> up making learning noisy and unstable. Now, thanks to the Central
> Limit Theorem, we know that as we collect more data, our estimate
> should eventually converge to the true gradient—but high variance
> means we need a lot of data to get there.
>
> Todealwiththis, weswitchtousingthe advantage function, definedas
> Aπ(st,at) = Qπ(st,at)−Vπ(st). It tells us how much better (or worse)
> an action is compared to what the agent would normally do in that
> state. So instead of crediting every
>
> action equally with the total return, we adjust for how good the
> action actually was. This gives us a new and improved gradient
> estimator:
>
> X X
>
> gˆ = ∇θ logπθ(at\|st)A (st,at),
>
> τ∈D t=0
>
> which is still unbiased but way less noisy, making training smoother
> and more eficient.
>
> 3.1 Advantage Function and Its Estimation
>
> The advantage function quantifies the relative benefit of taking a
> particular action in a given state, compared to the average
> performance of the policy from that state. It is defined as:
>
> Aπ(s,a) = Qπ(s,a)−Vπ(s)
>
> • Qπ(s,a) is the expected return when the agent starts in state s,
> takes action a, and then follows the policy π thereafter:
>
> " ∞ \# Qπ(s,a) = Eπ γtrt s0 = s,a0 = a
>
> t=0
>
> • Vπ(s) is the expected return when the agent starts in state s and
> follows the policy π from the beginning, with the first action also
> sampled from π: " ∞ \#
>
> Vπ(s) = Ea∼π(·\|s) \[Qπ(s,a)\] = Eπ γtrt s0 = s
>
> t=0
>
> Intuitively, Aπ(s,a) measures how much better (or worse) an action a
> is than what the policy would typically do in state s.
>
> 3.1.1 Monte Carlo Estimation
>
> Monte Carlo (MC) methods estimate returns by sampling entire
> trajectories (episodes) and using the observed total return from a
> state (or state-action pair) as an unbiased estimator of expected
> return.
>
> 4
>
> Let Gt denote the total return starting from time t:
>
> T−t−1
>
> Gt = γlrt+l
>
> l=0 Then, the MC estimate of the advantage is:
>
> Aπ(st,at) = Gt −Vπ(st)
>
> where Vπ(st) is estimated as the average of Gt’s over all times st is
> visited.
>
> Intuition: This approach directly compares what actually happened (via
> the observed return) to what the policy would expect from that state
> on average.
>
> 3.1.2 Temporal-Difference (TD) Estimation
>
> TD methods bootstrap from the value of the next state to estimate
> returns, which allows for online and incremental learning. The 1-step
> TD error is defined as:
>
> δt = rt +γVπ(st+1) −Vπ(st)
>
> This TD error serves as a low-variance, biased estimator of the
> advantage:
>
> Aπ(st,at) ≈ δt
>
> Intuition: Instead of waiting to see how the episode ends, TD uses the
> im-mediate reward and the estimated future return to approximate the
> advantage.
>
> 3.1.3 Generalized Advantage Estimation (GAE)
>
> Generalized Advantage Estimation (GAE) provides a principled way to
> interpo-late between the high-variance MC estimator and the high-bias
> TD estimator. It does so by taking an exponentially weighted sum of
> k-step TD errors.
>
> Let δt be the 1-step TD error as before. Then, GAE is defined as:
>
> ∞ AGAE(γ,λ) = (γλ)lδt+l
>
> l=0
>
> For finite trajectories, this is truncated at the episode end.
> Alternatively, it can be computed eficiently in reverse via the
> recursion:
>
> At = δt +γλAt+1
>
> Parameters:
>
> • γ: discount factor, controlling horizon of future rewards.
>
> • λ: GAE parameter, controlling the trade-off between bias and
> variance.
>
> 5
>
> Interpretation:
>
> • When λ = 0, GAE reduces to 1-step TD: fast and low variance but
> biased.
>
> • When λ = 1, GAE becomes equivalent to the MC estimate: unbiased but
> high variance.
>
> • Intermediate λ values allow tuning the bias-variance tradeoff.
>
> 3.1.4 Summary Table
>
> 3.2 Failure Modes of Vanilla Policy Gradient (VPG)
>
> The Vanilla Policy Gradient (VPG) method attempts to maximize the
> expected return by directly optimizing:
>
> " T \# J(θ) = Eπθ γtrt
>
> t=0
>
> Using the policy gradient theorem, the gradient is:
>
> h i ∇θJ(θ) = Eπθ ∇θ logπθ(at\|st) ·At
>
> The VPG loss is defined as:
>
> h i LVPG(θ) = Et logπθ(at\|st)·At
>
> Mathematical Issues:
>
> 1\. Unconstrained Update Magnitude: The policy πθ is updated without
> any mechanism to control how far it moves from the original policy
> πθold. A single large gradient step can lead to:
>
> πθ(a\|s) ≪ πθold(a\|s) even if A(s,a) \> 0
>
> This destroys the probability of good actions and leads to performance
> collapse.
>
> 2\. Distribution Mismatch: The trajectories are sampled from πθold,
> but the gradient is applied to πθ. The advantage estimates At are only
> valid under πθold, and large updates make them invalid for πθ.
>
> 3\. High Variance and Instability: Without any regularization or trust
> region, the updates are sensitive to noise in advantage estimates,
> leading to high variance and poor convergence.
>
> 6
>
> 3.3 Derivation of the PPO Objective
>
> To address these issues, Proximal Policy Optimization (PPO) introduces
> a clipped surrogate objective that discourages large policy updates.
>
> Step 1: Define the Probability Ratio Let πθold be the current policy
> and πθ the new policy. Define the probability ratio:
>
> πθ(at\|st) t πθold(at\|st)
>
> Step 2: Surrogate Objective We want to improve the policy by
> maximizing the expected advantage weighted by this ratio:
>
> h i LCPI(θ) = Et rt(θ)·At
>
> This is the basis for Conservative Policy Iteration. However, this
> still allows for large updates if rt(θ) becomes too large or too
> small.
>
> Step 3: Clipped Objective PPO introduces a clipped surrogate loss:
>
> h i LCLIP(θ) = Et min rt(θ) ·At, clip(rt(θ),1−ϵ,1+ ϵ)·At
>
> Interpretation:
>
> • If At \> 0: the objective increases with rt, but is capped at 1+ ϵ.
>
> • If At \< 0: the objective decreases with rt, but is floored at 1 −ϵ.
>
> • This prevents the optimizer from moving πθ too far from πθold.
>
> Final PPO Objective: In practice, the complete PPO loss also includes
> a value function loss and an entropy bonus:
>
> LPPO(θ) = Et LCLIP(θ)−c1 ·(Vθ(st)−Vtarget)2 +c2 ·H\[πθ\](st) Where:
>
> • c1 weights the value function MSE loss
>
> • c2 weights the entropy bonus
>
> • H\[π\] encourages exploration by maximizing policy entropy
>
> 4 Direct preference optimization
>
> 5 Group Relative Policy Optimization
>
> 7
