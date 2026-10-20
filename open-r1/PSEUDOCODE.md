# LUNAR Training: Pseudocode

**Input:** policy π_θ, reference π_ref, dataset P_Q of (q, y*),
group size G, clip ε, KL weight β, SCe weight λ_SCe, max length L_max

    for each training step:
        sample a batch of queries q ~ P_Q

        for each query q:
            # Rollout
            sample G responses {o_1, ..., o_G} ~ π_old(· | q)

            # Binary verifiable reward (S2)
            R_i ← 1 if final \boxed{} answer in o_i == y*, else 0

            # Self-certainty per response
            sc_t ← logsumexp(logits_t) − mean(logits_t)      # KL(U ‖ p_t) up to a constant
            u_i  ← mean of sc_t over the tokens of o_i

            # Advantages (one scalar per response, shared by all its tokens)
            A_R_i   ← R_i − mean_group(R)                                   # (S3)
            A_SCe_i ← (u_i − mean_group(u)) / (std_group(u) + 1e-4)         # (S4)
            s_i     ← 2·R_i − 1
            A_i     ← (1 − λ_SCe)·A_R_i + λ_SCe·s_i·tanh(A_SCe_i)           # (S5)

        # Per-token clipped surrogate + KL penalty (S6)
        for each response i, token t:
            r_it  ← exp(log π_θ(o_it) − log π_old(o_it))
            KL_it ← exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1
            ℓ_it  ← −min(r_it·A_i, clip(r_it, 1−ε, 1+ε)·A_i) + β·KL_it

        # 6. Dr.GRPO normalization (constant denominator)
        loss ← Σ_i Σ_t ℓ_it / (G · L_max)
        update θ by gradient descent on loss

> Note: we perform a single policy update per generation step, so π_old = π_θ,
> r_it = 1, and the clip is inactive; the objective reduces to the policy gradient
> A_i · ∇log π_θ(o_it) plus the KL penalty. The clipped form is kept for consistency
> with GRPO/Dr.GRPO.
