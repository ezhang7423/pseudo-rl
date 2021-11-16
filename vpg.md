# Vanilla policy gradient pseudocode


Simplest:

<pre>
avg_ret = 0
policy_net = policy_net()

while True: # termination criteria?
    done = false
    state = env.init()
    losses = []

    # sample n tranjectories
    for _ in range(num_trajectory_samples):
        all_action_probs = []
        rewards = []

        while not done:
            action_probs = policy_net(state)
            action_index = sample(action_probs) # sample as a probability distribuition

            reward, state, done = env.step(actions_probs[action_index])
            rewards.append(reward)
            all_action_probs.append(action_probs[action_index])
    
        # calc loss (negative of policy gradient)
        <b>losses.append(-sum(rewards) * sum(log(all_actions_probs))) # finite undiscounted</b>

    # backprop loss (estimate of policy gradient)
    policy_net.update(grad(sum(losses)))    
</pre>