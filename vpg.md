# Vanilla policy gradient pseudocode

## Simplest:

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
    policy_net.update(grad(mean(losses)))    
</pre>

## Reward-to-go: (less variance)

<pre>
...
        # calc loss (negative of policy gradient)
        reward_to_go = sum(rewards)
        total_loss = 0
        for i in range(all_actions_probs):
            total_loss += log(all_actions_probs[i]) * reward_to_go
            reward_to_go -= rewards[i]
        losses.append(-total_loss) # finite undiscounted
...
</pre>

## Value function baseline: (even less variance)

<pre>
avg_ret = 0
policy_net = policy_net()
<b>value_net = value_net()</b>

while True: # termination criteria?
    done = false
    state = env.init()
    losses = []

    # sample n tranjectories
    for _ in range(num_trajectory_samples):
        all_action_probs = []
        rewards = []
        <b>states = []</b>

        while not done:
            
            action_probs = policy_net(state)
            action_index = sample(action_probs) # sample as a probability distribuition
            <b>states.append(state)</b>

            reward, state, done = env.step(actions_probs[action_index])
            rewards.append(reward)
            all_action_probs.append(action_probs[action_index])


        # calc loss (negative of policy gradient)
        <b>rewards_to_go = []</b>

        reward_to_go = sum(rewards)
        total_loss = 0
        for i in range(all_actions_probs):
            <b>rewards_to_go.append(reward_to_go)</b>
            total_loss += log(all_actions_probs[i]) * (reward_to_go - value_net(states[i]))
            reward_to_go -= rewards[i]
        losses.append(-total_loss) # finite undiscounted

        <b>
        # calc value loss
        value_preds = [(value_net(states[i]) for i in range(states)]
        value_loss = mean(squared(value_preds - rewards_to_go))
        </b>


    # backprop loss (estimate of policy gradient)
    policy_net.update(grad(mean(losses)))  

    <b>
    # update value network
    value_net.update(grad(value_loss))  
    </b>
</pre>

## Advantage function with GAE-lambda: (best)
TODO
(https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8)