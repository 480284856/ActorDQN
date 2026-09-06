import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from env import GridWorldEnv
from ActorDQN.basement.agent import DQNAgent
from dqn import DQN

def designGrid():
    grid = [
                ["#", "#", " ",  " ",],
                ["S", " ", "#",  " ",],
                ["#", " ",  " ", 1,],
            ]
    return grid

def run():
    train_environment = gym.make("MyGridWorld-v0")

    # Initializing the neural network
    deep_q_network = DQN(
        observation_dimensions=train_environment.observation_space['agent'].shape[0],
        number_of_actions=train_environment.action_space.n,
    )

    # Initializing the optimizer and telling it which parameters to optimize,
    # what objective is being optimized is determined by the gradients these parameters receive
    optimizer = torch.optim.Adam(params=deep_q_network.parameters(), lr=1e-3)

    dqn_agent = DQNAgent(
        dqn=deep_q_network,
        optimizer=optimizer,
        epsilon=0.1,
        discount=0.99,
        action_space=train_environment.action_space,
    )

    counter = 0
    last_step = None
    observation, env_info = train_environment.reset(seed=42)
    action = dqn_agent.getAction(torch.tensor(observation['agent'], dtype=torch.float32).unsqueeze(0))
    for step in range(int(1e6)):
        next_observation, reward, terminated, truncated, info = train_environment.step(action.item())
        dqn_agent.update(
            state=torch.tensor(observation['agent'], dtype=torch.float32).unsqueeze(0),
            action=action,
            next_state=torch.tensor(next_observation['agent'], dtype=torch.float32).unsqueeze(0),
            reward=torch.tensor([[reward]], dtype=torch.float32),
            terminated=terminated,
        )
        if terminated or truncated:
            # if last_step is not None:
            #     print(f"Episode finished after {step - last_step} steps")
            last_step = step
            next_observation, env_info = train_environment.reset()
            counter += 1
        observation = next_observation
        action = dqn_agent.getAction(torch.tensor(observation['agent'], dtype=torch.float32).unsqueeze(0))

        if counter == 100 :
            with torch.no_grad():
                state = torch.tensor([[1., 0., 0.,     1., 0., 0.,     0., 1., 0.,     0., 1., 0.,     1., 0., 0.]], dtype=torch.float32)
                # logits = dqn_agent.dqn(state)
                # for logit,act in zip(logits[0], ["up", "right", "down", "left"]):
                #     print(f"Score for action {act}: {logit.item()}")
                # overfitting = logits.max().item() > logits[0][0].item()

                action = dqn_agent.getPolicy(state).item()
                overfitting = action != 0
            break
    
    train_environment.close()
    return overfitting,action

def main(times=100):
    gym.register(
        id="MyGridWorld-v0",
        entry_point="env:GridWorldEnv",
    )

    overfitting_count = 0
    action_stats = dict(
        up=0,
        right=0,
        down=0,
        left=0
    )
    action2text = {0: "up", 1: "right", 2: "down", 3: "left"}

    bar = tqdm(range(times), desc="Running experiments")
    for i in bar:
        overfitting,action = run()
        if overfitting:
            overfitting_count += 1
            bar.set_postfix({"Overfitting rate": "{:.2f}%".format(overfitting_count / (i + 1) * 100)})

        action_stats[action2text[action]] += 1
    print("Overfitting rate: {:.2f}%".format(overfitting_count / times * 100))

    # Plot action statistics
    plt.bar(action_stats.keys(), action_stats.values())
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Action Distribution")
    plt.savefig("action_distribution.png")
    plt.show()

if __name__ == "__main__":
    main()
