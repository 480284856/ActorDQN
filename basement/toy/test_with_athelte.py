import torch
import athlete
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

def run():
    env = gym.make("MyGridWorld-v0")
    # Initialize the agent, all hyperparameters have default values
    # which can be overridden with keyword arguments
    agent = athlete.make(
        algorithm_id="dqn",
        action_space=env.action_space,
        observation_space=env.observation_space["agent"],
        # seed=42, # optional,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    last_step = None
    counter = 0
    observation, env_info = env.reset(seed=42)
    # Inform the agent about the new episode and generate first action
    action, agent_info = agent.reset_step(observation["agent"], env_info)
    for step in range(int(1e6)):
        observation, reward, terminated, truncated, env_info = env.step(action)
        # Feed the agent new information to receive next action.
        # This will automatically perform updates as defined
        # (e.g. following update frequency, performing a warmup phase etc.).
        # agent_info contains internal information like the loss
        action, agent_info = agent.step(
                observation["agent"], reward, terminated, truncated, env_info
            )

        if terminated or truncated:
            observation, env_info = env.reset()
            action, agent_info = agent.reset_step(observation["agent"], env_info)
            # if last_step is not None:
            #     print(f"Episode finished after {step - last_step} steps")
            # last_step = step
            counter += 1

        if counter == 100:
            with torch.no_grad():
                state = torch.tensor([1., 0., 0.,     1., 0., 0.,     0., 1., 0.,     0., 1., 0.,     1., 0., 0.], dtype=torch.float32)
                action, _ = agent.evaluation_policy.act(
                    state.numpy()
                )
                overfitting = action != 0
            break

    env.close()
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
    plt.savefig("action_distribution-athlete.png")
    plt.show()

if __name__ == "__main__":
    main()