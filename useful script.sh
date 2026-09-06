python main_actor.py --tensorboard-log-dir "./logs/actorqdn/debug/maze4x4-low_buffer" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 200000 



with torch.no_grad():
    q_values = self.actor_dqn_network(batch.states)
    action_probs1 = torch.softmax(q_values, dim=1)
with torch.no_grad():
    q_values = self.actor_dqn_network(batch.states)
    action_probs2 = torch.softmax(q_values, dim=1)
for p1,p2 in zip(action_probs1, action_probs2):
    print(p1)
    print(p2)
    print()