# good
python main_actor.py --tensorboard-log-dir "./logs/actorqdn/maze4x4" \
    --width 4 \
    --height 4 \
    --max-episode-steps 16 \
    --total-time-steps 200000 
python main.py --tensorboard-log-dir "./logs/qdn/maze4x4" \
    --width 4 \
    --height 4 \
    --max-episode-steps 16 \
    --total-time-steps 200000 \
    --epsilon-decay 150000

# bad
python main_actor.py --tensorboard-log-dir "./logs/actorqdn/maze4x4" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 200000 
python main.py --tensorboard-log-dir "./logs/qdn/maze4x4" \
    --width 4 \
    --height 4 \
    --max-episode-steps-eval 16 \
    --total-time-steps 200000 \
    --epsilon-decay 150000

python main.py --tensorboard-log-dir "./logs/qdn/maze5x5" \
    --width 5 \
    --height 5 \
    --max-episode-steps-eval 25 \
    --total-time-steps 400000 \
    --epsilon-decay 200000