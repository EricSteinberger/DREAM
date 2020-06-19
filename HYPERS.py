from PokerRL.eval.rl_br import RLBRArgs

SDCFR_LEDUC_TRAVERSALS_ES = 346  # int(900 / 2.6)
SDCFR_LEDUC_TRAVERSALS_OS = 900
SDCFR_LEDUC_BATCHES = 3000
SDCFR_LEDUC_PERIOD = 1
SDCFR_LEDUC_BASELINE_BATCHES = 1000

OS_EPS = 0.5

SDCFR_FHP_TRAVERSALS_ES = 10000
SDCFR_FHP_TRAVERSALS_OS = 50000
SDCFR_FHP_BATCHES = 4000
SDCFR_FHP_BATCH_SIZE = 10000
SDCFR_FHP_BASELINE_BATCHES = 1000

N_LA_FHP_CFR = 12
N_LA_FHP_NFSP = 10

N_LA_RLBR_PER_PLAYER = 6  # per player, so count *2

RL_BR_FREQ_CFR = 60

DIST_RLBR_ARGS_games = RLBRArgs(
    n_brs_to_train=1,  # TODO 3? 5?
    DISTRIBUTED=True,
    n_las_per_player=N_LA_RLBR_PER_PLAYER,
    n_iterations=50000,
    n_hands_each_seat_per_la=int(1e6 / N_LA_RLBR_PER_PLAYER * 2),
    target_net_update_freq=300,
    buffer_size=int(4e5 / N_LA_RLBR_PER_PLAYER),
    batch_size=int(2048 / N_LA_RLBR_PER_PLAYER),
    play_n_steps_per_iter_per_la=int(512 / N_LA_RLBR_PER_PLAYER),
)
