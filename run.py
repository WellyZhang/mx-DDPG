from ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from policies import DeterministicMLPPolicy
from qfuncs import ContinuousMLPQ
from strategies import OUStrategy
from utils import SEED

# set environment, policy, qfunc, strategy

env = normalize(CartpoleEnv())

policy = DeterministicMLPPolicy(env.spec)
qfunc = ContinuousMLPQ(env.spec)
strategy = OUStrategy(env.spec)

# set the training algorithm and train

algo = DDPG(
    env=env,
    policy=policy,
    qfunc=qfunc,
    strategy=strategy,
    ctx=mx.gpu(0),
    max_path_length=100,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=1000,
    discount=0.99,
    scale_reward=0.01,
    qfunc_lr=1e-3,
    policy_lr=1e-4,
    seed=SEED)

algo.train()