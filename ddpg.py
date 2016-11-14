from replay_mem import ReplayMem
import mxnet as mx
import numpy as np


class DDPG(object):

    def __init__(
        self,
        env,
        policy,
        qfunc,
        strategy,
        ctx=mx.gpu(0),
        batch_size=32,
        n_epochs=1000,
        epoch_length=1000,
        memory_size=1000000,
        memory_start_size=1000,
        discount=0.99,
        max_path_length=1000,
        qfunc_weight_decay=0.0,
        qfunc_update="adam",
        qfunc_lr=1e-4,
        policy_weight_decay=0.0,
        policy_update="adam",
        policy_lr=1e-4,
        soft_target_tau=1e-3,
        n_updates_per_sample=1,
        include_horizon_terminal=False,
        seed=12345):

        mx.random.seed(seed)

        self.env = env
        self.ctx = ctx
        self.policy = policy
        self.qfunc = qfunc
        self.strategy = strategy
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.memory_size = memory_size
        self.memory_start_size = memory_start_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.qfunc_weight_decay = qfunc_weight_decay
        self.qfunc_update = qfunc_update
        self.qfunc_lr = qfunc_lr
        self.policy_weight_decay = policy_weight_decay
        self.policy_update = policy_update
        self.policy_lr = policy_lr
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal = include_horizon_terminal

        self.init_net()

        # logging

        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.strategy_path_returns = []


    def init_net(self):

        # qfunc init

        qfunc_init = mx.initializer.Normal()
        loss_symbols = self.qfunc.get_loss_symbols()
        qval_sym = loss_symbols["qval"]
        yval_sym = loss_symbols["yval"]
        weight_sym_dict = loss_symbols["weight"]
        reg = self.qfunc_weight_decay * 0.5 * mx.symbol.sum(
            [mx.symbol.square(arr) for name, arr in weight_sym_dict.items()])
        loss = 1.0 / self.batch_size * mx.symbol.sum(
            mx.symbol.square(qval_sym - yval_sym)) 
        qfunc_loss = loss + reg
        qfunc_updater = mx.optimizer.get_updater(
            mx.optimizer.create(self.qfunc_updater,
                                learning_rate=self.qfunc_lr))
        self.qfunc_input_shapes = {
            "obs": (self.batch_size, self.env.observation_space.flat_dim),
            "act": (self.batch_size, self.env.action_space.flat_dim),
            "yval": (self.batch_size, )}
        self.qfunc.define_loss(qfunc_loss)
        self.qfunc.define_exe(
            ctx=self.ctx, 
            init=qfunc_init, 
            updater=qfunc_updater,
            input_shapes=self.qfunc_input_shapes)

        # qfunc_target init

        self.qfunc_target = self.qfunc.loss.simple_bind(ctx=self.ctx, 
                                                        **qfunc_input_shapes)
        for name, arr in self.qfunc_target.arg_dict.items():
            if name not in qfunc_input_shapes:
                self.qfunc.arg_dict[name].copyto(arr)

        # policy init

        policy_init = mx.initializer.Normal()
        loss_symbols = self.policy.get_loss_symbols()
        act_sym = loss_symbols["act"]
        weight_sym_dict = loss["weight"]
        reg = self.policy_weight_decay * 0.5 * mx.symbol.sum(
            [mx.symbol.square(arr) for name, arr in weight_sym_dict.items()])
        policy_qval = qval_sym
        loss = -1.0 / self.batch_size * mx.symbol.sum(policy_qval)
        policy_loss = loss + reg
        policy_loss = mx.symbol.MakeLoss(policy_loss, name="policy_loss")
        self.policy_executor = policy_loss.bind(ctx=ctx,
                                                args=self.qfunc.arg_dict,
                                                grad_req="write")
        self.policy_executor_arg_dict = self.policy_executor.arg_dict
        self.policy_executor_grad_dict = dict(zip(
            policy_loss.list_arguments(), 
            self.policy_executor.grad_arrays))
        policy_updater = mx.optimizer.get_updater(
            mx.optimizer.create(self.policy_updater,
                                learning_rate=self.policy_lr))
        self.policy_input_shapes = {
            "obs": (self.batch_size, self.env.observation_space.flat_dim)}
        self.policy.define_exe(
            ctx=self.ctx, 
            init=policy_init, 
            updater=policy_updater,
            input_shapes=self.policy_input_shapes)

        # policy_target init

        self.policy_target = act_sym.simple_bind(ctx=self.ctx, 
                                                 **policy_input_shapes)
        for name, arr in self.policy_target.arg_dict.items():
            if name not in policy_input_shapes:
                self.policy.arg_dict[name].copyto(arr)

    def train(self):

        memory = ReplayMem(
            obs_dim=self.env.observation_space.flat_dim,
            act_dim=self.env.action_space.flat_dim,
            memory_size=self.memory_size)

        itr = 0
        path_length = 0
        path_return = 0
        end = False
        obs = self.env.reset()

        for epoch in xrange(self.n_epochs):
            logger.push_prefix("epoch #%d | " % epoch)
            logger.log("Training started")
            for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                if end:
                    obs = self.env.reset()
                    self.strategy.reset()
                    # self.policy.reset()
                    self.strategy_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0
                act = self.strategy.get_action(obs, self.policy)
                nxt, rwd, end, _ = self.env.step(act)

                path_length += 1
                path_return += rwd

                if not end and path_length >= self.max_path_length:
                    terminal = True
                    if self.include_horizon_terminal:
                        memory.add_sample(obs, act, rwd, end)
                else:
                    memory.add_sample(obs, act, rwd, end)

                obs = nxt

                if memory.size >= self.memory_start_size:
                    for update_time in xrange(self.n_updates_per_sample):
                        batch = memroy.get_batch(self.batch_size)
                        self.do_update(itr, batch)

                itr += 1

            logger.log("Training finished")
            if memory.size >= self.memory_start_size:
                self.evaluate(epoch, memory)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()

        self.env.terminate()
        # self.policy.terminate()

    def do_update(self, itr, batch):

        obss, acts, rwds, nxts, ends = batch

        next_acts = self.policy_target.get_actions(nxts)
        policy_acts = self.policy.get_actions(obss)
        next_qvals = self.qfunc_target.get_qvals(nxts, next_acts)

        ys = rwds + (1.0 - ends) * self.discount * next_qvals

        self.qfunc.update_params(obss, acts, ys)
        qfunc_loss = self.qfunc.exe.outputs[0].asnumpy()
        qvals = self.qfunc.exe.outputs[1].asnumpy()
        self.policy_executor.arg_dict["obs"][:] = obss
        self.policy_executor.arg_dict["act"][:] = policy_acts
        self.policy_executor.forward(is_train=True)
        policy_loss = self.policy_executor.outputs[0].asnumpy()
        self.policy_executor.backward()
        self.policy.update_params(self.policy_executor_grad_dict["act"])

        for name, arr in self.policy_target.arg_dict.items():
            if name not in self.policy_input_shapes:
                arr = (1.0 - self.soft_target_tau) * arr + \
                       self.soft_target_tau * self.policy.arg_dict[name]
        for name, arr in self.qfunc_target.arg_dict.items():
            if name not in self.qfunc_input_shapes:
                arr = (1.0 - self.soft_target_tau) * arr + \
                       self.soft_target_tau * self.qfunc.arg_dict[name]
        
        self.qfunc_loss_averages.append(qfunc_loss)
        self.policy_loss_averages.append(policy_loss)
        self.q_averages.append(qvals)
        self.y_averages.append(ys)

    def evaluate(self, epoch, memory):

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_qfunc_loss = np.mean(self.qfunc_loss_averages)
        average_policy_loss = np.mean(self.policy_loss_averages)

        if len(self.strategy_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.strategy_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.strategy_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.strategy_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.strategy_path_returns))
        logger.record_tabular('AverageQLoss', average_qfunc_loss)
        logger.record_tabular('AveragePolicySurr', average_policy_loss)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))

        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.strategy_path_returns = []