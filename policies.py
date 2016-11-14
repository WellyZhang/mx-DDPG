from utils import define_policy
import mxnet as mx


class Policy(object):

    def __init__(self, env_spec):

        self.env_spec = env_spec

    def get_actions(self, obs):

        raise NotImplementedError

    @property
    def observation_space(self):

        return self.env_spec.observation_space

    @property
    def action_space(self):

        return self.env_spec.action_space


class DeterministicMLPPolicy(Policy):

    def __init__(
        self,
        env_spec):

        super(DeterministicMLPPolicy, self).__init__(env_spec)

        self.obs = mx.symbol.Variable("obs")
        self.act, self.weight = define_policy(
            self.obs, 
            self.env_spec.action_space.flat_dim)

    def get_output_symbol(self):

        return self.act

    def get_loss_symbols(self):

        return {"obs": self.obs,
                "act": self.act, 
                "weight": self.weight}

    def define_loss(self, loss_exp):

        raise NotImplementedError

    def define_exec(self, ctx, init, updater, input_shapes=None, args=None, 
                    grad_req=None):

        self.exec = self.act.simple_bind(ctx=ctx, **input_shapes)
        self.arg_arrays = self.exec.arg_arrays
        self.grad_arrays = self.exec.grad_arrays
        self.arg_dict = self.exec.arg_dict

        for name, arr in self.arg_dict.items():
            if name not in input_shapes:
                init(name, arr)
                
        self.updater = updater

        new_input_shapes = {obs=(1, input_shapes["obs"][1])}
        self.exec_one = self.exec.reshape(**new_input_shapes)

    def update_params(self, grad_from_top):

        self.exec.forward(is_train=True)
        self.exec.backward([grad_from_top])

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_actions(self, obs):

        self.arg_dict["obs"][:] = obs
        self.exec.forward(is_train=False)

        return self.exec.outputs[0].asnumpy()

    def get_action(self, obs):

        self.arg_dict_one["obs"][:] = obs
        self.exec_one.forward(is_train=False)

        return self.exec_one.outputs[0].asnumpy()





        