from utils import define_qfunc
import mxnet as mx


class QFunc(object):

    def __init__(self, env_spec):

        self.env_spec = env_spec

    def get_qvals(self, obs, act):

        raise NotImplementedError


class ContinuousMLPQ(QFunc):

    def __init__(
        self,
        env_spec):

        super(ContinuousMLPQ, self).__init__(env_spec)

        self.obs = mx.symbol.Variable("obs")
        self.act = mx.symbol.Variable("act")
        self.qval, self.weight = define_qfunc(self.obs, self.act)
        self.yval = mx.symbol.Variable("yval")

    def get_output_simbol(self):

        return self.qval

    def get_loss_simbols(self):

        return {"qval": self.qval,
                "yval": self.yval,
                "weight": self.weight}

    def define_loss(self, loss_exp):

        self.loss = mx.symbol.MakeLoss(loss_exp, name="qfunc_loss")
        self.loss = mx.symbol.Group([self.loss, self.qval])

    def define_exe(self, ctx, init, updater, input_shapes=None, args=None, 
                    grad_req=None):

        self.exe = self.loss.simple_bind(ctx=ctx, **input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict
        
        for name, arr in self.arg_dict.items():
            if name not in input_shapes:
                init(name, arr)
                
        self.updater = updater

    def update_params(self, obs, act, yval):

        self.arg_dict["obs"][:] = obs
        self.arg_dict["act"][:] = act
        self.arg_dict["yval"][:] = yval

        self.exe.forward(is_train=True)
        self.exe.backward()

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_qvals(self, obs, act):

        self.arg_dict["obs"][:] = obs
        self.arg_dict["act"][:] = act
        self.exe.forward(is_train=False)

        return self.exe.outputs[1].asnumpy()


