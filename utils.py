import mxnet as mx


SEED = 12345


def define_qfunc(obs, act):

    net = mx.symbol.FullyConnected(
        data=obs, 
        name="qfunc_fc1", 
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net, 
        name="qfunc_relu1", 
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net, 
        name="qfunc_fc2", 
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net, 
        name="qfunc_relu2", 
        act_type="relu")
    net = mx.symbol.Concat(net, act, name="qunfc_concat")
    net = mx.symbol.FullyConnected(
        data=net, 
        name="qfunc_fc3", 
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net, 
        name="qfunc_relu3", 
        act_type="relu")
    qval = mx.symbol.FullyConnected(
        data=net, 
        name="qfunc_qval", 
        num_hidden=1)

    return qval


def define_policy(obs, action_dim):

    net = mx.symbol.FullyConnected(
        data=obs, 
        name="policy_fc1", 
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net, 
        name="policy_relu1", 
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net, 
        name="policy_fc2", 
        num_hidden=32)
    net = mx.symbol.Activation(
        data=net, 
        name="policy_relu2", 
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net, 
        name='policy_fc3', 
        num_hidden=action_dim)
    action = mx.symbol.Activation(
        data=net, 
        name="act", 
        act_type="tanh")

    return action