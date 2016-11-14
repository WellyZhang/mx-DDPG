import mxnet as mx


SEED = 12345


def define_qfunc(obs, act):

    weight = {
        qfunc_fc1_weight=mx.symbol.Variable("qfunc_fc1_weight"),
        qfunc_fc2_weight=mx.symbol.Variable("qfunc_fc2_weight"),
        qfunc_fc3_weight=mx.symbol.Variable("qfunc_fc3_weight")
        }

    net = mx.symbol.FullyConnected(
        data=obs, 
        name="qfunc_fc1", 
        num_hidden=32,
        weight=weight["qfunc_fc1_weight"])
    net = mx.symbol.Activation(
        data=net, 
        name="qfunc_relu1", 
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net, 
        name="qfunc_fc2", 
        num_hidden=32,
        weight=weight["qfunc_fc2_weight"])
    net = mx.symbol.Activation(
        data=net, 
        name="qfunc_relu2", 
        act_type="relu")
    net = mx.symbol.Concat(net, act, name="qunfc_concat")
    net = mx.symbol.FullyConnected(
        data=net, 
        name="qfunc_fc3", 
        num_hidden=32,
        weight=weight["qfunc_fc3_weight"])
    net = mx.symbol.Activation(
        data=net, 
        name="qfunc_relu3", 
        act_type="relu")
    qval = mx.symbol.FullyConnected(
        data=net, 
        name="qfunc_qval", 
        num_hidden=1)

    return qval, weight


def define_policy(obs, action_dim):

    weight = {
        policy_fc1_weight=mx.symbol.Variable("policy_fc1_weight"),
        policy_fc2_weight=mx.symbol.Variable("policy_fc2_weight"),
        policy_fc3_weight=mx.symbol.Variable("policy_fc3_weight")
        }

    net = mx.symbol.FullyConnected(
        data=obs, 
        name="policy_fc1", 
        num_hidden=32,
        weight=weight["policy_fc1_weight"])
    net = mx.symbol.Activation(
        data=net, 
        name="policy_relu1", 
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net, 
        name="policy_fc2", 
        num_hidden=32,
        weight=weight["policy_fc2_weight"])
    net = mx.symbol.Activation(
        data=net, 
        name="policy_relu2", 
        act_type="relu")
    net = mx.symbol.FullyConnected(
        data=net, 
        name='policy_fc3', 
        num_hidden=action_dim,
        weight=weight["policy_fc3_weight"])
    action = mx.symbol.Activation(
        data=net, 
        name="act", 
        act_typ="tanh")

    return action, weight