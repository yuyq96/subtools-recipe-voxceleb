# -*- coding:utf-8 -*-

# Copyright Nanjing University (Author: Ya-Qi Yu 2021-01-19)

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import libs.support.utils as utils

from libs.nnet import *


# _BaseActivationBatchNorm: copied from pytorch/libs/nnet/components.py
class _BaseActivationBatchNorm(torch.nn.Module):
    """[Affine +] Relu + BatchNorm1d.
    Affine could be inserted by a child class.
    """
    def __init__(self):
        super(_BaseActivationBatchNorm, self).__init__()
        self.affine = None
        self.activation = None
        self.batchnorm = None

    def add_relu_bn(self, output_dim=None, options:dict={}):
        default_params = {
            "bn-relu":False,
            "nonlinearity":'relu',
            "nonlinearity_params":{"inplace":True, "negative_slope":0.01},
            "bn":True,
            "bn_params":{"momentum":0.1, "affine":True, "track_running_stats":True},
            "special_init":True,
            "mode":'fan_out'
        }

        default_params = utils.assign_params_dict(default_params, options)

        # This 'if else' is used to keep a corrected order when printing model.
        # torch.sequential is not used for I do not want too many layer wrappers and just keep structure as tdnn1.affine 
        # rather than tdnn1.layers.affine or tdnn1.layers[0] etc..
        if not default_params["bn-relu"]:
            # ReLU-BN
            # For speaker recognition, relu-bn seems better than bn-relu. And w/o affine (scale and shift) of bn is 
            # also better than w/ affine.
            self.after_forward = self._relu_bn_forward
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
        else:
            # BN-ReLU
            self.after_forward = self._bn_relu_forward
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])

        if default_params["special_init"] and self.affine is not None:
            if default_params["nonlinearity"] in ["relu", "leaky_relu", "tanh", "sigmoid"]:
                # Before special_init, there is another initial way been done in TdnnAffine and it 
                # is just equal to use torch.nn.init.normal_(self.affine.weight, 0., 0.01) here. 
                if isinstance(self.affine, ChunkSeparationAffine):
                    torch.nn.init.kaiming_uniform_(self.affine.odd.weight, a=0, mode=default_params["mode"], 
                                               nonlinearity=default_params["nonlinearity"])
                    torch.nn.init.kaiming_uniform_(self.affine.even.weight, a=0, mode=default_params["mode"], 
                                               nonlinearity=default_params["nonlinearity"])
                else:
                    torch.nn.init.kaiming_uniform_(self.affine.weight, a=0, mode=default_params["mode"], 
                                               nonlinearity=default_params["nonlinearity"])
            else:
                torch.nn.init.xavier_normal_(self.affine.weight, gain=1.0)

    def _bn_relu_forward(self, x):
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _relu_bn_forward(self, x):
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = self.affine(inputs)
        outputs = self.after_forward(x)
        return outputs


class ReluBatchNormTdnnLayerR(_BaseActivationBatchNorm):
    """ ReLU-BN-TDNN.
    A 3-fold layer with TdnnAffine affine in the last layer.
    """
    def __init__(self, input_dim, output_dim, context=[0], affine_type="tdnn", **options):
        super(ReluBatchNormTdnnLayerR, self).__init__()

        affine_options = {
            "bias":True, 
            "groups":1,
            "norm_w":False,
            "norm_f":False
        }

        affine_options = utils.assign_params_dict(affine_options, options)

        self.add_relu_bn(input_dim, options=options)

        if affine_type == "tdnn":
            self.affine = TdnnAffine(input_dim, output_dim, context=context, **affine_options)
        else:
            self.affine = ChunkSeparationAffine(input_dim, output_dim, context=context, **affine_options)
    
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = self.after_forward(inputs)
        outputs = self.affine(x)
        return outputs


class DTdnnLayer(torch.nn.Module):

    def __init__(self, input_dim, output_dim, bn_dim, context, memory_efficient=False, **options):
        super(DTdnnLayer, self).__init__()

        self.memory_efficient = memory_efficient

        self.bn_function = ReluBatchNormTdnnLayerR(input_dim,bn_dim,**options)
        self.kn_function = ReluBatchNormTdnnLayerR(bn_dim,output_dim,context,**options)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.kn_function(x)
        return x


class DTdnnBlock(torch.nn.ModuleList):
    """ Densely connected TDNN block w.r.t https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1275.pdf.
    Reference: Yu, Y.-Q., and Li, W.-J. (2020).
               Densely Connected Time Delay Neural Network for Speaker Verification. Paper presented at the Interspeech.
    """
    def __init__(self, num_layers, input_dim, output_dim, bn_dim, context, memory_efficient=False, **options):
        super(DTdnnBlock, self).__init__()

        for i in range(num_layers):
            layer = DTdnnLayer(
                input_dim=input_dim + i * output_dim,
                output_dim=output_dim,
                bn_dim=bn_dim,
                context=context,
                memory_efficient=memory_efficient,
                **options
            )
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], 1)
        return x


class Xvector(TopVirtualNnet):
    """ A densely connected x-vector framework """

    ## Base parameters - components - loss - training strategy.
    def init(self, inputs_dim, num_targets,
             mixup=False, mixup_alpha=1.0,
             specaugment=False, specaugment_params={},
             aug_dropout=0., context_dropout=0., hidden_dropout=0., dropout_params={},
             xvector_params={},
             pooling="statistics", pooling_params={},
             fc_params={},
             margin_loss=False, margin_loss_params={},
             use_step=False, step_params={},
             transfer_from="softmax_loss",
             training=True
             ):

        ## Params.
        default_dropout_params = {
            "type":"default", # default | random
            "start_p":0.,
            "dim":2,
            "method":"uniform", # uniform | normals
            "continuous":False,
            "inplace":True
        }

        default_xvector_params = {
            "init_dim":128,
            "layers":[6, 12],
            "growth_rate":64,
            "bn_scale":2,
            "nonlinearity":"relu",
            "memory_efficient":True
        }

        default_pooling_params = {
            "num_head":1,
            "hidden_size":64,
            "share":True,
            "affine_layers":1,
            "context":[0],
            "stddev":True,
            "temperature":False,
            "fixed":True
        }

        default_fc_params = {
            "nonlinearity":'relu', "nonlinearity_params":{"inplace":True},
            "bn-relu":False,
            "bn":True,
            "bn_params":{"momentum":0.5, "affine":True, "track_running_stats":True}
        }

        default_margin_loss_params = {
            "method":"am", "m":0.2,
            "feature_normalize":True, "s":30,
            "double":False,
            "mhe_loss":False, "mhe_w":0.01,
            "inter_loss":0.,
            "ring_loss":0.,
            "curricular":False
        }

        default_step_params = {
            "T":None,
            "m":False, "lambda_0":0, "lambda_b":1000, "alpha":5, "gamma":1e-4,
            "s":False, "s_tuple":(30, 12), "s_list":None,
            "t":False, "t_tuple":(0.5, 1.2), 
            "p":False, "p_tuple":(0.5, 0.1)
        }

        dropout_params = utils.assign_params_dict(default_dropout_params, dropout_params)
        xvector_params = utils.assign_params_dict(default_xvector_params, xvector_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc_params = utils.assign_params_dict(default_fc_params, fc_params)
        margin_loss_params = utils.assign_params_dict(default_margin_loss_params, margin_loss_params)
        step_params = utils.assign_params_dict(default_step_params, step_params)

        ## Var.
        self.use_step = use_step
        self.step_params = step_params

        ## Nnet
        # Head
        self.mixup = Mixup(alpha=mixup_alpha) if mixup else None
        self.specaugment = SpecAugment(**specaugment_params) if specaugment else None
        self.aug_dropout = get_dropout_from_wrapper(aug_dropout, dropout_params)
        self.context_dropout = ContextDropout(p=context_dropout) if context_dropout > 0 else None
        self.hidden_dropout = get_dropout_from_wrapper(hidden_dropout, dropout_params)

        # Frame level
        in_dim = xvector_params["init_dim"]
        layers = xvector_params["layers"]
        out_dim = xvector_params["growth_rate"]
        bn_dim = out_dim * xvector_params["bn_scale"]
        nonlinearity = xvector_params["nonlinearity"]
        memory_efficient = xvector_params["memory_efficient"]
        options = {"bias": False, "bn-relu": True}
        self.tdnn = ReluBatchNormTdnnLayer(inputs_dim,in_dim,[-2,-1,0,1,2],nonlinearity=nonlinearity,**options)
        self.dense_block1 = DTdnnBlock(layers[0],in_dim,out_dim,bn_dim,[-1,0,1],memory_efficient,nonlinearity=nonlinearity,**options)
        in_dim += layers[0] * out_dim
        self.transit1 = ReluBatchNormTdnnLayerR(in_dim,in_dim//2,nonlinearity=nonlinearity,**options)
        in_dim //= 2
        self.dense_block2 = DTdnnBlock(layers[1],in_dim,out_dim,bn_dim,[-3,0,3],memory_efficient,nonlinearity=nonlinearity,**options)
        in_dim += layers[1] * out_dim
        self.transit2 = ReluBatchNormTdnnLayerR(in_dim,in_dim//2,nonlinearity=nonlinearity,**options)
        in_dim //= 2

        # Pooling
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.stats = LDEPooling(in_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(in_dim, affine_layers=pooling_params["affine_layers"], 
                                                    hidden_size=pooling_params["hidden_size"], 
                                                    context=pooling_params["context"], stddev=stddev)
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(in_dim, stddev=stddev, **pooling_params)
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(in_dim, **pooling_params)
        else:
            self.stats = StatisticsPooling(in_dim, stddev=stddev)

        # Segment level
        self.fc = ReluBatchNormTdnnLayer(self.stats.get_output_dim(), 512, **fc_params)

        # Loss
        # Do not need when extracting embedding.
        if training :
            if margin_loss:
                self.loss = MarginSoftmaxLoss(512, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(512, num_targets)

            self.wrapper_loss = MixupLoss(self.loss, self.mixup) if mixup else None

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["tdnn", "block1", "transit1", "block2", "transit2", "stats", "fc", "loss"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight":"loss.weight"} 

    @utils.for_device_free
    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """

        x = inputs

        x = self.auto(self.mixup, x)
        x = self.auto(self.specaugment, x)
        x = self.auto(self.aug_dropout, x)
        x = self.auto(self.context_dropout, x)

        x = self.tdnn(x)
        x = self.dense_block1(x)
        x = self.transit1(x)
        x = self.dense_block2(x)
        x = self.transit2(x)
        x = self.stats(x)
        x = self.fc(x)
        x = self.auto(self.hidden_dropout, x)

        return x

    @utils.for_device_free
    def get_loss(self, inputs, targets):
        """Should call get_loss() after forward() with using Xvector model function.
        e.g.:
            m=Xvector(20,10)
            loss=m.get_loss(m(inputs),targets)
        
        model.get_loss [custom] -> loss.forward [custom]
          |
          v
        model.get_accuracy [custom] -> loss.get_accuracy [custom] -> loss.compute_accuracy [static] -> loss.predict [static]
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss(inputs, targets)
        else:
            return self.loss(inputs, targets)

    @utils.for_device_free
    def get_accuracy(self, targets):
        """Should call get_accuracy() after get_loss.
        @return: return accuracy
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_accuracy(targets)
        else:
            return self.loss.get_accuracy(targets)

    def get_posterior(self):
        """Should call get_posterior after get_loss. This function is to get outputs from loss component.
        @return: return posterior
        """
        if self.wrapper_loss is not None:
            return self.wrapper_loss.get_posterior()
        else:
            return self.loss.get_posterior()

    @for_extract_embedding(maxChunk=10000, isMatrix=True)
    def extract_embedding(self, inputs):
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs

        x = self.tdnn(x)
        x = self.dense_block1(x)
        x = self.transit1(x)
        x = self.dense_block2(x)
        x = self.transit2(x)
        x = self.stats(x)
        xvector = self.fc.affine(x)

        return xvector

    def get_warmR_T(T_0, T_mult, epoch):
        n = int(math.log(max(0.05, (epoch / T_0 * (T_mult - 1) + 1)), T_mult))
        T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
        T_i = T_0 * T_mult ** (n)
        return T_cur, T_i

    def compute_decay_value(self, start, end, T_cur, T_i):
        # Linear decay in every cycle time.
        return start - (start - end)/(T_i-1) * (T_cur%T_i)

    def step(self, epoch, this_iter, epoch_batchs):
        # Heated up for t and s.
        # Decay for margin and dropout p.
        if self.use_step:
            if self.step_params["m"]:
                current_postion = epoch*epoch_batchs + this_iter
                lambda_factor = max(self.step_params["lambda_0"], 
                                 self.step_params["lambda_b"]*(1+self.step_params["gamma"]*current_postion)**(-self.step_params["alpha"]))
                self.loss.step(lambda_factor)

            if self.step_params["T"] is not None and (self.step_params["t"] or self.step_params["p"]):
                T_cur, T_i = get_warmR_T(*self.step_params["T"], epoch)
                T_cur = T_cur*epoch_batchs + this_iter
                T_i = T_i * epoch_batchs

            if self.step_params["t"]:
                self.loss.t = self.compute_decay_value(*self.step_params["t_tuple"], T_cur, T_i)

            if self.step_params["p"]:
                self.aug_dropout.p = self.compute_decay_value(*self.step_params["p_tuple"], T_cur, T_i)

            if self.step_params["s"]:
                self.loss.s = self.step_params["s_tuple"][self.step_params["s_list"][epoch]]
