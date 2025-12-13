# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta


def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None, mask=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        if mask is None:
            outputs = predict(xvar + delta)
        else:
            outputs = predict(xvar + delta * mask)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            delta.data = delta.data.to(xvar.device)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()
    if mask is None:
        x_adv = clamp(xvar + delta, clip_min, clip_max)
    else:
        x_adv = clamp(xvar + delta * mask, clip_min, clip_max)
    return x_adv


def perturb_iterative_watermarked(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                                  watermark_decoder, secret_message, lambda_weight, # <--- NEW ARGS
                                  delta_init=None, minimize=False, ord=np.inf,
                                  clip_min=0.0, clip_max=1.0,
                                  l1_sparsity=None, mask=None):
    """
    Modified iterative loop that minimizes MIST loss AND Watermark extraction error.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    
    # We use MSE for the watermark ensuring the message matches the secret
    mse_loss = nn.MSELoss() 

    for ii in range(nb_iter):
        # 1. Generate the candidate adversarial image
        if mask is None:
            adv_image = xvar + delta
        else:
            adv_image = xvar + delta * mask
        
        outputs = predict(adv_image)
        
        # 2. Calculate MIST Loss (The "Attack")
        # In MIST, we minimize the distance to the target texture.
        mist_loss = loss_fn(outputs, yvar)

        # 3. Calculate Watermark Loss (The "Defense") <--- NEW LOGIC
        # We pass the image (pixels) into the decoder, not the class logits
        decoded_msg = watermark_decoder(adv_image) 
        wat_loss = mse_loss(decoded_msg, secret_message)

        # 4. Combine Losses
        # We want to minimize (MIST_Loss + Lambda * Watermark_Loss)
        total_loss = mist_loss + (lambda_weight * wat_loss)

        if minimize:
            # If minimize=True, the loop minimizes the value.
            # (MIST sets minimize=True to get closer to the target texture)
            loss = total_loss 
        else:
            # Standard PGD maximizes loss, so we would flip it.
            # But for MIST targeted attacks, we usually stay in 'minimize' mode.
            loss = -total_loss

        loss.backward()

        # 5. Standard Gradient Update (Same as original)
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            # If minimize=True, we subtract gradient (descent)
            # The original code handles this by negating loss earlier if needed.
            # Let's stick to the original logic:
            # If minimize=True, loss was NOT negated above? 
            # WAIT: The original code does `loss = -loss` if minimize=True.
            # Then it ADDS grad_sign. adding negative gradient = descent. 
            
            # So if we want to minimize total_loss:
            if minimize:
                 loss = -total_loss # Negate so backward() gives negative grad
            
            # The rest remains identical to standard PGD
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
            
        else:
            # (Omitted L2/L1 logic for brevity, but you can copy it from original)
            raise NotImplementedError("Only Linf implemented for this demo")

        delta.grad.data.zero_()

    # Final Clamp
    if mask is None:
        x_adv = clamp(xvar + delta, clip_min, clip_max)
    else:
        x_adv = clamp(xvar + delta * mask, clip_min, clip_max)
        
    return x_adv

class PGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.
        """
        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None, mask=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity, mask=mask
        )

        return rval.data


class L2PGDAttack(PGDAttack):
    """
    PGD Attack with order=L2
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)


class LinfPGDAttack(PGDAttack):
    """
    PGD Attack with order=Linf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)

class WatermarkedPGDAttack(LinfPGDAttack):
    def __init__(self, predict, watermark_decoder, lambda_weight=0.5, **kwargs):
        super().__init__(predict, **kwargs)
        self.watermark_decoder = watermark_decoder
        self.lambda_weight = lambda_weight

    def perturb(self, x, y=None, mask=None, secret_message=None):
        # Verify inputs (from parent class)
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        
        # (Optional: Rand Init logic from parent can go here)

        # Call our CUSTOM engine
        rval = perturb_iterative_watermarked(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, 
            
            # NEW PARAMS passed down
            watermark_decoder=self.watermark_decoder,
            secret_message=secret_message,
            lambda_weight=self.lambda_weight,
            
            minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity, mask=mask
        )

        return rval.data