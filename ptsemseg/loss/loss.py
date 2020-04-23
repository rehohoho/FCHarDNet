import torch
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    
    loss = F.cross_entropy(
              input, target, weight=weight, size_average=size_average, ignore_index=250, reduction='mean')

    return loss


def multi_scale_cross_entropy2d(input, target, loss_th, weight=None, size_average=True, scale_weight=[1.0, 0.4]):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    K = input[0].size()[2] * input[0].size()[3] // 128
    loss = 0.0

    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * bootstrapped_cross_entropy2d(
            input=inp, target=target, min_K=K, loss_th=loss_th, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, min_K, loss_th, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    batch_size = input.size()[0]
    
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    
    thresh = loss_th
    
    def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)

        # takes logits as input, hard labels as target
        # performs log_softmax and nll_loss
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )
        sorted_loss, _ = torch.sort(loss, descending=True)
        
        if sorted_loss[K] > thresh:
            loss = sorted_loss[sorted_loss > thresh]
        else:
            loss = sorted_loss[:K]
        reduced_topk_loss = torch.mean(loss)

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=min_K,
            thresh=thresh,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)


def soft_and_hard_target_cross_entropy(input, hard_target, soft_target, temperature, ignore_mask, weight=None):

    n, c, h, w = input.size()
    n_softtar, c_softtar, h_softtar, w_softtar = soft_target.size()
    n_hardtar, h_hardtar, w_hardtar = hard_target.size()
    batch_size = input.size()[0]
    
    assert h_softtar == h_hardtar and w_softtar == w_hardtar, 'Height and width of soft and hard targets are different! soft: %s hard: %s' %(soft_target.size(), hard_target.size())
    assert c == c_softtar, 'Classes in prediction and target is different! pred: %s target %s' %(input.size(), soft_target.size())
    
    if h != h_hardtar and w != w_hardtar:  # upsample labels
        input = F.interpolate(input, size=(h_hardtar, w_hardtar), mode="bilinear", align_corners=True)

    soft_input = F.log_softmax(input / temperature, dim=1) # KL-div expects log probabilities for inputs, probabilities for targets

    def _soft_and_hard_xentropy_single(
        hard_input, hard_target, 
        soft_input, soft_target, ignore_mask, 
        weight=None):
        
        n, c, h, w = hard_input.size()
        hard_input = hard_input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) # CHW -> HWC -> NC
        hard_target = hard_target.view(-1) # HW -> N
        
        # performs log_softmax and nll_loss
        hard_target_loss = F.cross_entropy(
            hard_input, hard_target, weight=weight, reduce=True, reduction='mean', ignore_index=250
        )

        # ignore mask is repeated to fit shape of hard_input / target
        soft_target_loss = F.kl_div(
            soft_input*ignore_mask, soft_target*ignore_mask, reduce = True, reduction='batchmean'
        )
        
        return hard_target_loss, soft_target_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        hard_target_loss, soft_target_loss = _soft_and_hard_xentropy_single(
            hard_input=torch.unsqueeze(input[i], 0),
            hard_target=torch.unsqueeze(hard_target[i], 0),
            soft_input=torch.unsqueeze(soft_input[i], 0),
            soft_target=torch.unsqueeze(soft_target[i], 0),
            ignore_mask=ignore_mask[i]
        )
        soft_target_loss *= 1000
        loss += hard_target_loss + soft_target_loss
        
    return loss / float(batch_size)


def l1(input, target, positive_example_weight = 1.0):
    # expects input of size N and target of size N
    assert input.size()[0] == target.size()[0]

    weight = target*positive_example_weight
    weight[weight<0] = 1

    loss = torch.abs(input.view(-1) - target)
    loss *= weight

    return torch.mean(loss)


def cross_entropy1d(input, target, weight=None):
    # expects input of NxC and target of N
    assert input.size()[0] == target.size()[0]

    loss = F.cross_entropy(
        input, target, weight=weight, reduce=True, reduction='mean'
    )

    return loss