import torch
from torchmetrics import Metric

torch.set_default_dtype(torch.float64)

class LocalAccuracy(Metric):
    def __init__(self, dist_sync_on_step=True, ignore_index=20):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state("correct",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_max = preds.argmax(dim=-1)
        assert preds_max.shape == target.shape

        correct_unmasked = preds_max == target
        target_unmasked = torch.ones(target.shape).type_as(target)
        
        keep_tensor = (~(target == self.ignore_index)).type_as(target)
        
        self.correct += torch.sum(correct_unmasked*keep_tensor)
        self.total += torch.sum(target_unmasked*keep_tensor)

    def compute(self):
        
        if self.total == 0:
            ret_val = self.correct.double() / torch.ones(
                self.total.shape).type_as(self.total)
        else:
            ret_val = self.correct.double() / self.total
        #print('ret val ', ret_val.shape, ret_val)
        return ret_val

class LocalAccuracyBinary(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #print(preds.shape)
        # logits -> so threshold of 0
        preds_max = (preds > 0.0).long()
        assert preds_max.shape == target.shape

        correct_unmasked = preds_max == target
        target_unmasked = torch.ones(target.shape).type_as(target)
        
        self.correct += torch.sum(correct_unmasked)
        self.total += torch.sum(target_unmasked.long())

    def compute(self):
        
        if self.total == 0:
            ret_val = self.correct.double() / torch.ones(
                self.total.shape).type_as(self.total)
        else:
            ret_val = self.correct.double() / self.total
        #print('ret val ', ret_val.shape, ret_val)
        return ret_val


