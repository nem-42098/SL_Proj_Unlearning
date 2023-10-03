from torch.nn import Module, CrossEntropyLoss, KLDivLoss
from torch import Tensor
from torch.nn.functional import softmax


class ReconstructionLoss(Module):
    """
    :param alpha: the weight of the KLD loss 
    """
    def __init__(self, alpha: float = 1):
        super(ReconstructionLoss, self).__init__()
        self.cel = CrossEntropyLoss(reduction='mean')
        self.kld = KLDivLoss(reduction='batchmean')
        self.alpha = alpha

    def forward(self, output_student: Tensor, output_teacher: Tensor, labels: Tensor = None) -> float:
        """
        Forward pass for the reconstruction algorithm
        :param output_teacher: The output of the teacher network as a torch.Tensor
        :param output_student: The output of the student network as a torch.Tensor
        :param labels: The actual target variable as a torch.Tensor
        :return: The loss value
        """
        kld_loss = self.kld(softmax(output_student, dim=1), softmax(output_teacher, dim=1))
        if labels is None:
            return kld_loss
        
        cel_loss = self.cel(output_student, labels)
        return cel_loss + self.alpha * kld_loss
