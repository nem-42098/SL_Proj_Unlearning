from torch.nn import Module, CrossEntropyLoss, KLDivLoss
from torch import Tensor


class ReconstructionLoss(Module):
    """
    :param alpha: the weight of the KLD loss 
    """
    def __init__(self, alpha: float = 1):
        super(ReconstructionLoss, self).__init__()
        self.cel = CrossEntropyLoss(reduction='mean')
        self.kld = KLDivLoss(reduction='batchmean')
        self.alpha = alpha

    def forward(self, output_teacher: Tensor, output_student: Tensor, labels: Tensor) -> float:
        """
        Forward pass for the reconstruction algorithm
        :param output_teacher: The output of the teacher network as a torch.Tensor
        :param output_student: The output of the student network as a torch.Tensor
        :param labels: The actual target variable as a torch.Tensor
        :return: The loss value
        """

        loss = self.cel(output_student, labels) + self.alpha * self.kld(output_teacher, output_student)

        return loss
