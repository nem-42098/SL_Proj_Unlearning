from torch.nn import Module, CrossEntropyLoss, KLDivLoss
from torch import Tensor


class ReconstructionLoss(Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, output_teacher: Tensor, output_student: Tensor,
                labels: Tensor, alpha: float = 0) -> float:
        """
        Forward pass for the reconstruction algorithm
        :param output_teacher: The output of the teacher network as a torch.Tensor
        :param output_student: The output of the student network as a torch.Tensor
        :param labels: The actual target variable as a torch.Tensor
        :param alpha: The hyperparameter to tune
        :return: The loss value
        """

        loss = CrossEntropyLoss(output_student, labels, reduction='mean') + \
               alpha * KLDivLoss(output_teacher, output_student, reduction='batchmean')

        return loss
