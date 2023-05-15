import string
from tensorboardX import SummaryWriter


class Writer:
    def __init__(self, logdir: string):
        self.logdir = logdir
        self.tb_writer = SummaryWriter(logdir=logdir)

    def step(self, global_step, lr, loss_per_sample):
        self.tb_writer.add_scalar('learning rate', scalar_value=lr, global_step=global_step)
        self.tb_writer.add_scalar('loss per train sample', scalar_value=loss_per_sample, global_step=global_step)

    def epoch(self, global_step, avg_loss):
        self.tb_writer.add_scalar('avg loss over epoch', scalar_value=avg_loss, global_step=global_step)

    def evaluation(self, global_step, epoch, avg_test_loss):
        self.tb_writer.add_scalar('evaluation loss per sample', scalar_value=avg_test_loss, global_step=global_step)
        print('Epoch: ' + str(epoch) + ' , Evaluation loss per sample: ' + str(avg_test_loss))
