from torch.utils.tensorboard import SummaryWriter

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger:
    def __init__(self, save_dir):
        self.writer = SummaryWriter(comment=save_dir)
        self.avg_loss = AverageMeter()
        self.avg_Q_max = AverageMeter()
        self.avg_reward = AverageMeter()
        self.avg_score = AverageMeter()
        self.avg_fps = AverageMeter()

    def log_step(self, reward, loss, q):
        pass

    def log_episode(self):
        pass

    def init_episode(self):
        pass

    def record(self, episode, epsilon, step):
        pass