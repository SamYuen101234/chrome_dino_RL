import pickle
from replay_buffer.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
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

def init_cache(INITIAL_EPSILON, REPLAY_MEMORY, prioritized_replay):
    """initial variable caching, done only once"""
    if prioritized_replay:
        t, D = 0, PrioritizedReplayBuffer(maxlen=REPLAY_MEMORY)
    else:
        t, D = 0, ReplayBuffer(maxlen=REPLAY_MEMORY) # for experience replay
    set_up_dict = {"epsilon": INITIAL_EPSILON, "step": t, "D": D, "highest_score": 0}
    save_obj(set_up_dict, "set_up")

def save_obj(obj, name):
    with open('./result/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
        # Use the latest protocol that supports the lowest Python version you want to support reading the data
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open('result/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


