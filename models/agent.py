
class ChromeDinoAgent:
    def __init__(self, img_channels, ACTIONS, lr, batch_size, gamma, device):
        # prototype constructor
        self.img_channels = img_channels
        self.num_actions = ACTIONS
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

    def sync_target(self):
        pass

    def save_model(self):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

    def step(self, state_t, action_t, reward_t, state_t1, terminal):
        raise NotImplementedError

