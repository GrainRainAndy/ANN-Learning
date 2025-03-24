

class SimpleEarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss >= self.best_loss - self.min_delta:
            self.counter += 1
        else:
            self.best_loss = loss
            self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True

class AvgEarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.losses = []
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        self.losses.append(loss)
        if len(self.losses) > 10:
            self.losses.pop(0)

        if len(self.losses) == 10:
            avg_loss = sum(self.losses) / 10
            if loss >= avg_loss - self.min_delta:
                self.counter += 1
            else:
                self.counter = 0

            if self.counter >= self.patience:
                self.early_stop = True