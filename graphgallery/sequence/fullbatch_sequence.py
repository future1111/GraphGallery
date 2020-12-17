from graphgallery.sequence.base_sequence import Sequence


class FullBatchSequence(Sequence):

    def __init__(self, x, y=None, sample_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x = self.astensors(x, device=self.device)
        self.y = self.astensor(y, device=self.device)
        self.sample_weight = self.astensor(sample_weight, device=self.device)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.x, self.y, self.sample_weight

    def on_epoch_begin(self):
        ...

    def on_epoch_end(self):
        ...

    def _shuffle_batches(self):
        ...
