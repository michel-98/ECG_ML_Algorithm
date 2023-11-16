import os

from tensorboard.compat import tf
from torch.utils.tensorboard.writer import SummaryWriter


class CustomEventFileWriter(SummaryWriter):
    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=""):
        super().__init__(log_dir=logdir, max_queue=max_queue, flush_secs=flush_secs)
        self.filename_suffix = filename_suffix

    def _get_file_writer(self):
        path_template = os.path.join(self.log_dir, "%s.%s.%s" % ("events", self.filename_suffix, "%d"))
        path = path_template % 1  # Use 1 as the initial step
        return tf.summary.create_file_writer_v2(path, flush_millis=(self.flush_secs * 1000))
