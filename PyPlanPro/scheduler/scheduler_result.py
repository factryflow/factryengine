# scheduler_result.py

import pandas as pd

class SchedulerResult():
    def __init__(self, task_vars):
        self.task_vars = task_vars

    def to_dict(self):
        return self.task_vars

    def to_dataframe(self):
        df = pd.DataFrame(self.task_vars)
        return df