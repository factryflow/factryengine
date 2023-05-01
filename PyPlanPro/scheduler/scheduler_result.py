# scheduler_result.py

import pandas as pd

class SchedulerResult():
    def __init__(self, task_vars):
        self.task_vars = task_vars
        self.unscheduled_tasks = [task["task_id"] for task in self.task_vars if task["assigned_resource_id"] is None]

    def to_dict(self):
        return self.task_vars

    def to_dataframe(self):
        df = pd.DataFrame(self.task_vars)
        return df
    
    def summary(self):
        summary = f"Scheduled {len(self.task_vars) - len(self.unscheduled_tasks)} of {len(self.task_vars)} tasks."
        if self.unscheduled_tasks: 
            summary += f"\nNo available resources found for task ids: {', '.join(map(str, self.unscheduled_tasks))}"
        return summary