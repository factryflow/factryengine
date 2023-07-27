# scheduler_result.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SchedulerResult:
    def __init__(self, task_vars):
        self.task_vars = task_vars
        self.unscheduled_tasks = [
            task["task_uid"]
            for task in self.task_vars
            if task["assigned_resource_ids"] is None
        ]

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

    def plot_resource_plan(self) -> None:
        df = self.get_resource_intervals_df()

        # Create a color dictionary for each unique resource for distinction in the plot
        tasks = df["task_uid"].unique()
        colors = sns.color_palette(
            "deep", len(tasks)
        )  # Using seaborn "dark" color palette
        color_dict = dict(zip(tasks, colors))

        # Set seaborn style
        sns.set_style("whitegrid")

        plt.figure(figsize=(12, 6))

        for task_uid, group_df in df.groupby("task_uid"):
            for task in group_df.itertuples():
                plt.barh(
                    task.resource_id,
                    left=task.interval_start,
                    width=task.interval_end - task.interval_start,
                    color=color_dict[task_uid],
                    edgecolor="black",
                )
                plt.text(
                    x=(task.interval_start + task.interval_end)
                    / 2,  # x position, in the middle of task bar
                    y=task.resource_id,  # y position, on the resource line
                    s=task.task_uid,  # text string, which is task_uid here
                    va="center",  # vertical alignment
                    ha="center",  # horizontal alignment
                    color="black",  # text color
                    fontsize=10,
                )  # font size

        plt.xlabel("Time")
        plt.ylabel("Resource")
        plt.title("Resource Plan")
        plt.yticks(df["resource_id"].unique())
        plt.tight_layout()  # adjusts subplot params so that the subplot(s) fits into the figure area.
        plt.show()

    def get_resource_intervals_df(self) -> pd.DataFrame:
        df = self.to_dataframe()
        df = df.explode(["assigned_resource_ids", "resource_intervals"]).explode(
            "resource_intervals"
        )
        df = df.dropna()
        df["interval_start"] = df.resource_intervals.apply(lambda x: x[0])
        df["interval_end"] = df.resource_intervals.apply(lambda x: x[1])
        df = df.rename(columns={"assigned_resource_ids": "resource_id"})
        df = df[["task_uid", "resource_id", "interval_start", "interval_end"]]
        df = df.infer_objects()
        return df
