# scheduler_result.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SchedulerResult:
    def __init__(self, task_vars):
        self.task_vars = task_vars
        self.unscheduled_tasks = [
            task["task_id"]
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
        df = self.to_dataframe()
        # Sort dataframe by assigned_resource_ids and task_start to visualize in order
        df = df.sort_values(["assigned_resource_ids", "task_start"])

        # Create a color dictionary for each unique resource for distinction in the plot
        resources = df["assigned_resource_ids"].unique()
        colors = sns.color_palette(
            "deep", len(resources)
        )  # Using seaborn "dark" color palette
        color_dict = dict(zip(resources, colors))

        # Set seaborn style
        sns.set_style("whitegrid")

        plt.figure(figsize=(12, 6))

        for resource_id, group_df in df.groupby("assigned_resource_ids"):
            for task in group_df.itertuples():
                plt.barh(
                    resource_id,
                    left=task.task_start,
                    width=task.task_end - task.task_start,
                    color=color_dict[resource_id],
                    edgecolor="black",
                )
                plt.text(
                    x=(task.task_start + task.task_end)
                    / 2,  # x position, in the middle of task bar
                    y=resource_id,  # y position, on the resource line
                    s=task.task_id,  # text string, which is task_id here
                    va="center",  # vertical alignment
                    ha="center",  # horizontal alignment
                    color="black",  # text color
                    fontsize=10,
                )  # font size

        plt.xlabel("Time")
        plt.ylabel("Resource")
        plt.title("Resource Plan")
        plt.yticks(df["assigned_resource_ids"].unique())
        plt.tight_layout()  # adjusts subplot params so that the subplot(s) fits into the figure area.
        plt.show()
