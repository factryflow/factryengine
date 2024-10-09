# scheduler_result.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SchedulerResult:
    def __init__(self, task_vars: list[dict], unscheduled_task_ids: list[int]):
        self.task_vars = task_vars
        self.unscheduled_task_ids = unscheduled_task_ids

    def to_dict(self) -> list[dict]:
        return self.task_vars

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.task_vars)
        return df

    def summary(self) -> str:
        """
        Generate a summary of the scheduling results.
        """

        # Calculate the number of scheduled and total tasks
        num_scheduled_tasks = len(self.task_vars) - len(self.unscheduled_task_ids)
        total_tasks = len(self.task_vars)

        # Start the summary with the number of scheduled tasks
        summary = f"Scheduled {num_scheduled_tasks} of {total_tasks} tasks."

        # If there are any unscheduled tasks, add them to the summary
        if self.unscheduled_task_ids:
            unscheduled_task_ids = ", ".join(map(str, self.unscheduled_task_ids))
            summary += (
                f"\nNo available resources found for task ids: {unscheduled_task_ids}"
            )

        return summary

    def plot_resource_plan(self) -> None:
        # Get the resource intervals DataFrame
        df = self.get_resource_intervals_df()

        # Create a color dictionary for each unique task for distinction in the plot
        unique_tasks = df["task_id"].unique()
        color_palette = sns.color_palette(
            "deep", len(unique_tasks)
        )  # Using seaborn "deep" color palette
        color_dict = dict(zip(unique_tasks, color_palette))

        # Set seaborn style
        sns.set_style("whitegrid")

        # Create a new figure with a specific size
        plt.figure(figsize=(12, 6))

        # Iterate over each task group
        for task_id, group_df in df.groupby("task_id"):
            # Iterate over each task in the group
            for task in group_df.itertuples():
                # Draw a horizontal bar for the task
                plt.barh(
                    task.resource_id,
                    left=task.interval_start,
                    width=task.interval_end - task.interval_start,
                    color=color_dict[task_id],
                    edgecolor="black",
                )

                # Add a text label in the middle of the bar
                text_position = (task.interval_start + task.interval_end) / 2
                plt.text(
                    x=text_position,
                    y=task.resource_id,
                    s=task.task_id,
                    va="center",
                    ha="center",
                    color="black",
                    fontsize=10,
                )

        # Set the labels and title of the plot
        plt.xlabel("Time")
        plt.ylabel("Resource")
        plt.title("Resource Plan")

        # Set the y-ticks to the unique resource IDs
        plt.yticks(df["resource_id"].unique())

        # Adjust the layout so everything fits nicely
        plt.tight_layout()

        # Display the plot
        plt.show()

    def get_resource_intervals_df(self) -> pd.DataFrame:
        """
        Explodes the resource intervals to create a dataframe with one row per resource interval
        """
        # Convert the object to a DataFrame
        df = self.to_dataframe()

        # Explode the 'assigned_resource_ids' and 'resource_intervals' columns
        exploded_df = df.explode(["assigned_resource_ids", "resource_intervals"])

        # Drop any rows with missing values
        cleaned_df = exploded_df.dropna()

        exploded_intervals_df = cleaned_df.explode("resource_intervals")
        exploded_intervals_df = exploded_intervals_df.reset_index(drop=True)

        # Extract the start and end of the interval from the 'resource_intervals' column
        exploded_intervals_df["interval_start"] = exploded_intervals_df.resource_intervals.apply(
            lambda x: x[0]
        )

        print('PASS INTERVAL START')
        exploded_intervals_df["interval_end"] = exploded_intervals_df.resource_intervals.apply(lambda x: x[1])

        print('PASS INTERVAL END')

        # Rename the 'assigned_resource_ids' column to 'resource_id'
        renamed_df = exploded_intervals_df.rename(columns={"assigned_resource_ids": "resource_id"})

        # Select only the columns we're interested in
        selected_columns_df = renamed_df[
            ["task_id", "resource_id", "interval_start", "interval_end"]
        ]

        # Infer the best data types for each column
        final_df = selected_columns_df.infer_objects()

        return final_df
