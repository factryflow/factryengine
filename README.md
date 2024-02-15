# 📅 factryengine

`factryengine` is a high-speed Python package for effortless and efficient task scheduling, specifically tailored for production scheduling. Built with `numpy`, it ensures tasks are executed in the correct order while considering their priorities, resources, and dependencies.

## 🛠 Installation

Install `factryengine` with a simple pip command:

```bash
pip install factryengine
```

## 🌟 Features

- ⚡ **Fast Performance**: Built with `numpy` for high-speed task scheduling.
- 🏭 **Production Scheduling**: Specifically designed for seamless production scheduling.
- 📝 **Simple Task Creation**: Easily define tasks with attributes like duration, priority, and resources.
- 🛠️ **Resource Management**: Assign resources with availability windows to tasks.
- 🔄 **Task Dependencies**: Ensure tasks that depend on others are scheduled in the correct order.
- 📅 **Efficient Scheduling**: Automatically schedule tasks while considering their priorities and dependencies.

## 🚀 Quick Start

Get started with `factryengine` with this basic example:

```python
from factryengine import Task, Resource, Scheduler

# Creating a Resource object
resource = Resource(id=1, available_windows=[(0,10)])

# Creating Task objects
task1 = Task(id=1, duration=3, priority=2, constraints=[resource])
task2 = Task(id=2, duration=5, priority=1, constraints=[resource], predecessor_ids=[1])

# Creating a Scheduler object and scheduling the tasks
scheduler = Scheduler(tasks=[task1, task2], resources=[resource])
scheduler_result = scheduler.schedule()
```

In this example, `task1` is scheduled before `task2` as `task2` depends on `task1`, despite its lower priority.

## 📖 Documentation

For more detailed information, check out the [documentation](https://yacobolo.github.io/factryengine/).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📝 License

This project is [MIT](../LICENSE) licensed.
