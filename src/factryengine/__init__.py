import warnings

import numpy as np

from .models import Assignment, Resource, ResourceGroup, Task
from .scheduler.core import Scheduler

# Ignore numpy's UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.*")
