import os
import yaml
from typing import Dict, List, Optional

TaskConfig = Dict[str, str]


class TaskManager:
    def __init__(self, task_path: str = "./") -> None:
        self.task_path = task_path
        self.tasks: Dict[str, TaskConfig] = {}
        self._init_all_tasks()
        self.task_list = list(self.tasks.keys())

    def _init_all_tasks(self):
        task_dirs = []
        for root, _, files in os.walk(self.task_path):
            yaml_files = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
            if yaml_files:
                for file in yaml_files:
                    task_dirs.append(root)
                    task = self._init_task(root, file)
                    if task:
                        task_dirs.append(task["task_name"])
                        self.tasks[task["task_name"]] = task

    def get_task(self, task_name: List[str]) -> List[TaskConfig]:
        if isinstance(task_name, str):
            return [self.tasks[task_name]]
        if isinstance(task_name, List):
            return [self.tasks[n] for n in task_name]

    def _init_task(self, root_path: str, file_name: str) -> Optional[TaskConfig]:
        with open(f"{root_path}/{file_name}") as f:
            try:
                output_file = yaml.safe_load(f)
                return output_file
            except yaml.YAMLError as exc:
                print(exc)
                return None


if __name__ == "__main__":
    tm = TaskManager()
    breakpoint()
