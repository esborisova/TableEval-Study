import os
import yaml
from typing import Dict, List, Optional, Union
import importlib


TaskConfig = Dict[str, str]


class TaskManager:
    def __init__(self, task_path: str = "./") -> None:
        yaml.add_constructor("!function", self.import_function)
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

    def get_task(self, task_name: Union[List[str], str]) -> List[TaskConfig]:
        if isinstance(task_name, str):
            return [self.tasks[task_name]]
        else:
            return [self.tasks[n] for n in task_name]

    def import_function(self, loader, node):
        function_name = loader.construct_scalar(node)
        yaml_path = os.path.dirname(loader.name)

        *module_name, function_name = function_name.split(".")
        if isinstance(module_name, list):
            module_name = ".".join(module_name)
        module_path = os.path.normpath(
            os.path.join(yaml_path, "{}.py".format(module_name))
        )

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        function = getattr(module, function_name)
        return function

    def _init_task(self, root_path: str, file_name: str) -> Optional[TaskConfig]:
        with open(f"{root_path}/{file_name}") as f:
            try:
                output_file = yaml.full_load(f)
                return output_file
            except yaml.YAMLError as exc:
                print(exc)
                return None
