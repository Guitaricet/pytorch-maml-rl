from metaworld.benchmarks import ML1


class MetaworldWrapper(ML1):
    def reset_task(self, task):
        self.set_task(task)
