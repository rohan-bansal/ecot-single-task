from libero.libero import benchmark
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_90']()


num_tasks_in_suite = task_suite.n_tasks

for i in range(num_tasks_in_suite):
    task = task_suite.get_task(i)
    print(f"Task {i}: {task.name}")