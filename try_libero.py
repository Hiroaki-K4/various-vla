import math
import os
import sys
import time

# libero/ is a git submodule directory which causes a namespace package collision.
# Move the editable install finder to the front of sys.meta_path so it takes
# priority over PathFinder, which would otherwise resolve `libero` as a namespace
# package pointing to the submodule root instead of the installed package.
for _i, _finder in enumerate(sys.meta_path):
    if getattr(_finder, "__name__", "") == "_EditableFinder" and "libero" in getattr(
        _finder, "__module__", ""
    ):
        sys.meta_path.insert(0, sys.meta_path.pop(_i))
        break

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


def main():
    task_suite_name = "libero_spatial"
    task_id = 0

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print(f"[info] task {task_id} from {task_suite_name}: {task.language}")
    print(f"[info] bddl file: {task_bddl_file}")

    env = ControlEnv(
        bddl_file_name=task_bddl_file,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    env.seed(0)
    env.reset()

    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    for step in range(500):
        t = step / 20.0
        # action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        action = [
            0.3 * math.sin(t),
            0.3 * math.cos(t),
            0.2 * math.sin(t * 0.5),
            0.0,
            0.0,
            0.0,
            1.0 if step % 40 < 20 else -1.0,
        ]
        obs, reward, done, info = env.step(action)
        env.env.render()
        time.sleep(0.02)  # ~50fps

    env.close()


if __name__ == "__main__":
    main()
