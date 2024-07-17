import os
try :
    os.chdir("RAY_TESTER")
except:
    pass

from rand_tester import runner
import ray

ray.init(runtime_env={
            "working_dir": str(os.getcwd()),
        })
@ray.remote
def mul_runner():
    return runner()

futures = []
for i in range(10):
    futures.append(mul_runner.remote())

results = ray.get(futures)
print(results)
ray.shutdown()

