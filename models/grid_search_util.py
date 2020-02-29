import numpy as np
import itertools
from typing import Dict, List


def profile_generator(scope: Dict[str, List]) -> List[dict]:
    """
    This method generates individual profiles from the source.
    """
    src = scope.copy()
    for k, v in src.items():
        if type(v) not in [list, tuple]:
            src[k] = [v]
    gen = list()
    for vals in itertools.product(*list(src.values())):
        gen.append(dict(zip(src.keys(), vals)))
        # yield dict(zip(src.keys(), vals))
    print("Profiles generated: {}".format(len(gen)))
    return gen


def grid_search(
    scope: dict,
    model_constructor: callable,
    data: List[np.ndarray],
    training_pipeline: callable,
    log_dir: str = "./grid_result.csv",
) -> None:
    header_written = False
    with open(log_dir, "w") as f:
        profile_set = profile_generator(scope)
        total_profiles = len(profile_set)
        for (i, profile) in enumerate(profile_set):
            print("**** Profile: [{}/{}] ****".format(i + 1, total_profiles))
            print("Training model...")
            model = model_constructor(profile)
            perf = training_pipeline(model, data, num_fold=5)
            perf.update(profile)
            print("Done, training results: ")
            print(perf)
            if not header_written:
                f.write("RUN,")
                f.write(",".join(perf.keys()))
                f.write("\n")
                header_written = True
            f.write(str(i) + ",")
            f.write(",".join(
                [str(x).replace(",", ";") for x in perf.values()]
            ))
            f.write("\n")
    print("Log stored to {}".format(log_dir))
