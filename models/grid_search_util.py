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
        data_feed: callable,
        train_main: callable,
        log_dir: str = "./grid_result.csv",
) -> None:
    header_written = False
    with open(log_dir, "w") as f:
        profile_set = profile_generator(scope)
        total_profiles = len(profile_set)
        for (i, profile) in enumerate(profile_set):
            print("**** Profile: [{}/{}] ****".format(i+1, total_profiles))
            print(profile)
            print("Training model...")
            result_lst = train_main(data_feed, **profile)
            print("Done, training results: ")
            for result in result_lst:
                print(result)
                if not header_written:
                    f.write("RUN,")
                    f.write(",".join(result.keys()))
                    f.write("\n")
                    header_written = True
                f.write(str(i)+",")
                f.write(",".join(
                    [str(x).replace(",", ";") for x in result.values()]
                ))
                f.write("\n")
    print("Log stored to {}".format(log_dir))
