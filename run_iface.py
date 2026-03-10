import os
import argparse
import multiprocessing as mp
from itertools import combinations_with_replacement
from halo import Halo
import time

from source.model import IFACE
from source.config import *



def list_available_surfaces():
    data_path = os.path.join(BASEPATH, 'data', 'processed')
    return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])


def _run_pair(task):
    s1, s2, cfg = task
    try:
        print(f"\n[PID {mp.current_process().pid}] Running IFACE between: {s1} and {s2}")
        iface = IFACE(
            surface1_name=s1,
            surface2_name=s2,
            features_list=cfg["features_list"],
            verbose= cfg["verbose"]
           
        )
           
        iface.compute()
        return (s1, s2, "ok")
    except Exception as e:
        print(f"[PID {mp.current_process().pid}] ERROR for {s1} and {s2}: {e}")
        return (s1, s2, f"error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run IFACE model for surface comparison")

    parser.add_argument('--surf1', type=str, required=True, help='Name of first surface or "all"')
    parser.add_argument('--surf2', type=str, required=True, help='Name of second surface or "all"')
    parser.add_argument('--features_list', nargs='+', default=['charge', 'hphob', 'hbond', 'mean_curvature'],
                        help='List of features to use')
    parser.add_argument('--verbose', action='store_true', help='Option for verbose', default=False)
    # parallelism options
    parser.add_argument('--processes', type=int, default=3,
                        help='Parallel processes for "all" mode (0 = use CPU count)')
    parser.add_argument('--start_method', choices=['spawn', 'forkserver', 'fork'], default='forkserver',
                        help='Multiprocessing start method (default=forkserver).')
    
    args = parser.parse_args()


    if args.start_method:
        mp.set_start_method(args.start_method, force=True)

    def build_tasks():
        if args.surf1 == "all" and args.surf2 == "all":
            pairs = combinations_with_replacement(all_surfaces, 2)
        elif args.surf1 == "all":
            pairs = ((s1, args.surf2) for s1 in all_surfaces)
        elif args.surf2 == "all":
            pairs = ((args.surf1, s2) for s2 in all_surfaces)
        else:
            return []
        return [(s1, s2, {
            "features_list": args.features_list, "verbose": args.verbose,
        }) for s1, s2 in pairs]

    # Handle "all" for surf1 or surf2
    if args.surf1 == "all" or args.surf2 == "all":
        all_surfaces = list_available_surfaces()
        print(f"Available surfaces: {len(all_surfaces)} found.")
        spinner = Halo(text="Running IFACE computation", spinner="triangle")
        t0 = time.time()

        try:
            spinner.start()

            tasks = build_tasks()
            if not tasks:
                spinner.warn("No tasks generated -- check your inputs.")
                return

            nproc = args.processes if args.processes > 0 else mp.cpu_count()
            print(f" Launching {len(tasks)} comparisons across {nproc} process(es)...")

            with mp.Pool(processes=nproc) as pool:
                for (s1, s2, status) in pool.imap_unordered(_run_pair, tasks):
                    if status == "ok":
                        print(f"Completed: between {s1} and {s2}")
                    else:
                        print(f"Failed: between {s1} and {s2} -> {status}")

            spinner.succeed("IFACE computation finished.")

        except KeyboardInterrupt:
            spinner.fail("Interrupted (Ctrl+C).")
            raise

        except Exception as e:
            spinner.fail("IFACE computation failed.")
            raise

        finally:
            t1 = time.time()
            print(f"\nAll tasks completed in {(t1 - t0)/60:.1f} minutes.")
    
    else:
        # Single pair mode            
        print(f"Running IFACE between {args.surf1} and {args.surf2}")

        t0 = time.time()

        spinner = Halo(text="Running IFACE computation", spinner="triangle")
        spinner.start()

        try:
            iface = IFACE(
                surface1_name=args.surf1,
                surface2_name=args.surf2,
                features_list=args.features_list,
                verbose=args.verbose
            )

            iface.compute()

            spinner.succeed("IFACE computation finished.")

        except Exception as e:
            spinner.fail("IFACE computation failed.")
            raise e

        finally:
            t1 = time.time()
            print(f"\nTask completed in {(t1 - t0)/60:.1f} minutes.")

if __name__ == "__main__":
    main()
