import descarteslabs as dl
import argparse
import subprocess
# I find I need to do the import like this, from same directory:
from compute_ndvi_forCloud import silly

def get_states():
    """ get a list of state names"""
    state_fc = dl.places.prefix('north-america_united-states',
                                placetype='region')

    # parse the feature collection
    state_slugs = [f['properties']['slug'] for f in state_fc['features']]

    # slug example: 'north-america_united-states_new-mexico'
    states = sorted([slug.split('_')[-1] for slug in state_slugs])

    return states

def main(args):
    """run the tasks, with check for what's already done"""

    states = set(get_states())

    if args.dontcheck:
        done = set()
    else:
        path = 'gs://rick-chartrand-test/data'
        filelist = subprocess.check_output('gsutil ls %s/*_statistics.json'
                                           % path, shell=True).splitlines()
        done = {f.split('/')[-1].split('_')[0] for f in filelist}

    print('running %d of %d states' % (len(states - done), len(states)))
    
    # fire off the tasks
    for state in sorted(states - done):
        silly.apply_async([state], {'arraysize': args.arraysize,
                                    'sleeptime': args.sleeptime},
                          queue='myQueue')

    
# we make the module executable, with optional command-line arguments
# example: python run_tasks -sleeptime 50 --dontcheck
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-arraysize', type=int, default=100000,
                        help='make an array of this size')
    parser.add_argument('-sleeptime', type=int, default=100,
                        help='sleep this many seconds')
    parser.add_argument('--dontcheck', action='store_const', const=True,
                        default=False, help="don't check for done tasks")
    args = parser.parse_args()

    main(args)