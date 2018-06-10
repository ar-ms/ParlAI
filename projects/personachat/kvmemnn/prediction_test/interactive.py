# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which allows local human keyboard input to talk to a trained model.

For example:
`wget https://s3.amazonaws.com/fair-data/parlai/_models/drqa/squad.mdl`
`python examples/interactive.py -m drqa -mf squad.mdl`

Then enter something like:
"Bob is Blue.\nWhat is Bob?"
as the user input (or in general for the drqa model, enter
a context followed by '\n' followed by a question all as a single input.)
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task


import random
import threading
import time

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('--display-prettify', type='bool', default=False,
                        help='Set to use a prettytable when displaying '
                             'examples with text candidates')
    return parser


def interactive(opt):
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        opt = opt.parse_args()
    opt['task'] = 'projects.personachat.kvmemnn.prediction_test.local_human:LocalHumanAgent'

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    def magueule(world):
        time.sleep(1)
        world.agents[0].set_message('Hello !')
        ret = None
        while ret == None:
            ret = world.agents[0].get_res()

    download_thread = threading.Thread(target=magueule, args=[world])
    download_thread.start()
        
    # Show some example dialogs:
    while True:
        world.parley()
        
        if opt.get('display_examples'):
            print("---")
            print(world.display() + "\n~~")
        if world.epoch_done():
            print("EPOCH DONE")
            break


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(
        #task='parlai.agents.local_human.local_human:LocalHumanAgent',
        task='projects.personachat.kvmemnn.prediction_test.local_human:LocalHumanAgent',
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:personachat/kvmemnn/kvmemnn/persona-self_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops1_lins0_model',
    )
    opt = parser.parse_args()

    # add additional model args
    opt['override'] = ['interactive_mode']
    opt['interactive_mode'] = True

    interactive(opt)
