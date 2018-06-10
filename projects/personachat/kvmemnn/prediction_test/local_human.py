# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Agent does gets the local keyboard input in the act() function.
   Example: python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
"""

from parlai.core.agents import Agent
from parlai.core.utils import display_messages

import copy
import time

class LocalHumanAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False
        self.msg = None
        self.res = None
        
    def observe(self, msg):
        self.res = copy.deepcopy(msg)
        print('res: ', self.res)
        #print(display_messages([msg], prettify=self.opt.get('display_prettify', False)))

    def set_message(self, msg):
        print("Setting message..")
        self.msg = msg

    def get_res(self):
        res = self.res
        if self.res != None:
            res = copy.deepcopy(self.res)
            self.res = None
        return res

    def wait_message(self):
        while self.msg == None:
            time.sleep(1)
        print("[*] Info: received message: {}".format(self.msg))
        msg = copy.deepcopy(self.msg)
        self.msg = None
        return msg
    
    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        reply_text = self.wait_message()#input("Enter Your Message: ")
        reply_text = reply_text.replace('\\n', '\n')
        reply['episode_done'] = False
        if '[DONE]' in reply_text:
            reply['episode_done'] = True
            self.episodeDone = True
            reply_text = reply_text.replace('[DONE]', '')
        reply['text'] = reply_text
        return reply

    def episode_done(self):
        return self.episodeDone
