# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils
from parlai.core.thread_utils import SharedTable

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import copy
import pickle
import pandas as pd


def load_cands(path):
    """Load global fixed set of candidate labels that the teacher provides
    every example (the true labels for a specific example are also added to
    this set, so that it's possible to get the right answer).
    """
    if path is None:
        return None
    cands = []
    lines_have_ids = False
    cands_are_replies = False
    cnt = 0
    with open(path) as read:
        for line in read:
            line = line.strip().replace('\\n', '\n')
            if len(line) > 0:
                cnt = cnt + 1
                # If lines are numbered we strip them of numbers.
                if cnt == 1 and line[0:2] == '1 ':
                    lines_have_ids = True
                # If tabs then the label_candidates are all the replies.
                if '\t' in line and not cands_are_replies:
                    cands_are_replies = True
                    cands = []
                if lines_have_ids:
                    space_idx = line.find(' ')
                    line = line[space_idx + 1:]
                    if cands_are_replies:
                        sp = line.split('\t')
                        if len(sp) > 1 and sp[1] != '':
                            cands.append(sp[1])
                    else:
                        cands.append(line)
                else:
                    cands.append(line)
    return cands

class KVmemNN(nn.Module):
    def __init__(self, vocs, embedding_size=100, n_hops=2):
        super(KVmemNN, self).__init__()

        #print(len(load_cands('/home/arms/research/ParlAI/data/ConvAI2/train_self_original.txt')))
        self.embedding_size = embedding_size
        vocab_size = len(vocs)
        self.n_hops = n_hops
        self.shared_emb = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_size,
                                       sparse=True).cuda()
        dialogs_df = pd.read_csv('/home/arms/dialog.csv')

        sampled_df = dialogs_df.sample(1000)
        self.keys = []
        self.values = []
        for i, el in sampled_df.iterrows():
            key_vec = vocs.txt2vec(el['keys'])
            key_vec = torch.tensor(key_vec, dtype=torch.long).cuda()
            self.keys.append(key_vec)
            value_vec = vocs.txt2vec(el['values'])
            value_vec = torch.tensor(value_vec, dtype=torch.long).cuda()
            self.values.append(value_vec)
        
        self.softmax = nn.Softmax(dim=0).cuda()
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
        self.R = nn.Linear(embedding_size, embedding_size, bias=False).cuda()

    def forward(self, xs, candidates, persona, label):
        """
        xs: utterance of the other persona
        label: reply of our persona
        persona: our persona
        candidates: candidates answers called 'distractor responses' in the paper $3.4
        """
        encoded_keys = []
        encoded_values = []
        for k in self.keys:
            encoded_keys.append(self.shared_emb(k).mean(0))
            
        for v in self.values:
            encoded_values.append(self.shared_emb(v).mean(0))

        encoded_candidates = []
        for cand in candidates:
            cand = torch.tensor(cand, dtype=torch.long).cuda()
            encoded_candidates.append(self.shared_emb(cand).mean(0))
            
        encoded_persona = []
        encoded_xs = []
        for p in persona:
            p = torch.tensor(p, dtype=torch.long).cuda()
            encoded_persona.append(self.shared_emb(p).mean(0))

            x = torch.tensor(xs, dtype=torch.long).cuda()
            encoded_xs.append(self.shared_emb(x).mean(1))

        encoded_keys = torch.stack(encoded_keys)
        encoded_values = torch.stack(encoded_values)
        encoded_candidates = torch.stack(encoded_candidates)
        encoded_persona = torch.stack(encoded_persona)
        encoded_questions = torch.stack(encoded_xs).view(-1, self.embedding_size)

        x = torch.tensor(xs, dtype=torch.long).cuda()
        encoded_question = self.shared_emb(x).mean(1).view(-1, self.embedding_size)
        q = encoded_question

        sim = self.cosine(encoded_questions, encoded_persona)
        softSim = self.softmax(sim)
        test = torch.mm(softSim.view(1, -1), encoded_persona)
        q = self.R(test)

        tmp = torch.mm(encoded_keys, q.view(-1, 1))
        ph = self.softmax(tmp)
        #eprint("ph:",ph.size())
        test = torch.mm(ph.view(1, -1), encoded_values)
        #o = torch.sum(ph*encoded_values)
        #eprint("o:", o.size())
        #eprint("*"*50)
        #q = self.R[0] * (q+o)
        q = self.R(test)


        preds = torch.mm(encoded_candidates, q.view(-1, 1))
        #preds = q*encoded_cands
        #eprint("preds:", preds.size())
        #preds = F.softmax(preds, dim=0)
        preds = self.softmax(preds)

        #print(preds)
        #print(preds.argmax())
        return preds
    #return (encoded_candidates[preds.argmax()], encoded_candidates[-1])


class ArmsAgent(Agent):
    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('KvMemNN Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=1,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('--gpu', type=int, default=-1,
                           help='which GPU device to use')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.001,
                           help='Report frequency of prediction during eval.')
        ArmsAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        # initialize defaults first
        super().__init__(opt, shared)

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()
        #if opt.get('numthreads', 1) > 1:
        #    torch.set_num_threads(1)
        self.id = 'armsagent'

        if not shared:
            
            # set up model from scratch
            self.dict = DictionaryAgent(opt)
            emb_size = opt['embeddingsize']
            nl = opt['numlayers']

            self.model = KVmemNN(self.dict, emb_size)
            print(self.model.parameters)
            self.model.share_memory()
            print('[*] Info: loading dialog file...')
            all_cands_df = pd.read_csv('/home/arms/dialog.csv')
            all_cands = all_cands_df['keys'].tolist() + all_cands_df['values'].tolist()
            self.all_cands = [self.dict.txt2vec(cand) for cand in all_cands]
            print('[*] Info: cands ready...')
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']
            self.model = shared['model']
            self.all_cands = shared['all_cands']
            
        if hasattr(self, 'model'):
            # we set up a model for original instance and multithreaded ones
            # self.criterion = nn.NLLLoss()
            self.criterion = nn.CrossEntropyLoss().cuda()
            
            # set up optims for each module
            lr = opt['learningrate']
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

            # # we use END markers to end our output
            # self.END_IDX = self.dict[self.dict.end_token]
            # # get index of null token from dictionary (probably 0)
            # self.NULL_IDX = self.dict[self.dict.null_token]
            # # we use START markers to start our output
            # self.START_IDX = self.dict[self.dict.start_token]
            # self.START = torch.LongTensor([self.START_IDX])
            # if self.use_cuda:
            #     self.START = self.START.cuda()

        self.reset()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def zero_grad(self):
        """Zero out optimizer."""
        #for optimizer in self.optims.values():
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        #for optimizer in self.optims.values():
        self.optimizer.step()

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict
        shared['all_cands'] = self.all_cands
        shared['model'] = self.model
        return shared

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']

        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def predict(self, xs, cands, persona, ys=None, is_training=False):
        """Produce a prediction from our model.

        Update the model using the targets if available.
        """
        if is_training:
            self.model.train()
            self.zero_grad()
            pred = self.model(xs, cands, persona, ys)
            loss = self.criterion(pred.view(1, -1), torch.tensor([ys]).cuda())
            loss.backward()
            self.update_params()
            return pred.argmax()
        else:
            self.model.eval()
            pred = self.model(xs, cands, persona, ys)
            return pred.argmax()

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        is_training = any(('labels' in obs for obs in observations))

        xs, ys, cands, persona = [None]*4

        xs = [self.dict.txt2vec(obs['text'].split('\n')[-1]) for obs in observations]
        if is_training:
            #ys = list(observations[0]['label_candidates']).index(
            #    observations[0]['labels'][0])
            #ys = [self.dict.txt2vec(obs['labels'][0]) for obs in observations]
            ys = 19
            cands = [self.dict.txt2vec(cand) for obs in observations
                     for cand in obs['label_candidates']]
        else:
            print('*'*20)
            print([obs['text'].split('\n')[-1] for obs in observations])
            print('*'*20)
        persona_label = 'your persona:'
        persona = [self.dict.txt2vec(p[len(persona_label):]) for obs in observations
                   for p in obs['text'].split('\n')
                   if p.find(persona_label) != -1]
        if persona == []:
            import pprint
            pprint.pprint(observations)

        return xs, ys, cands, persona, is_training

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        xs, ys, cands, persona, is_training = self.vectorize(observations)
        if is_training:
            pred = self.predict(xs, cands, persona, ys, is_training)
            
            batch_reply[0]['text'] = observations[0]['label_candidates'][pred]

        else:
            pred = self.predict(xs, self.all_cands, persona, ys, is_training)
            batch_reply[0]['text'] = self.dict.vec2txt(self.all_cands[pred])
            print(pred,
                  observations[0]['text'],
                  batch_reply)

        return batch_reply
        
    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def shutdown(self):
        #"""Save the state of the model when shutdown."""
        super().shutdown()

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path
        if path and hasattr(self, 'model'):
            data = {}
            data['model'] = self.model.state_dict()
            data['optimizer'] = self.optimizer.state_dict()
            data['opt'] = self.opt
            with open(path, 'wb') as handle:
                torch.save(data, handle)
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Return opt and model states."""
        print('Loading existing model params from ' + path)
        data = torch.load(path, map_location=lambda cpu, _: cpu)
        self.model.load_state_dict(data['model'])
        self.reset()
        self.optimizer.load_state_dict(data['optimizer'])
        self.opt = self.override_opt(data['opt'])
