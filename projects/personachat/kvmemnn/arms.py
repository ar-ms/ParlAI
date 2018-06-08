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

import pandas as pd

class KVmemNN(nn.Module):
    def __init__(self, vocs, embedding_size=100, n_hops=2):
        super(KVmemNN, self).__init__()

        vocab_size = len(vocs)
        self.n_hops = n_hops
        self.shared_emb = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_size,
                                       sparse=True)
        dialogs_df = pd.read_csv('/home/arms/dialog.csv')
        print(dialogs_df.head())

        sampled_df = dialogs_df.sample(1000)
        self.keys = []
        for key in sampled_df['keys']:
            key_vec = vocs.txt2vec(key)
            key_vec = torch.tensor(key_vec, dtype=torch.long)
            self.keys.append(key_vec)
            #self.encoded_keys.append(self.shared_emb(key_vec))

        self.values = []
        for value in sampled_df['values']:
            value_vec = vocs.txt2vec(value)
            value_vec = torch.tensor(value_vec, dtype=torch.long)
            self.values.append(value_vec)
            #self.encoded_values.append(self.shared_emb(value_vec))
        # TODO: emd.mean(0) ?
        
        self.softmax = nn.Softmax(dim=0)
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.R = nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, xs, candidates, persona, label):
        """
        xs: utterance of the other persona
        label: reply of our persona
        persona: our persona
        candidates: candidates answers called 'distractor responses' in the paper $3.4
        """
        # print('='*50)
        # print(xs)
        # x = torch.tensor(xs, dtype=torch.long)
        # x = self.shared_emb(x).mean(1)
        # print(x)
        # print(x.size())
        # print('='*50)
        # exit(0)
        encoded_keys = []
        encoded_values = []
        for k in self.keys:
            encoded_keys.append(self.shared_emb(k).mean(0))
            
        for v in self.values:
            encoded_values.append(self.shared_emb(v).mean(0))

        encoded_candidates = []
        for cand in candidates:
            cand = torch.tensor(cand, dtype=torch.long)
            encoded_candidates.append(self.shared_emb(cand).mean(0))
            
        encoded_persona = []
        encoded_xs = []
        for p in persona:
            p = torch.tensor(p, dtype=torch.long)
            encoded_persona.append(self.shared_emb(p).mean(0))

            x = torch.tensor(xs, dtype=torch.long)
            encoded_xs.append(self.shared_emb(x).mean(1))

        encoded_keys = torch.stack(encoded_keys)
        encoded_values = torch.stack(encoded_values)
        encoded_candidates = torch.stack(encoded_candidates)
        encoded_persona = torch.stack(encoded_persona)
        encoded_questions = torch.stack(encoded_xs).view(-1, 128)

        x = torch.tensor(xs, dtype=torch.long)
        encoded_question = self.shared_emb(x).mean(1).view(-1, 128)
        q = encoded_question
        # print('keys: {}, values: {}, candidates: {}'.format(encoded_keys.size(),
        #                                                     encoded_values.size(),
        #                                                     encoded_candidates.size())
        #       , 'persona: {}, xs: {}, x: {}'.format(encoded_persona.size(),
        #                                             encoded_questions.size(),
        #                                             encoded_question.size()))

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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, numlayers):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=numlayers,
                          batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, numlayers):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=numlayers,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        emb = self.embedding(input)
        rel = F.relu(emb)
        output, hidden = self.gru(rel, hidden)
        scores = self.softmax(self.out(output))
        return scores, hidden


class ArmsAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    This model is based of Sean Robertson's seq2seq tutorial
    `here <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.
    """

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
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
        if opt.get('numthreads', 1) > 1:
            torch.set_num_threads(1)
        self.id = 'armsagent'

        if not shared:
            
            # set up model from scratch
            self.dict = DictionaryAgent(opt)
            hsz = opt['hiddensize']
            nl = opt['numlayers']

            self.model = KVmemNN(self.dict, hsz)

            """HEHEHE
            # encoder captures the input text
            self.encoder = EncoderRNN(len(self.dict), hsz, nl)
            # decoder produces our output states
            self.decoder = DecoderRNN(len(self.dict), hsz, nl)

            if self.use_cuda:
                self.encoder.cuda()
                self.decoder.cuda()

            if opt.get('numthreads', 1) > 1:
                self.encoder.share_memory()
                self.decoder.share_memory()
            HEHEHE"""
        else:
            # ... copy initialized data from shared table
            self.opt = shared['opt']
            self.dict = shared['dict']
            self.model = shared['model']
            """HEHEHE
            if 'encoder' in shared:
                # hogwild shares model as well
                self.encoder = shared['encoder']
                self.decoder = shared['decoder']
            HEHEHE"""

        if hasattr(self, 'model'):
            # we set up a model for original instance and multithreaded ones
            # self.criterion = nn.NLLLoss()
            self.criterion = nn.CrossEntropyLoss()
            
            # set up optims for each module
            lr = opt['learningrate']
            self.optims = {
                'model': optim.SGD(self.model.parameters(), lr=lr),
            }

            self.longest_label = 1
            self.hiddensize = opt['hiddensize']
            self.numlayers = opt['numlayers']
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            self.START = torch.LongTensor([self.START_IDX])
            if self.use_cuda:
                self.START = self.START.cuda()

        """HEHEHE
        if hasattr(self, 'encoder'):
            # we set up a model for original instance and multithreaded ones
            self.criterion = nn.NLLLoss()

            # set up optims for each module
            lr = opt['learningrate']
            self.optims = {
                'encoder': optim.SGD(self.encoder.parameters(), lr=lr),
                'decoder': optim.SGD(self.decoder.parameters(), lr=lr),
            }

            self.longest_label = 1
            self.hiddensize = opt['hiddensize']
            self.numlayers = opt['numlayers']
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            self.START = torch.LongTensor([self.START_IDX])
            if self.use_cuda:
                self.START = self.START.cuda()
        HEHEHE"""
        self.reset()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def zero_grad(self):
        """Zero out optimizer."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['dict'] = self.dict

        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            shared['model'] = self.model

        return shared

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
            
        # import pprint
        # pprint.pprint(observation)
        # if observation['episode_done']:
        #     exit(0)

        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def predict(self, xs, cands, persona, ys=None, is_training=False):
        """Produce a prediction from our model.

        Update the model using the targets if available.
        """
        if is_training:
            self.zero_grad()
            self.model.train()
            pred = self.model(xs, cands, persona, ys)
            loss = self.criterion(pred.view(1, -1), torch.tensor([19]))
            loss.backward()
            self.update_params()
            return pred.argmax()
        """HEHEHE
        bsz = xs.size(0)
        zeros = Variable(torch.zeros(self.numlayers, bsz, self.hiddensize))
        if self.use_cuda:
            zeros = zeros.cuda()
        starts = Variable(self.START)
        starts = starts.expand(bsz, 1)  # expand to batch size

        if is_training:
            loss = 0
            self.zero_grad()
            self.encoder.train()
            self.decoder.train()
            target_length = ys.size(1)
            # save largest seen label for later
            self.longest_label = max(target_length, self.longest_label)

            encoder_outputs, encoder_hidden = self.encoder(xs, zeros)

            # Teacher forcing: Feed the target as the next input
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            decoder_input = torch.cat([starts, y_in], 1)
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          encoder_hidden)

            scores = decoder_output.view(-1, decoder_output.size(-1))
            loss = self.criterion(scores, ys.view(-1))
            loss.backward()
            self.update_params()

            _max_score, idx = decoder_output.max(2)
            predictions = idx
        else:
            # just predict
            self.encoder.eval()
            self.decoder.eval()
            encoder_output, encoder_hidden = self.encoder(xs, zeros)
            decoder_hidden = encoder_hidden

            predictions = []
            scores = []
            done = [False for _ in range(bsz)]
            total_done = 0
            decoder_input = starts

            for _ in range(self.longest_label):
                # generate at most longest_label tokens
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                _max_score, idx = decoder_output.max(2)
                preds = idx
                decoder_input = preds
                predictions.append(preds)

                # check if we've produced the end token
                for b in range(bsz):
                    if not done[b]:
                        # only add more tokens for examples that aren't done
                        if preds.data[b][0] == self.END_IDX:
                            # if we produced END, we're done
                            done[b] = True
                            total_done += 1
                if total_done == bsz:
                    # no need to generate any more
                    break
            predictions = torch.cat(predictions, 1)
        HEHEHE"""
        return predictions

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        is_training = any(('labels' in obs for obs in observations))

        xs, ys, cands, persona = [None]*4

        # print("="*50)
        # print('XS: ')
        xs = [self.dict.txt2vec(obs['text'].split('\n')[-1]) for obs in observations]
        # print([obs['text'].split('\n')[-1] for obs in observations])
        # print(xs)
        # print("="*50)
        if is_training:
            # print("="*50)
            # print("YS: ")
            ys = [self.dict.txt2vec(obs['labels'][0]) for obs in observations]
            # print([obs['labels'][0] for obs in observations])
            # print(ys)
            # print("="*50)
        # print("="*50)
        # print('candidates: ')
        cands = [self.dict.txt2vec(cand) for obs in observations
                 for cand in obs['label_candidates']]
        # print([cand for obs in observations
        #        for cand in obs['label_candidates']])
        # print(cands)
        # print(len(cands))
        # print("="*50)
        
        persona_label = 'your persona:'
        persona = [self.dict.txt2vec(p[len(persona_label):]) for obs in observations
                   for p in obs['text'].split('\n')
                   if p.find(persona_label) != -1]
        # print([p[len(persona_label):] for obs in observations
        #        for p in obs['text'].split('\n')
        #        if p.find(persona_label) != -1])
        # print(persona)

        """HEHEHE
        # utility function for padding text and returning lists of indices
        # parsed using the provided dictionary
        xs, ys, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, end_idx=self.END_IDX,
            null_idx=self.NULL_IDX, dq=False, eval_labels=True)
        if xs is None:
            return None, None, None, None, None

        # move lists of indices returned above into tensors
        xs = torch.LongTensor(xs)
        if self.use_cuda:
            xs = xs.cuda()
        xs = Variable(xs)

        if ys is not None:
            ys = torch.LongTensor(ys)
            if self.use_cuda:
                ys = ys.cuda()
            ys = Variable(ys)
        HEHEHE"""
        return xs, ys, cands, persona, is_training

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]
        
        xs, ys, cands, persona, is_training = self.vectorize(observations)
        predictions = self.predict(xs, cands, persona, ys, is_training)
        batch_reply[0]['text'] = observations[0]['label_candidates'][predictions.argmax()]
        return batch_reply
        """HEHEHE
        # convert the observations into batches of inputs and targets
        # `labels` stores the true labels returned in the `ys` vector
        # `valid_inds` tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys, labels, valid_inds, is_training = self.vectorize(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply
        HEHEHE"""
        
        #predictions = self.predict(xs, cands, persona, ys, is_training)

        """HEHEHE
        # maps returns predictions back to the right `valid_inds`
        # in the example above, a prediction `world` should reply to `hello`
        PaddingUtils.map_predictions(
            predictions.cpu().data, valid_inds, batch_reply, observations,
            self.dict, self.END_IDX, labels=labels,
            answers=labels, ys=ys.data if ys is not None else None,
            report_freq=self.opt.get('report_freq', 0))

        return batch_reply
        HEHEHE"""
        
    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]
