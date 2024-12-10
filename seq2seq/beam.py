import torch

from itertools import count
from queue import PriorityQueue


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS
        self.highest_finished_seq_prob = None

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node, add_padding=False, check_best=False):
        """ Adds a new beam search node to the queue of current nodes """
        if add_padding is True:
            missing = self.max_len - node.length
            node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.nodes.put((score, next(self._counter), node))
        if check_best is True:
            if self.highest_finished_seq_prob is None:
                self.highest_finished_seq_prob = score
            elif score < self.highest_finished_seq_prob:
                self.highest_finished_seq_prob = score

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        # ensure all node paths have the same length for batch ops
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes
    
    def get_current_beams_constant_beam_size(self, check_best=False):
        """ Returns beam_size unfinished nodes with the lowest negative log probability - where effective beam size remains constant"""
        nodes_ret = []
        finished_nodes = PriorityQueue()

        while not self.nodes.empty():
            node = self.nodes.get()
            if node[2].is_finished:
                finished_nodes.put(node)
            else:
                if len(nodes_ret) < self.beam_size:
                    if ((check_best is False) or (self.highest_finished_seq_prob is None) or (node[0] <= self.highest_finished_seq_prob)):
                        nodes_ret.append((node[0], node[2]))                    
        
        self.nodes = finished_nodes
        return nodes_ret

    def get_current_beams_constant_beam_size_v2(self, check_best=False):
        """ Returns beam_size unfinished nodes with the lowest negative log probability - where effective beam size remains constant (v2)"""
        nodes_ret = []
        active_nodes = []
        finished_nodes = PriorityQueue()

        while not self.nodes.empty():
            node = self.nodes.get()
            if node[2].is_finished:
                finished_nodes.put(node)
            else:
                if ((check_best is False) or (self.highest_finished_seq_prob is None) or (node[0] <= self.highest_finished_seq_prob)):
                    active_nodes.append(node)
        
        active_beam_size = self.beam_size - finished_nodes.qsize()
        for i in range(active_beam_size):
            if len(active_nodes) > i:
                nodes_ret.append((active_nodes[i][0], active_nodes[i][2]))
        
        self.nodes = finished_nodes
        return nodes_ret
    
    def get_current_beams_constant_beam_size_v3(self):
        """ Returns beam_size unfinished nodes with the lowest negative log probability - where effective beam size remains constant (v3)"""
        finished_nodes = PriorityQueue()
        nodes_ret = []

        while ((not self.nodes.empty()) and ((finished_nodes.qsize() + len(nodes_ret)) < self.beam_size)):
            node = self.nodes.get()
            if node[2].is_finished:
                finished_nodes.put(node)
            else:
                nodes_ret.append((node[0], node[2]))
        
        self.nodes = finished_nodes
        return nodes_ret
    
    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        node = merged.get()
        node = (node[0], node[2])

        return node

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        nodes = PriorityQueue()
        # Keep track of how many search paths are already finished (EOS)
        finished = self.final.qsize()
        for _ in range(self.beam_size-finished):
            node = self.nodes.get()
            nodes.put(node)
        self.nodes = nodes

    def prune_constant_beam_size(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) - where effective beam size remains constant"""
        finished = []
        nodes = PriorityQueue()
        while not self.nodes.empty():
            node = self.nodes.get()
            if node[2].is_finished:
                finished.append(node)
            else:
                if nodes.qsize() < self.beam_size:
                    nodes.put(node)

        for fn in finished:
            nodes.put(fn)

        self.nodes = nodes

    def prune_constant_beam_size_v2(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) - where effective beam size remains constant (v2)"""
        finished = []
        active = []
        nodes = PriorityQueue()

        while not self.nodes.empty():
            node = self.nodes.get()
            if node[2].is_finished:
                finished.append(node)
            else:
                active.append(node)

        for fn in finished:
            nodes.put(fn)

        for i in range(len(active)):
            if nodes.qsize() < self.beam_size:
                nodes.put(active[i])
            else:
                break

        self.nodes = nodes

    def prune_constant_beam_size_v3(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) - where effective beam size remains constant (v3)"""
        nodes = PriorityQueue()
        for _ in range(self.beam_size):
            node = self.nodes.get()
            nodes.put(node)

        self.nodes = nodes

    def prune_pruning(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) - where effective beam size remains constant and we prune some unfinished hypotheses"""
        nodes = PriorityQueue()
        finished_nodes = []

        while not self.nodes.empty():
            node = self.nodes.get()
            if node[2].is_finished:
                finished_nodes.append(node)
            else:
                if nodes.qsize() < self.beam_size:
                    if ((self.highest_finished_seq_prob is None) or (node[0] <= self.highest_finished_seq_prob)):
                        nodes.put(node)

        for fn in finished_nodes:
            nodes.put(fn)

        self.nodes = nodes

    def prune_pruning_v2(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) - where effective beam size remains constant and we prune some unfinished hypotheses (v2)"""
        nodes = []
        finished_nodes = []
        nodes_queue = PriorityQueue()

        while not self.nodes.empty():
            node = self.nodes.get()
            if node[2].is_finished:
                finished_nodes.append(node)
            else:
                nodes.append(node)

        for fn in finished_nodes:
            nodes_queue.put(fn)

        for _ in range(len(nodes)):
            if nodes_queue.qsize() < self.beam_size:
                if ((self.highest_finished_seq_prob is None) or (node[0] <= self.highest_finished_seq_prob)):
                        nodes_queue.put(node)
            else:
                break

        self.nodes = nodes_queue


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length, is_finished=False):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

        self.is_finished = is_finished

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node 

        params: 
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect
        
        """
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer
        