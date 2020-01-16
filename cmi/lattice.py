from context import Context, extend_predicates
from collections import defaultdict
import numpy as np


class LatticeCMI:
    def __init__(self, df_feat, model_interpration, context_set_size=20, min_context_size=100, min_similarity=0.01,
                 max_conjunction_size=5, beam_width=100):
        self.df_feat = df_feat
        self.model_interpration = model_interpration

        self.return_set = set()
        self.candidate_set = defaultdict(list)  # need to keep track of the different levels of predicates

        self.context_set_size = context_set_size
        self.min_context_size = min_context_size
        self.min_similarity = min_similarity
        self.max_conjunction_size = max_conjunction_size
        self.beam_width = beam_width

        self.level = 0

    def sort_subcontext(self, subcontext, glob_mi):
        # based on the model importance delta and size, bucket the subcontext accordingly
        # remember that large valid contexts are returned and removed from futher splits
        contextual_mi = self.model_interpration.on(subcontext)
        delta = np.linalg.norm(glob_mi - contextual_mi, ord=1)

        if delta > self.min_similarity and len(subcontext) > self.min_context_size:
            self.return_set.add(subcontext)
        else:
            self.candidate_set[self.level].append((delta, subcontext))

    def order_candidate_set(self):
        self.candidate_set[self.level] = sorted(self.candidate_set[self.level], reverse=True)

    def inital_candidate_set(self, global_context, glob_mi):
        # Calculate first level predicates and add them to candidate set
        for col_name in self.model_interpration.attribute_cols:
            for feat_value in self.df_feat[col_name].unique():
                subcontext = extend_predicates(global_context, {col_name: feat_value})
                self.sort_subcontext(subcontext, glob_mi)

        # build in order of the predicates with the highest delta, maintain a beam width param
        self.order_candidate_set()

    def check_all_conditions(self):
        ret_set_len = len(self.return_set) < self.context_set_size
        cand_set = len(self.candidate_set[self.level - 1]) > 0
        conj_size = self.level <= self.max_conjunction_size
        return ret_set_len and cand_set and conj_size

    def generate_lattice(self, glob_mi):
        while self.check_all_conditions():
            # build predicates from level 1 and level-1 and store in current level
            for _, prev_context in self.candidate_set[self.level - 1][:self.beam_width]:
                for _, base_context in self.candidate_set[1][:self.beam_width]:

                    # check if the combined predicates do no contradict each other, ie combining q=q1^z=z2 and q=q2
                    # note that overlap is ok q=q2^z=z2 and q=q2, thus the width of the predicate is <= level
                    if base_context.conflicts(prev_context):
                        continue

                    subcontext = extend_predicates(prev_context, base_context.predicates)
                    # if delta is still not high enough and the size dips below min_context_size, remove that context
                    if len(subcontext) < self.min_context_size:
                        continue

                    self.sort_subcontext(subcontext, glob_mi)

            self.order_candidate_set()
            self.level += 1

    def generate_return_set(self):
        self.level = 0
        global_context = Context({}, self.df_feat)
        self.candidate_set[self.level].append(global_context)
        glob_mi = self.model_interpration.on(global_context)

        self.level += 1
        self.inital_candidate_set(global_context, glob_mi)

        self.level += 1
        self.generate_lattice(glob_mi)

        return self.return_set
