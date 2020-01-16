from operator import and_
from functools import reduce
from copy import copy


# predicates in the form {'q': 1, 'v': 2, ...}
class Context:
    def __init__(self, predicates, df):
        self.predicates = predicates
        self.df = df

    def conflicts(self, context):
        other_predicates = context.predicates
        for key, value in self.predicates.items():
            if key in other_predicates and other_predicates[key] != value:
                return True
        return False

    def get_context_df(self):
        if len(self.predicates) == 0:
            return self.df

        # find rows in the df where the conjunction of predicates hold
        df_predicates = [df_feat[col_name] == col_value for col_name, col_value
                         in self.predicates.items()]
        row_mask = reduce(and_, df_predicates)
        return self.df[row_mask]

    def __len__(self):
        return len(self.df)

    def __str__(self):
        return ' ^ '.join([f'{key} = {value}' for key, value in self.predicates.items()])

    def __repr__(self):
        return self.__str__()


# create a new context with additional predicates
def extend_predicates(context, new_predicates):
    # avoid using deepcopy to prevent redundant df copies, only predicates are updated
    new_context = Context(context.predicates.copy(), context.df)
    new_context.predicates.update(new_predicates)
    return new_context
