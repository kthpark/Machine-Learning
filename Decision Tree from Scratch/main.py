import pandas as pd
from sklearn.metrics import confusion_matrix


class Node:

    def __init__(self):
        self.left = None
        self.right = None
        self.term = False
        self.label = None
        self.feature = None
        self.value = None

    def set_split(self, feature, value):
        self.feature = feature
        self.value = value

    def set_term(self, label):
        self.term = True
        self.label = label

    def __str__(self):
        if self.term:
            return f'Leaf = {self.label}'
        else:
            return f'Node split by {self.feature} = {self.value}:\n {self.left} {self.right}'


class DecisionTree:
    def __init__(self, leaf_size=1, num_list=None):
        if num_list is None:
            num_list = []
        self.root = Node()
        self.leaf_size = leaf_size
        self.num_list = num_list

    def fit(self, X, y):
        self._run_split(self.root, X, y)

    def predict(self, X):
        y_pred = []
        for i, x in X.iterrows():
            pred = self._chk_sample(self.root, x)
            y_pred.append(pred)
        return y_pred

    def _chk_sample(self, node, x):
        if node.term:
            return node.label
        elif node.feature in self.num_list:
            return self._chk_sample(node.left, x) if x[node.feature] <= node.value else self._chk_sample(node.right, x)
        else:
            return self._chk_sample(node.left, x) if x[node.feature] == node.value else self._chk_sample(node.right, x)

    @staticmethod
    def _gini(lst: list):
        vals, sum_of_prob = set(lst), 0
        for v in vals:
            sum_of_prob += (lst.count(v) / len(lst)) ** 2
        return 1 - sum_of_prob

    def _gini_w(self, split_1, split_2):
        n1, n2 = len(split_1), len(split_2)
        return (n1 * self._gini(split_1) + n2 * self._gini(split_2)) / (n1 + n2)

    def _is_leaf(self, X, y):
        if X.shape[0] <= self.leaf_size or self._gini(y.to_list()) == 0:
            return True
        return all(map(lambda x: len(set(x)) == 1, X.values.T))

    def _chose_split(self, X, y, feature=None, f_value=None):
        g_min, split_1, split_2 = 1, None, None
        for col_name, values in X.iteritems():
            vals = values.unique()
            for v in vals:
                if col_name in self.num_list:
                    idx_1 = X.index[X[col_name] <= v].tolist()
                    idx_2 = X.index[X[col_name] > v].tolist()
                else:
                    idx_1 = X.index[X[col_name] == v].tolist()
                    idx_2 = X.index[X[col_name] != v].tolist()
                wg = self._gini_w(y.iloc[idx_1].tolist(), y.iloc[idx_2].tolist())
                if wg < g_min:
                    g_min = wg
                    feature = col_name
                    f_value = v
                    split_1, split_2 = idx_1, idx_2
        return g_min, feature, f_value, split_1, split_2

    def _run_split(self, node, X, y):
        gmi, feature, f_value, split_1, split_2 = self._chose_split(X, y)

        if self._is_leaf(X, y):
            node.set_term(y.value_counts().idxmax())
            return

        node.set_split(feature, f_value)
        node.left = Node()
        node.right = Node()

        left_X, right_X = X.iloc[split_1].reset_index(drop=True), X.iloc[split_2].reset_index(drop=True)
        left_y, right_y = y.iloc[split_1].reset_index(drop=True), y.iloc[split_2].reset_index(drop=True)

        self._run_split(node.left, left_X, left_y)
        self._run_split(node.right, right_X, right_y)


def main():
    file_name = input().split(" ")
    df_0, df_1 = pd.read_csv(file_name[0], index_col=0), pd.read_csv(file_name[1], index_col=0)
    X_train, X_test, y_train, y_test = df_0.iloc[:, :-1], df_1.iloc[:], df_0['Survived'], df_1['Survived']

    decision_tree = DecisionTree(74, ['Age', 'Fare'])
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print(round(cm[1, 1], 3), round(cm[0, 0], 3))


if __name__ == '__main__':
    main()
