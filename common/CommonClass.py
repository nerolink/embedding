class Node:
    def __init__(self, count=None, code=None, parent=None):
        self.count = count  # 用于组建霍夫曼树
        self.code = code  # 0 或 1 ,左0   右1
        self.parent = parent  # 父节点

    def __lt__(self, other):
        return self.count < other.count

    def __eq__(self, other):
        return self.count == other.count

    def __gt__(self, other):
        return self.count > other.count


class VocabNode(Node):
    def __init__(self, word):
        super(VocabNode, self).__init__()
        self.word = word
        self.path = None


class StemNode(Node):
    def __init__(self, left, right):
        """
        :param left: Node 类型，
        :param right: Node 类型 left.count<=right.count
        """
        super(StemNode, self).__init__(count=left.count + right.count)
        self.children = [left, right]
        left.parent = self
        right.parent = self
        left.code = 0
        right.code = 1
        self.theta = None
