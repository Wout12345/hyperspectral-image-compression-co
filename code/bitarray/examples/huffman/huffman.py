"""
This library contains useful functionality for working with Huffman trees
and codes.
"""
from __future__ import print_function
import sys
from heapq import heappush, heappop
from bitarray import bitarray

is_py3k = bool(sys.version_info[0] == 3)


class Node(object):
    def __init__(self):
        self.child = [None, None]
        self.symbol = None
        self.freq = None

    def __lt__(self, other):
        # heapq needs to be able to compare the nodes
        return self.freq < other.freq


def huffTree(freq):
    """
    Given a dictionary mapping symbols to thier frequency, construct a Huffman
    tree and return its root node.
    """
    minheap = []
    # create all the leaf nodes and push them onto the queue
    for c in sorted(freq):
        nd = Node()
        nd.symbol = c
        nd.freq = freq[c]
        heappush(minheap, nd)

    # repeat the process until only one node remains
    while len(minheap) > 1:
        # take the nodes with smallest frequencies from the queue
        childR = heappop(minheap)
        childL = heappop(minheap)
        # construct the new internal node and push it onto the queue
        parent = Node()
        parent.child = [childL, childR]
        parent.freq = childL.freq + childR.freq
        heappush(minheap, parent)

    # return the one remaining node, which is the root of the Huffman tree
    return minheap[0]


def huffCode(tree):
    """
    Given a Huffman tree, traverse the tree and return the Huffman code, i.e.
    a dictionary mapping symbol to bitarrays.
    """
    result = {}

    def traverse(nd, prefix=bitarray()):
        if nd.symbol is None: # parent, so traverse each of the children
            for i in range(2):
                traverse(nd.child[i], prefix + bitarray([i]))
        else: # leaf
            result[nd.symbol] = prefix

    traverse(tree)
    return result


def insert_symbol(tree, ba, sym):
    """
    Insert symbol into a tree at the position described by the bitarray,
    creating nodes as necessary.
    """
    nd = tree
    for k in ba:
        prev = nd
        nd = nd.child[k]
        if nd and nd.symbol:
            raise ValueError("ambiguity")
        if not nd:
            nd = Node()
            prev.child[k] = nd
    if nd.symbol or nd.child[0] or nd.child[1]:
        raise ValueError("ambiguity")
    nd.symbol = sym


def make_tree(codedict):
    """
    Create a tree from the given code dictionary, and return its root node.
    Unlike trees created by huffTree, all nodes will have .freq set to None.
    """
    tree = Node()
    for sym, ba in codedict.items():
        insert_symbol(tree, ba, sym)
    return tree


def traverse(tree, it):
    """
    Traverse tree until a leaf node is reached, and return its symbol.
    This function takes an iterator on which next() is called during each
    step of traversing.
    """
    nd = tree
    while 1:
        nd = nd.child[next(it)]
        if not nd:
            raise ValueError("prefix code does not match data in bitarray")
            return None
        if nd.symbol is not None:
            return nd.symbol
    if nd != tree:
        raise ValueError("decoding not terminated")
        return None


def decode(tree, bitsequence):
    """
    Given a tree and a bitsequence, decode the bitsequence and return a
    list of symbols.
    """
    res = []
    it = iter(bitsequence)
    while True:
        try:
            r = traverse(tree, it)
        except StopIteration:
            break
        res.append(r)
    return res


def write_dot(tree, fn, binary=False):
    """
    Given a tree (which may or may not contain frequencies), write
    a graphviz '.dot' file with a visual representation of the tree.
    """
    special_ascii = {' ': 'SPACE', '\n': 'LF', '\r': 'CR', '\t': 'TAB',
                     '\\': r'\\', '"': r'\"'}
    def disp_char(c):
        if is_py3k and isinstance(c, int):
            c = chr(c)
        if binary:
            return 'x%02x' % ord(c)
        else:
            if special_ascii:
                res = special_ascii.get(c, c)
                assert res.strip(), repr(c)
                return res

    def disp_freq(f):
        if f is None:
            return ''
        return '%d' % f

    with open(fn, 'w') as fo:    # dot -Tpng tree.dot -O
        def write_nd(fo, nd):
            if nd.symbol: # leaf node
                a, b = disp_freq(nd.freq), disp_char(nd.symbol)
                fo.write('  %d  [label="%s%s%s"];\n' %
                         (id(nd), a, ': ' if a and b else '', b))
            else: # parent node
                fo.write('  %d  [shape=circle, style=filled, '
                         'fillcolor=grey, label="%s"];\n' %
                         (id(nd), disp_freq(nd.freq)))

            for k in range(2):
                if nd.child[k]:
                    fo.write('  %d->%d;\n' % (id(nd), id(nd.child[k])))

            for k in range(2):
                if nd.child[k]:
                    write_nd(fo, nd.child[k])

        fo.write('digraph BT {\n')
        fo.write('  node [shape=box, fontsize=20, fontname="Arial"];\n')
        write_nd(fo, tree)
        fo.write('}\n')


def print_code(freq, codedict):
    """
    Given a frequency map (dictionary mapping symbols to thier frequency)
    and a codedict, print them in a readable form.
    """
    special_ascii = {0: 'NULL', 9: 'TAB', 10: 'LF', 13: 'CR', 127: 'DEL'}
    def disp_char(i):
        if 32 <= i < 127:
            return repr(chr(i))
        return special_ascii.get(i, '')

    print(' symbol     char    hex   frequency     Huffman code')
    print(70 * '-')
    for c in sorted(codedict, key=lambda c: (freq[c], c), reverse=True):
        i = c if is_py3k else ord(c)
        print('%7r     %-4s    0x%02x %10i     %s' % (
            c, disp_char(i),
            i, freq[c], codedict[c].to01()))


def test():
    freq = {'a': 10, 'b': 2, 'c': 1}
    tree = huffTree(freq)
    code = huffCode(tree)
    assert len(code['a']) == 1
    assert len(code['b']) == len(code['c']) == 2

    code = {'a': bitarray('0'),
            'b': bitarray('10'),
            'c': bitarray('11')}
    tree = make_tree(code)
    txt = 'abca'
    a = bitarray()
    a.encode(code, txt)
    assert a == bitarray('010110')
    assert decode(tree, a) == ['a', 'b', 'c', 'a']


if __name__ == '__main__':
    test()
