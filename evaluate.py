import argparse
import os

#from evaluation import test_trees
#from vocab import Vocabulary

def extract_spans(tree):
    answer = list()
    stack = list()
    items = tree.split()
    curr_index = 0
    for item in items:
        if item == ')':
            pos = -1
            right_margin = stack[pos][1]
            left_margin = None
            while stack[pos] != '(':
                left_margin = stack[pos][0]
                pos -= 1
            assert left_margin is not None
            assert right_margin is not None
            stack = stack[:pos] + [(left_margin, right_margin)]
            answer.append((left_margin, right_margin))
        elif item == '(':
            stack.append(item)
        else:
            stack.append((curr_index, curr_index))
            curr_index += 1
    return answer


def extract_statistics(gold_tree_spans, produced_tree_spans):
    gold_tree_spans = set(gold_tree_spans)
    produced_tree_spans = set(produced_tree_spans)
    precision_cnt = sum(list(map(lambda span: 1.0 if span in gold_tree_spans else 0.0, produced_tree_spans)))
    recall_cnt = sum(list(map(lambda span: 1.0 if span in produced_tree_spans else 0.0, gold_tree_spans)))
    precision_denom = len(produced_tree_spans)
    recall_denom = len(gold_tree_spans)
    return precision_cnt, precision_denom, recall_cnt, recall_denom


def f1_score(produced_trees, gold_trees):
    gold_trees = list(map(lambda tree: extract_spans(tree), gold_trees))
    #produced_trees = list(map(lambda tree: extract_spans(tree), produced_trees))
    # TODO: get spans from pmi baseline, $$DONE
    assert len(produced_trees) == len(gold_trees)
    precision_cnt, precision_denom, recall_cnt, recall_denom = 0, 0, 0, 0
    for i, item in enumerate(produced_trees):
        pc, pd, rc, rd = extract_statistics(gold_trees[i], item)
        precision_cnt += pc
        precision_denom += pd
        recall_cnt += rc
        recall_denom += rd
    precision = float(precision_cnt) / precision_denom * 100.0
    recall = float(recall_cnt) / recall_denom * 100.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=str, required=False,
                        help='model path to evaluate')
    parser.add_argument('--produced_path', required=True, default='./',
                        help='the path to produced_tree_spans')
    args = parser.parse_args()
    # TODO: change path: Done
    #ground_truth = [line.strip() for line in open(
    #    os.path.join('/home/zijiao/work/data/mscoco/', 'test_ground-truth.txt'))]
    ground_truth = [line.strip() for line in open(
        os.path.join('/Users/zijiaoyang/Documents/data/mscoco/', 'test_ground-truth.txt'))]

    import pickle
    with open(args.produced_path, 'rb') as f:
        trees = pickle.load(f)
    #trees = [line.strip() for line in open(os.path.join('/home/zijiao/work/VGSNLextend/trees.txt'))]
    f1, precision, recall =  f1_score(trees, ground_truth)
    #print('Model:', args.candidate)
    print('F1 score:', f1)
    print('precision', precision)
    print('recall', recall)
# TODO: check if it works, it worked......
# TODO: generate tree file for test:DONE
# ! change vocab to default