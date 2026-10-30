"""
week01_embeddings —— Word2Vec 最小可运行示例

流程：
  1. 文本读取      read_corpus()
  2. 词频统计      count_words()
  3. Word2Vec 训练 train_word2vec()
  4. 5 组近邻词查询 query_neighbors()
"""

import argparse
import collections
from pathlib import Path

from gensim.models import Word2Vec


# ----------------------------------------------------------------------------
# 1. 文本读取 + 分词
# ----------------------------------------------------------------------------
def read_corpus(path: str, lang: str = "en"):
    """读取文本文件，返回「句子列表」，每个句子是「词列表」。

    Word2Vec 的输入格式就是 [ [w1, w2, ...], [w3, w4, ...], ... ]
    - 英文：按行切分、转小写、按空格分词（先清掉标点）
    - 中文：用 jieba 分词（中文没有空格，必须先分词）
    """
    text = Path(path).read_text(encoding="utf-8")
    sentences = []
    if lang == "zh":
        import jieba
        jieba.setLogLevel(20)  # 关掉 jieba 的提示日志
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = [w for w in jieba.lcut(line) if w.strip()]
            if tokens:
                sentences.append(tokens)
    else:
        for line in text.splitlines():
            line = line.strip().lower()
            if not line:
                continue
            # 去掉常见标点，避免 "king." 和 "king" 被当成两个词
            for p in [",", ".", ";", ":", "!", "?", '"', "'"]:
                line = line.replace(p, " ")
            tokens = line.split()
            if tokens:
                sentences.append(tokens)
    return sentences


# ----------------------------------------------------------------------------
# 2. 词频统计
# ----------------------------------------------------------------------------
def count_words(sentences):
    counter = collections.Counter()
    for sent in sentences:
        counter.update(sent)
    return counter


# ----------------------------------------------------------------------------
# 3. Word2Vec 训练
# ----------------------------------------------------------------------------
def train_word2vec(sentences, vector_size=100, window=5, min_count=1, epochs=200):
    """skip-gram (sg=1) 对小语料、低频词更友好。

    vector_size : 词向量维度
    window      : 上下文窗口大小（左右各看几个词）
    min_count   : 词最少出现次数，低于此值被丢弃（正式语料常用 5）
    epochs      : 训练轮数，小语料要调大才能收敛
    """
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,         # 1 = skip-gram, 0 = CBOW
        workers=1,    # 小数据用 1，结果可复现
        epochs=epochs,
        seed=42,
    )
    return model


# ----------------------------------------------------------------------------
# 4. 5 组近邻词查询
# ----------------------------------------------------------------------------
def query_neighbors(model, queries, topn=5):
    """对每个目标词，输出向量最相近的 topn 个词及余弦相似度。"""
    print("\n========== 5 组近邻词查询（most_similar） ==========")
    for word in queries:
        if word not in model.wv:
            print(f"  [{word}]  不在词表中（可能被 min_count 过滤了）")
            continue
        similar = model.wv.most_similar(word, topn=topn)
        print(f"  {word}:")
        for w, score in similar:
            print(f"      {w:<12} {score:.4f}")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="week01 Word2Vec 最小示例")
    parser.add_argument("--corpus", default=str(Path(__file__).parent / "corpus.txt"),
                        help="语料文件路径（默认同目录 corpus.txt）")
    parser.add_argument("--lang", choices=["en", "zh"], default="en",
                        help="en=英文按空格分词(默认)；zh=中文用 jieba 分词")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model-out", default=str(Path(__file__).parent / "week01_embeddings.model"),
                        help="训练好的模型保存路径")
    args = parser.parse_args()

    # 1) 文本读取
    print(">>> 1. 读取文本：", args.corpus)
    sentences = read_corpus(args.corpus, lang=args.lang)
    total_tokens = sum(len(s) for s in sentences)
    vocab = set(w for s in sentences for w in s)
    print(f"    句子数={len(sentences)}  总词数={total_tokens}  词表大小={len(vocab)}")

    # 2) 词频统计
    print("\n>>> 2. 词频统计（Top 15）")
    counter = count_words(sentences)
    for w, c in counter.most_common(15):
        print(f"    {w:<12} {c}")

    # 3) Word2Vec 训练
    print("\n>>> 3. 训练 Word2Vec ...")
    model = train_word2vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
    )
    model.save(args.model_out)
    print(f"    模型已保存：{args.model_out}")

    # 4) 5 组近邻词查询
    queries = ["king", "cat", "car", "happy", "run"]
    # 若目标词不在词表，自动退回词频最高的 5 个词，保证一定有输出
    queries = [w for w in queries if w in model.wv] or [w for w, _ in counter.most_common(5)]
    query_neighbors(model, queries, topn=5)


if __name__ == "__main__":
    main()
