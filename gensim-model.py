


from gensim.test.utils import common_corpus
import gensim.corpora as corpora
from gensim.models import ldamodel 
from collections import defaultdict
from random import random, randint
from glob import glob
from math import log
from collections import Counter
import argparse
import time
import gensim.models
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from nltk.tokenize import TreebankWordTokenizer
kTOKENIZER = TreebankWordTokenizer()
kDOC_NORMALIZER = True



class VocabBuilder:
    """
    Creates a vocabulary after scanning a corpus.
    """

    def __init__(self, lang="english", min_length=3, cut_first=100):
        """
        Set the minimum length of words and which stopword list (by language) to
        use.
        """
        self._counts = FreqDist()
        self._stop = set(stopwords.words(lang))
        self._min_length = min_length
        self._cut_first = cut_first

        print(("Using stopwords: %s ... " % " ".join(list(self._stop)[:10])))

    def scan(self, words):
        """
        Add a list of words as observed.
        """

        for ii in [x.lower() for x in words if x.lower() not in self._stop and len(x) >= self._min_length]:
            self._counts[ii] += 1

    def vocab(self, size=5000):
        """
        Return a list of the top words sorted by frequency.
        
        
        if len(self._counts) > self._cut_first + size:
            return list(self._counts.keys())[self._cut_first:(size + self._cut_first)]
        else:
            return list(self._counts.keys())[:size]
        """

        most_common_words = self._counts.most_common(size)
        return [most_common_words[i][0] for i in range(size)]
            


def tokenize_file(filename):
    contents = open(filename, encoding="utf8").read()
    return kTOKENIZER.tokenize(contents)



class Gensim_Model:
    def __init__(self, vocab, files):
        self.vocab = vocab
        self.files = files
        self.id2word = {}

    def create_id2word(self):
        self.id2word = {vocab.index(token):token for token in vocab}
        return self.id2word
    
    def doc2bow(self):
        for document in self.files:
            word_count = Counter()
            tokenized_document = tokenize_file(document)
            word_count.update(tokenized_document)
            yield [(self.vocab.index(token), word_count[token]) for token in tokenized_document if token in self.vocab]

    def run_model(self,num_topics = 5, num_iterations = 1000):
        self.lda = ldamodel.LdaModel(list(self.doc2bow()), id2word=self.id2word, num_topics=5, iterations=1000)

    def report_gensim_output(self,Goutput_filename):
        output = self.lda.print_topics(num_topics=5, num_words=50)
        with open(Goutput_filename + '.txt', "w") as gensim_output:
            for topic_id, words in output:
                gensim_output.write("\n Topic %d\n" %(topic_id))
                for word in words.split(" + "):
                    gensim_output.write("%s \n" %(word.split("*")[1]))
        gensim_output.close()

    def gensim_mallet(self, mallet_path, num_topics, num_iterations):
        self.ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, list(self.doc2bow()), num_topics=num_topics, id2word=self.id2word, iterations = num_iterations)
        
    
    def report_mallet_output(self, Moutput_filename):
        output = self.ldamallet.print_topics(num_topics = 5, num_words = 50)
        with open(Moutput_filename + '.txt', "w") as mallet_output:
            for topic_id, words in output:
                mallet_output.write("\n Topic %d\n" %(topic_id))
                for word in words.split(" + "):
                    mallet_output.write("%s \n" %(word.split("*")[1]))
        mallet_output.close()






if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                           type=str, default=".", required=False)
    argparser.add_argument("--language", help="The language we use",
                           type=str, default="english", required=False)
    argparser.add_argument("--output", help="Where we write results",
                           type=str, default="gensim", required=False)    
    argparser.add_argument("--vocab_size", help="Size of vocabulary",
                           type=int, default=1000, required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=5, required=False)
    argparser.add_argument("--num_iterations", help="Number of iterations",
                           type=int, default=1000, required=False)   

    args = argparser.parse_args()

    vocab_scanner = VocabBuilder(args.language)

    # Create a list of the files
    search_path = "%s/*.txt" % args.doc_dir
    files = glob(search_path)
    assert len(files) > 0, "Did not find any input files in %s" % search_path
    
    # Create the vocabulary
    for ii in files:
        vocab_scanner.scan(tokenize_file(ii))

    vocab = vocab_scanner.vocab(args.vocab_size)
    print((len(vocab), vocab[:10])) 


    gensim_lda = Gensim_Model(vocab, files)
    gensim_lda.create_id2word()
    start_gensim = time.time()
    gensim_lda.run_model(args.num_topics, args.num_iterations)
    print(f"Time taken by gensim is: {time.time()-start_gensim}")
    gensim_lda.report_gensim_output(args.output)



    print("-----------------------------   MALLET IMPLEMENTATION   ------------------------")
    MALLET_HOME = 'mallet-2.0.8/bin/mallet'
    gensim_lda.gensim_mallet('mallet-2.0.8/bin/mallet', args.num_topics, args.num_iterations)
    gensim_lda.report_mallet_output('mallet')
    



    

    