from corpus import Document, WordPOSCorpus, WordCorpus, POSCorpus
from hmm import HMM
from unittest import TestCase, main
from evaluator import compute_cm
from random import shuffle, seed
import sys


class HMMTest(TestCase):
    u"""Tests for the HMM sequence labeler."""

    def split_np_chunk_corpus(self, corpus, document_class):
        """Split the wsj corpus into training, dev, and test sets"""
        sentences = corpus(
            'np_chunking_wsj', document_class=document_class)

        seed(hash("np_chunk"))
        shuffle(sentences)
        return (sentences[:8936], sentences[8936:])

    def test_np_chunk_wordpos(self):
        """Test NP chunking with word and postag feature"""
        train, test = self.split_np_chunk_corpus(WordPOSCorpus,Document)
        classifier = HMM('wordpos')
        classifier.train(train)
        test_result = compute_cm(classifier, test)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i >= .90 for i in f1),
                        'not all greater than 90.0%')

    def test_np_chunk_pos(self):
        """Test NP chunking with postag feature"""
        train, test = self.split_np_chunk_corpus(POSCorpus,Document)
        classifier = HMM('pos')
        classifier.train(train)
        test_result = compute_cm(classifier, test)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i >= .90 for i in f1),
                        'not all greater than 90.0%')

    def test_np_chunk_word(self):
        """Test NP chunking with word feature"""
        train, test = self.split_np_chunk_corpus(WordCorpus,Document)
        classifier = HMM('word')
        classifier.train(train)
        test_result = compute_cm(classifier, test)
        _, _, f1, accuracy = test_result.print_out()
        self.assertGreater(accuracy, 0.55)
        self.assertTrue(all(i >= .90 for i in f1),
                        'not all greater than 90.0%')

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
