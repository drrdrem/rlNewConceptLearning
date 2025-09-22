import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

class papers_recommendation(object):
    def __init__(self, papers_fea, papers_lab, com_set_fea=False):
        """Create a papers subset recommendation given a question.

        Arguments:
            papers_fea (pandas datafram): all papers ids with features.
            papers_lab (pandas datafram): all papers ids with labels.
        """
        self.all_papers_fea = papers_fea.reset_index(drop=True)
        self.all_papers_lab = papers_lab.reset_index(drop=True)

        self.com_set_fea_flag = com_set_fea
        self.com_set_fea = com_set_fea
        self.length = len(self.all_papers_fea)

    def reocmmend(self, question):
        """Create a papers subset recommendation given a question.

        Arguments:
            question: a desired keyword
        """
        self.all_papers_lab = self.all_papers_lab[self.all_papers_lab['drug']==question].reset_index(drop=True)
        self.all_papers_fea = self.all_papers_fea[self.all_papers_fea['file'].isin(self.all_papers_lab['file'])].reset_index(drop=True)
        
        self.length = len(self.all_papers_fea)
        if self.com_set_fea_flag:
            print('Starting Converting:')
            tmp = self.all_papers_fea['features'].str.split()
            files = self.all_papers_fea['file']
            self.com_set_fea = {files[i]: set(content) for i, content in enumerate(list(tmp)) if not isinstance(content, float)}
            print('End of Converting.')


    def convert2vec(self, corpus, method='tfidf'):
        """Convert corpuses in the papers subset to vectors.

        Arguments:
            corpus: all features.
        Returns:
            Vectors representation of papers based on title/abstract.
        """
        if method=='tfidf':
            tfidf = TfidfVectorizer().fit_transform(corpus)
            return tfidf.toarray()
        elif method=='embedding':
            pass

    def get_all_papers_ids(self):
        """Get all papers ids.

        Returns:
            A list of all papers ids.
        """
        return list(self.all_papers_lab['file'])
        
    def get_paper_fea(self, paper_id):
        """Get the paper features.

        Arguments:
            paper_id (str): id of the paper.

        Returns:
            The features of the paper.
        """
        return self.all_papers_fea[self.all_papers_fea['file']==paper_id]['features'].values[0]
    
    def get_paper_lab(self, paper_id):
        """Get the paper label.

        Arguments:
            paper_id (str): id of the paper.

        Returns:
            The label of the paper.
        """
        return self.all_papers_lab[self.all_papers_lab['file']==paper_id]['lab'].values[0]

    def sim(self, corpus1, corpus2, method='jaccard'):
        """Get Similarity of papers.

        Arguments:
            corpus1 (txt): corpus 1 features.
            corpus2 (txt): corpus 2 features.
            method (str): Similarity methods.
        Returns:
            Similarity between papers.
        """
        if method=='tfidf_cos':
            tfidf = TfidfVectorizer().fit_transform([corpus1, corpus2])
            return 1. - ((tfidf * tfidf.T).A)[0,1]

        elif method=='tfidf_l2':
            tfidf = TfidfVectorizer().fit_transform([corpus1, corpus2])
            embedding1, embedding2 = tfidf.toarray()[0], tfidf.toarray()[1]
            distance = 0.0
            for i in range(len(embedding1)-1):
                distance += (embedding1[i] - embedding2[i])**2
            return np.sqrt(distance)

        elif method=='jaccard':
            a = set(corpus1.split())
            b = set(corpus2.split())
            c = a.intersection(b)
            return 1. - (float(len(c)) / (len(a) + len(b) - len(c)))

        elif method=='jaccard_set_comp':
            c = corpus1.intersection(corpus2)
            return 1. - (float(len(c)) / (len(corpus1) + len(b) - len(corpus2)))

        else:
            print('Please Select among [tfidf_cos, tfidf_l2, jaccard, jaccard_set_comp]')