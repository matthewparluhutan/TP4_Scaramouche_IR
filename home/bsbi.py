import os
import pickle
import contextlib
import heapq
import time
import math
import re

from .index import InvertedIndexReader, InvertedIndexWriter
from .util import IdMap, sorted_merge_posts_and_tfs
from .compression import StandardPostings, VBEPostings
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm
import nltk as nt
nt.download('stopwords')
nt.download('punkt')

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.doc_length = dict()
        self.avg_doc_length = -1

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []
        self.stemmer = nt.stem.PorterStemmer()
        self.stopword = set(nt.corpus.stopwords.words('english'))
        self.tokenizer = nt.tokenize


    def process_corp(self, corpus):
        return [self.stemmer.stem(x) for x in [w for w in self.tokenizer.word_tokenize(re.sub("[^\w\s]", " ", re.sub("\s+", " ", corpus.lower()))) if not w.lower() in self.stopword]]


    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as qero:
            self.doc_length = qero.doc_length
            self.avg_doc_length = qero.avg_doc_length


    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TDO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            term_dict.setdefault(term_id, {})
            term_dict.get(term_id).setdefault(doc_id, 0)
            term_dict[term_id][doc_id] = term_dict[term_id][doc_id] + 1
        for term_id in sorted(term_dict.keys()):
            post_list = sorted(list(term_dict[term_id]))
            tf_list = [term_dict[term_id][x] for x in post_list]
            index.append(term_id, post_list, tf_list)


    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi sorted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        td_doc = set({})
        files = os.listdir(os.path.join(self.data_dir, block_dir_relative))

        for f in files:
            text = open(os.path.join(self.data_dir, os.path.join(block_dir_relative, f))).read()
            text = self.process_corp(text.lower())
            doc_id = self.doc_id_map.__getitem__(os.path.join(block_dir_relative, f))
            for w in text:
                t_id = self.term_id_map.__getitem__(text)
                td_doc.add((t_id, doc_id))
        return list(td_doc)


    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TDO
        self.load()

        queries = self.process_corp(query)
        res = {}
        resultat = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as qero:
            for i in queries:
                if i not in self.term_id_map: continue
                postings_list_tf= qero.get_postings_list(self.term_id_map[i])
                for j in range(len(postings_list_tf[0])):
                    doc_name = self.doc_id_map[postings_list_tf[0][j]]
                    n = len(self.doc_id_map)
                    tf = postings_list_tf[1][j]
                    df = qero.postings_dict[postings_list_tf[0][j]][1]
                    wtq = math.log(n/df)
                    wtd = 0
                    if tf > 0:
                        wtd = 1 + math.log(tf)
                    if res.get(doc_name):
                        res[doc_name] = res[doc_name] + (wtd * wtq)
                    else:
                        res[doc_name] = (wtd * wtq)
            resultat = list(zip(res.values(), res.keys()))
        resultat = sorted(resultat, key=lambda x: x[0], reverse=True)
        return resultat[:k]

    def retrieve_0_5_tf_max_norm_smooth_idf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = a + (1 - a) * tf(t, D) / max(tf(t, D)) dengan a = 0.5, maka
        w(t, D) = 0.5 + 0.5 * tf(t, D) / max(tf(t, D))

        w(t, Q) = IDF = log (N / (1 + df(t))) + 1

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TDO
        self.load()

        queries = self.process_corp(query)
        res = {}
        resultat = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as qero:
            for i in queries:
                if i not in self.term_id_map: continue
                postings_list_tf= qero.get_postings_list(self.term_id_map[i])
                for j in range(len(postings_list_tf[0])):
                    doc_name = self.doc_id_map[postings_list_tf[0][j]]
                    n = len(self.doc_id_map)
                    tf = postings_list_tf[1][j]
                    max_tf = max(postings_list_tf[1])
                    df = qero.postings_dict[postings_list_tf[0][j]][1]
                    wtq = math.log(n/(1 + df)) + 1
                    wtd = 0.5 + 0.5 * tf / max_tf
                    if res.get(doc_name):
                        res[doc_name] = res[doc_name] + (wtd * wtq)
                    else:
                        res[doc_name] = (wtd * wtq)
            resultat = list(zip(res.values(), res.keys()))
        resultat = sorted(resultat, key=lambda x: x[0], reverse=True)
        return resultat
    
    def retrieve_bm25(self, query, k=10, k1=1.5, b=0.7):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = ((k1 + 1) * tf(t, D)) / (k1 * ((1 - b) + b * docsLength(D) / averageDocsLength) + tf(t, D))

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        self.load()

        queries = self.process_corp(query)
        res = {}
        resultat = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as qero:
            for i in queries:
                if i not in self.term_id_map: continue
                postings_list_tf= qero.get_postings_list(self.term_id_map[i])
                for j in range(len(postings_list_tf[0])):
                    doc_name = self.doc_id_map[postings_list_tf[0][j]]
                    doc_length = self.doc_length[postings_list_tf[0][j]]
                    n = len(self.doc_id_map)
                    tf = postings_list_tf[1][j]
                    df = qero.postings_dict[postings_list_tf[0][j]][1]
                    wtq = math.log(n/df)
                    wtd = ((k1 + 1) * tf) / (k1 * ((1 - b) + (b * doc_length / self.avg_doc_length)) + tf)
                    if res.get(doc_name):
                        res[doc_name] = res[doc_name] + (wtd * wtq)
                    else:
                        res[doc_name] = (wtd * wtq)
            resultat = list(zip(res.values(), res.keys()))
        resultat = sorted(resultat, key=lambda x: x[0], reverse=True)
        return resultat[:k]


    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
