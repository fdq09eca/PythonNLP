import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

class MetaData:
    def __init__(self, ):
        self.identifier = None
        self.title = None
        self.abstract = None
    
    def to_dict(self):
        return {'identifier': self.identifier, 'title': self.title, 'abstract': self.abstract}


class XmlParser:
    @classmethod
    def ceda(cls, xml_fp: str) -> list[MetaData]:
        namespaces = {
            'csw': 'http://www.opengis.net/cat/csw/2.0.2',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'dct': 'http://purl.org/dc/terms/'
        }
        
        
        tree = ET.parse(xml_fp)
        root = tree.find('csw:SearchResults', namespaces=namespaces)
        records = root.findall('csw:Record', namespaces=namespaces)
        mds = []
        for record in records:
            md = MetaData()
            md.identifier = record.find('dc:identifier', namespaces=namespaces).text
            md.title = record.find('dc:title', namespaces=namespaces).text
            md.abstract = record.find('dct:abstract', namespaces=namespaces).text
            mds += [md]
        return mds


class DataSourceBuilder:

    def __init__(self, src_dir: str):
        self.src_dir = src_dir
        self.metadatas = []

    def load_data(self, xml_parser: callable = XmlParser.ceda):
        self.metadatas.clear()
        fps = glob.glob(os.path.join(self.src_dir, '*.xml'))
        
        for fp in fps:
            mds = xml_parser(fp)
            self.metadatas += mds
    
    def build_sqlite(self, db_fp: str = ":memory:") -> int:
        df = self.build_df()
        return df.to_sql(db_fp)
    
    def build_piclke(self, dst_fp: str = "metadata.pkl") -> None:
        df = self.build_df()
        df.to_pickle(dst_fp)

    def build_df(self) -> pd.DataFrame:
        data = [md.to_dict() for md in self.metadatas]
        return pd.DataFrame(data)
    
    def build_csv(self, dst_fp:str = "metadata.csv", index:bool = False) -> None:
        df = self.build_df()
        df.to_csv(dst_fp, index=index)

class Searcher:
    
    pretrained_models = [
        'msmarco-distilbert-base-dot-prod-v3', # this works okay
        'distilbert-base-nli-stsb-mean-tokens', # for short desciptions
        'all-mpnet-base-v2',
        'multi-qa-MiniLM-L6-cos-v1', # specifically trained for semantic search
        'msmarco-bert-base-dot-v5' # could try this one too
    ]
    
    def __init__(self, df: pd.DataFrame, model:str='multi-qa-MiniLM-L6-cos-v1'):
        if df is None:
            raise ValueError('df cannot be None')

        if model not in self.__class__.pretrained_models:
            raise ValueError(f'"{model}" is not a valid model. Please use one of the following: {self.pretrained_models}')
        
        self.df = df
        self.model_name = model
        self.model = SentenceTransformer(model)
        
    def embed_col(self, embed_col:str="abstract", **kwargs) -> None:
        '''embedding the text in the column, pickle the dataframe'''
        embeddings = self.model.encode(self.df.abstract.tolist(), show_progress_bar=True, **kwargs)
        self.df[f'{embed_col}_emb'] = embeddings.tolist()
        plk_fn = self.__get_emb_df_fp(col=embed_col)
        self.df.to_pickle(plk_fn)
    
    def __get_emb_df_fp(self, col:str="abstract"):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        emb_df_dir = os.path.join(current_dir, 'embedded_dataframes')
        if not os.path.exists(emb_df_dir):
            os.mkdir(emb_df_dir)
        
        return os.path.join(
            current_dir, 
            'embedded_dataframes',
            f'{self.model_name}_{col}.pkl')
    
    def get_embedded_df(self, embed_col:str="abstract"):
        plk_fp = self.__get_emb_df_fp(col=embed_col)
        return pd.read_pickle(plk_fp)
    
    def train_model(self):
        raise NotImplementedError()

    def search(self, query:str, col:str="abstract", k:int=5) -> pd.DataFrame:
        '''search for the query'''
        if f'{col}_emb' not in self.df.columns:
            plk_fp = self.__get_emb_df_fp(col=col)
            if os.path.exists(plk_fp):
                self.df = self.get_embedded_df(embed_col=col)
            else:
                self.embed_col(col)
        query_embedding = self.model.encode(query)
        cos_scores = util.cos_sim(query_embedding, self.df[f'{col}_emb'])
        top_results = torch.topk(cos_scores, k=k)
        
        row_idxs = top_results[1].tolist()[0]
        scores = top_results[0].tolist()[0]
        
        results = self.df.iloc[row_idxs].copy()
        results["scores"] = scores
        
        return results


if __name__ == '__main__':
    src_dir = r'C:\Users\ChrisLam\Downloads\CEDA-2022-08-03\2.0.2\CSW'
    builder = DataSourceBuilder(src_dir)
    builder.load_data(xml_parser=XmlParser.ceda)
    df = builder.build_df()
    
    searcher = Searcher(df = df, model = 'multi-qa-MiniLM-L6-cos-v1')
    
    r = searcher.search(
        query = 'find me datasets showing precipitation in the uk for the last 20 years',
        col = "abstract",
        k = 5
    )

    print(r)
    
