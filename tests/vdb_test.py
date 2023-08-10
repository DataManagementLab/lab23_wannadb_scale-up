from typing import List

import pytest

from wannadb.data.data import Attribute, Document, DocumentBase, InformationNugget
from wannadb.data.data import Document, DocumentBase
import random
from wannadb.data.vector_database import compute_embedding_distances, generate_and_store_embedding, vectordb, VECTORDB
from pymilvus import Collection, utility
import re

@pytest.fixture
def documents() -> List[Document]:
    return [
        Document(
            "document-0",
            "Wilhelm Conrad Röntgen (/ˈrɛntɡən, -dʒən, ˈrʌnt-/; [ˈvɪlhɛlm ˈʁœntɡən]; 27 March 1845 – 10 "
            "February 1923) was a German physicist, who, on 8 November 1895, produced and detected "
            "electromagnetic radiation in a wavelength range known as X-rays or Röntgen rays, an achievement "
            "that earned him the first Nobel Prize in Physics in 1901. In honour of his accomplishments, in "
            "2004 the International Union of Pure and Applied Chemistry (IUPAC) named element 111, "
            "roentgenium, a radioactive element with multiple unstable isotopes, after him."
        ),
        Document(
            "document-1",
            "Wilhelm Carl Werner Otto Fritz Franz Wien ([ˈviːn]; 13 January 1864 – 30 August 1928) was a "
            "German physicist who, in 1893, used theories about heat and electromagnetism to deduce Wien's "
            "displacement law, which calculates the emission of a blackbody at any temperature from the "
            "emission at any one reference temperature. He also formulated an expression for the black-body "
            "radiation which is correct in the photon-gas limit. His arguments were based on the notion of "
            "adiabatic invariance, and were instrumental for the formulation of quantum mechanics. Wien "
            "received the 1911 Nobel Prize for his work on heat radiation. He was a cousin of Max Wien, "
            "inventor of the Wien bridge."
        )
    ]

def create_random_float_vector_dimension_1024() -> List[float]:
    return [random.random() for _ in range(1024)]


@ pytest.fixture
def attributes() -> List[Attribute]:
    name_attr = Attribute('name')
    name_attr.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    month_attr = Attribute('month')
    month_attr.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    year_attr = Attribute('year')
    year_attr.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    return [
        name_attr,
        month_attr,
        year_attr
    ]

@pytest.fixture
def information_nuggets(documents) -> List[InformationNugget]:
    nugget_one = InformationNugget(documents[0], 0, 22)
    nugget_one.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_one.__setitem__(key='LabelSignal', value='test1')
    nugget_two = InformationNugget(documents[0], 56, 123)
    nugget_two.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_two.__setitem__(key='LabelSignal', value='test2')
    nugget_three = InformationNugget(documents[1], 165, 176)
    nugget_three.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_three.__setitem__(key='LabelSignal', value='test3')
    nugget_four = InformationNugget(documents[1], 234, 246)
    nugget_four.__setitem__(key='LabelEmbeddingSignal', value=create_random_float_vector_dimension_1024())
    nugget_four.__setitem__(key='LabelSignal', value='test4')
    return [
        nugget_one,
        nugget_two,
        nugget_three,
        nugget_four,    
    ]

@pytest.fixture
def document_base(documents, information_nuggets, attributes) -> DocumentBase:
    # link nuggets to documents
    for nugget in information_nuggets:
        nugget.document.nuggets.append(nugget)

    return DocumentBase(
        documents=documents,
        attributes=attributes
    )

def test_nugget_extraction(document_base, information_nuggets) -> None:
    with VECTORDB as vb:

        for i in utility.list_collections():
                utility.drop_collection(i)

        vb.extract_nuggets(document_base)

        for i in document_base.documents:

            #Check if document-nugget collection was created
            assert re.sub('[^a-zA-Z0-9 \n\.]', '_', i.name) in utility.list_collections()

            #Check if document-nugget collection has the correct number of entities
            collection = Collection(re.sub('[^a-zA-Z0-9 \n\.]', '_', i.name))
            assert collection.num_entities == 2

        #Check if correct nugget data was saved in db
        collection = Collection("document_0")
        collection.load()

        res = collection.query(
         expr="id != '0'",
         offset=0,
         limit=10,
         output_fields=["id", "embedding_type","embedding_value"]
        )

        #LabelSignal1
        assert len(res) == 2
        assert res[0]['id'] == "document_0;0;22"
        assert res[0]['embedding_type'] == "LabelEmbeddingSignal"
        assert len(res[0]['embedding_value']) == 1024
        for i in range(len(res[0]['embedding_value'])):
            assert abs(res[0]['embedding_value'][i] - information_nuggets[0].signals['LabelEmbeddingSignal'].value[i]) < 0.0001

      
        assert res[1]['id'] == "document_0;56;123"
        assert res[1]['embedding_type'] == "LabelEmbeddingSignal"
        assert len(res[1]['embedding_value']) == 1024
        for i in range(len(res[1]['embedding_value'])):
            assert abs(res[1]['embedding_value'][i] - information_nuggets[1].signals['LabelEmbeddingSignal'].value[i]) < 0.0001

        collection = Collection("document_1")
        collection.load()

        res = collection.query(
         expr="id != '0'",
         offset=0,
         limit=10,
         output_fields=["id", "embedding_type","embedding_value"]
        )

        #LabelSignal3
        assert len(res) == 2
        assert res[0]['id'] == "document_1;165;176"
        assert res[0]['embedding_type'] == "LabelEmbeddingSignal"
        assert len(res[0]['embedding_value']) == 1024
        for i in range(len(res[0]['embedding_value'])):
            assert abs(res[0]['embedding_value'][i] - information_nuggets[2].signals['LabelEmbeddingSignal'].value[i]) < 0.0001

      
        assert res[1]['id'] == "document_1;234;246"
        assert res[1]['embedding_type'] == "LabelEmbeddingSignal"
        assert len(res[1]['embedding_value']) == 1024
        for i in range(len(res[1]['embedding_value'])):
            assert abs(res[1]['embedding_value'][i] - information_nuggets[3].signals['LabelEmbeddingSignal'].value[i]) < 0.0001

def test_vector_search(document_base):
    with vectordb() as vb:

        vb.extract_nuggets(document_base)

        #Check if correct nugget data was saved in db
        collection = Collection("Embeddings")
        collection.load()

        vectors_right =[create_random_float_vector_dimension_1024()]

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 0}

        results = collection.search(
            data = vectors_right,
            anns_field="embedding_value",
            param=search_params,
            limit=10,
            )
        assert results[0].ids 
        assert results[0].distances

    
def test_conoa_bson():
    generate_and_store_embedding("D:\\UNI\\wannaDB\\datasets\\corona\\raw-documents")
    compute_embedding_distances()