from wannadb.data.vector_database import vectordb
from pymilvus import Collection
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from wannadb.data.data import DocumentBase
from wannadb.data.signals import  CombinedEmbeddingSignal
from wannadb.data.vector_database import generate_new_index


def test_distance_cosine(document_base: DocumentBase, index_params: dict, search_params: dict): 
    ''''
    with vectordb() as vb:
        collection = Collection('embeddings')
        collection.release()
    generate_new_index(index_params)
    '''

    with vectordb() as vb:

        collection = Collection('adjusted_embeddings')
        print(f"Anzahl der Nuggets: {collection.num_entities}")
        collection.load()

        embedding_list = []
        attribute = document_base.attributes[0]
        for signal in vb._embedding_identifier:
            if  signal in attribute.signals:
                embedding_list.append(attribute[signal])
                print(f"Attribute Signal: {signal}")

        if len(embedding_list) > 0:
            combined_embedding =np.concatenate(embedding_list)

        #assert len(combined_embedding) == 1024*3
        #print(f"Dimensionality Attribute: {len(combined_embedding)}")

        #Query vector embeddings und überprüfe übereinstimmung mit custom Datenstruktur
        res = collection.query(
            expr = "dbid  != -1",
            offset = 0,
            limit = 10, 
            output_fields = ["id", "document_id", "embedding_value"],
            )
        
        print(f"Es sollten 9 Ergebnisse in der Query enthalten sein: {len(res)}")
        #print(f"Der erste Query-Eintrag sieht so aus: {res[0]}")

        def vectors_are_equal(vector1, vector2):
            # Überprüfe, ob die Längen der Vektoren gleich sind
            if len(vector1) != len(vector2):
                return False
            
            # Vergleiche die Elemente der Vektoren
            for i in range(len(vector1)):
                if vector1[i] != vector2[i]:
                    return False
            
            # Wenn alle Elemente übereinstimmen, gib True zurück
            return True
        
        #Distance Computation VDB
        result_docs = vb.compute_inital_distances(combined_embedding,document_base)

        vdb_results = {}
        custom_results = {}
        
        for i in res:

            #Check if elements and dimensionality are the same
            vdb_value=i['embedding_value']
            custom_value = document_base.documents[i['document_id']].nuggets[i['id']][CombinedEmbeddingSignal]
            #assert vectors_are_equal(vdb_value, custom_value)
            #assert len(vdb_value) == 1024*3
            #assert len(custom_value) == 1024*3
            #assert 'LabelEmbeddingSignal' in document_base.documents[i['document_id']].nuggets[i['id']].signals

            #Compute Custom Distances
            matrix = np.vstack((combined_embedding, vdb_value))
            cosine_distance = cosine_distances(matrix)
            custom_results[(i['id'],i['document_id'])] = cosine_distance[0, 1]


            #Get VDB Result
            for j in result_docs[0]:
                if (j.entity.get('id') == i['id']) and (j.entity.get('document_id') == i['document_id']):
                    vdb_results[(j.entity.get('id'), j.entity.get('document_id'))] = 1-j.distance


        # Dictionary nach den Werten sortieren
        sorted_vdb_results = dict(sorted(vdb_results.items(), key=lambda item: item[1]))
        sorted_custom_results = dict(sorted(custom_results.items(), key=lambda item: item[1]))


        # Geordnete Einträge ausgeben
        print("VDB Results Cosine")
        for key, value in sorted_vdb_results.items():
            print(f'{key}: {value}')

        print("Custom Results Cosine")
        for key, value in sorted_custom_results.items():
            print(f'{key}: {value}')

        collection.release()

def test_distance_ip(document_base: DocumentBase, index_params: dict, search_params: dict): 

    with vectordb() as vb:

        collection = Collection('embeddings')
        collection.release()

    generate_new_index(index_params)

    with vectordb() as vb:
        collection = Collection('embeddings')
        collection.load()

        embedding_list = []
        attribute = document_base.attributes[0]
        for signal in vb._embedding_identifier:
            if  signal in attribute.signals:
                embedding_list.append(attribute[signal])
                print(f"Attribute Signal: {signal}")
            else:
                embedding_list.append(np.zeros(1024))

        if len(embedding_list) > 0:
            combined_embedding =np.concatenate(embedding_list)

        assert len(combined_embedding) == 1024*3

        #Query vector embeddings und überprüfe übereinstimmung mit custom Datenstruktur
        res = collection.query(
            expr = "dbid  != -1",
            offset = 0,
            limit = 10, 
            output_fields = ["id", "document_id", "embedding_value"],
            )     
        
        #Distance Computation VDB
        result_docs = vb.compute_inital_distances(combined_embedding,document_base, search_params)

        vdb_results = {}
        custom_results = {}
        
        for i in res:

            #Check if elements and dimensionality are the same
            vdb_value=i['embedding_value']

            #Compute Custom Distances
            ip = np.dot(combined_embedding,vdb_value)
            custom_results[(i['id'],i['document_id'])] = ip


            #Get VDB Result
            for j in result_docs[0]:
                if (j.entity.get('id') == i['id']) and (j.entity.get('document_id') == i['document_id']):
                    vdb_results[(j.entity.get('id'), j.entity.get('document_id'))] = j.distance


        # Dictionary nach den Werten sortieren
        sorted_vdb_results = dict(sorted(vdb_results.items(), key=lambda item: item[1]))
        sorted_custom_results = dict(sorted(custom_results.items(), key=lambda item: item[1]))


        # Geordnete Einträge ausgeben
        print("VDB Results - IP")
        for key, value in sorted_vdb_results.items():
            print(f'{key}: {value}')

        print("Custom Results -IP")
        for key, value in sorted_custom_results.items():
            print(f'{key}: {value}')

        collection.release()
   

####################################################################################################
#Test Cosine and IP similarity ranking
####################################################################################################

with open("C:/Users/Pascal/Desktop\WannaDB/lab23_wannadb_scale-up/cache/exp-2-corona-preprocessed.bson", "rb") as file:
    document_base = DocumentBase.from_bson(file.read())

if not document_base.validate_consistency():
    print("Document base is inconsistent!")
    raise

index_ip = {
    "metric_type":"IP",
    "index_type":"FLAT"
    }

search_ip = {
        "metric_type": "IP", 
        "offset": 0, 
        "ignore_growing": False
        }

index_cosine = {
        "metric_type":"COSINE",
        "index_type":"FLAT"
        }

search_cosine = {
        "metric_type": "COSINE", 
        "offset": 0, 
        "ignore_growing": False
        }

####################################################################################################
#Load VDB if necessary
####################################################################################################

#with vectordb() as vb:
 #  vb.extract_nuggets(document_base)

####################################################################################################
#Compute distances
####################################################################################################

test_distance_cosine(document_base, index_cosine, search_cosine)
#test_distance_ip(document_base, index_cosine, search_cosine)


