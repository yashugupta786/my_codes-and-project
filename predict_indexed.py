from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir

ix = open_dir(r"C:\Users\FFDH7900\PycharmProjects\SBU-QA-master\whoosh_search_engine\indexdir")


def find_indexed_document(query_str):
    # query_str is query string
    # query_str = "Establish the cause and extent of the breach"
    # Top 'n' documents as result
    topN = 5
    file_data = ""
    file_path = r'C:\Users\FFDH7900\PycharmProjects\SBU-QA-master\data\combined.txt'

    file_pointer = open(file_path, 'w', errors="ignore")

    # with ix.searcher(weighting=scoring.Frequency) as searcher:
    try :
        searcher =  ix.searcher(weighting=scoring.Frequency)
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query, limit=topN)
        if len(results) < topN:
            topN = len(results)

        for i in range(topN):
            # print(results[i]['title'], str(results[i].score),
            #       results[i]['textdata'])

            file_data = file_data + ". " + results[i]['textdata']

            # print("\n\n\n\n*****************************************************    ")

            # print(results[i]['title'], str(results[i].score))
        file_pointer.write(file_data)
        return file_data
    finally:
        searcher.close()

# print(find_indexed_document("Establish the cause and extent of the breach"))