from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

def ce(query,topk,passages):
    # Initialize Cross-Encoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Replace with the actual cross-encoder model name or path
    #cross-encoder/ms-marco-MiniLM-L-6-v2
    #cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    print(query)

    candidate_passages = [passages[i] for i in topk]

    # Generate query-passage pairs
    query_passage_pairs = [(query, passage) for passage in candidate_passages]

    # Compute similarity scores using CrossEncoder
    similarity_scores = cross_encoder.predict(query_passage_pairs)

    # Rank the candidate passages based on similarity scores
    ranked_passages = [passage for _, passage in sorted(zip(similarity_scores, candidate_passages), reverse=True)]

    print(similarity_scores)
    print("CE Result")
    context=[]
    # Print the ranked passages
    for i, passage in enumerate(ranked_passages):
        context.append(passage)
        #print(f"Rank {i+1}: {passage}")

    return similarity_scores.sort(), context

def ce_doc(query,topk,passages):
    # Initialize Cross-Encoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Replace with the actual cross-encoder model name or path
    #cross-encoder/ms-marco-MiniLM-L-6-v2
    #cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    
    print("CE_DOC")
    docs=[]
    scores=[]
    for i in range(len(topk)):
        candidate_passages = [passages[j] for j in topk[i]]
        tmp_passage = {passages[j]:j for j in topk[i]}
        # Generate query-passage pairs
        query_passage_pairs = [(query[i], passage) for passage in candidate_passages]
        #print(candidate_passages)
        #print(tmp_passage)
        #if i==0:
        #    print(query_passage_pairs)
        # Compute similarity scores using CrossEncoder
        similarity_scores = cross_encoder.predict(query_passage_pairs)

        # Rank the candidate passages based on similarity scores
        ranked_passages = [passage for _, passage in sorted(zip(similarity_scores, candidate_passages), reverse=True)]
        
        scores.append(similarity_scores.sort())
        tmp=[tmp_passage[ind] for ind in ranked_passages]
        docs.append(tmp)

    return scores, docs