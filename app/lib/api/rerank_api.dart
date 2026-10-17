// Reranking over the local API's core, in the shape the common clients send:
// Jina's and Cohere's rerank APIs, as self-hosted servers (vLLM, llama.cpp)
// serve them at POST /v1/rerank (also /rerank and /v2/rerank here).
//
//   request:  {"model", "query", "documents": [string | {"text": string}],
//              "top_n"?, "return_documents"? (default true)}
//   response: {"id", "model", "results": [{"index", "relevance_score",
//              "document": {"text"}}], "usage": {"total_tokens"}}
//
// Results are sorted by relevance_score, highest first, and cut to top_n;
// `index` is the document's position in the request. Scores come from the
// loaded reranker (its ranking head: P(relevant) in (0, 1)); any other model
// is refused.
import 'api_core.dart';
import 'local_api.dart';

class RerankApi {
  RerankApi(this.core);
  final ApiCore core;

  /// The query, the documents' texts, top_n (null: all) and whether documents are echoed in the results.
  static (String, List<String>, int?, bool) request(Map<String, Object?> body) {
    final query = body['query'];
    if (query is! String || query.isEmpty) throw const ApiError(400, 'query is required: a non-empty string');
    final raw = body['documents'];
    if (raw is! List) throw const ApiError(400, 'documents is required: a list of strings or {"text": ...} objects');
    final documents = [
      for (final d in raw)
        switch (d) {
          String text => text,
          {'text': String text} => text,
          _ => throw const ApiError(400, 'each document must be a string or a {"text": ...} object'),
        },
    ];
    if (documents.isEmpty) throw const ApiError(400, 'documents must not be empty');
    final topN = body['top_n'];
    if (topN != null && (topN is! int || topN <= 0)) throw const ApiError(400, 'top_n must be a positive integer');
    final echo = body['return_documents'];
    if (echo != null && echo is! bool) throw const ApiError(400, 'return_documents must be true or false');
    return (query, documents, topN as int?, echo as bool? ?? true);
  }

  Future<void> rerank(Map<String, Object?> body, ApiConnection conn) async {
    final (query, documents, topN, echo) = request(body);
    final result = await core.rerank(query, documents, gone: conn.gone);
    final order = List<int>.generate(documents.length, (i) => i)
      // Highest first; equal scores keep the request's order.
      ..sort((a, b) {
        final byScore = result.scores[b].compareTo(result.scores[a]);
        return byScore != 0 ? byScore : a.compareTo(b);
      });
    return conn.json(200, {
      'id': responseId('rerank-'),
      'model': core.backend.modelName,
      'results': [
        for (final i in order.take(topN ?? documents.length))
          {
            'index': i,
            'relevance_score': result.scores[i],
            if (echo) 'document': {'text': documents[i]},
          },
      ],
      'usage': {'total_tokens': result.tokens},
    });
  }
}
