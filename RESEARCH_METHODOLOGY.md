# Research Methodology

## 1. Research Design

### 1.1 Research Questions

1. How does Graph RAG compare to standard RAG in terms of retrieval quality and answer accuracy?
2. What is the impact of hybrid search (semantic + keyword) on retrieval performance?
3. How does reranking affect the quality of retrieved context?
4. What are the trade-offs between retrieval speed and quality across different methods?
5. How effective is knowledge graph construction for research paper exploration?

### 1.2 Research Approach

This project follows an experimental research design with:
- **Independent Variables**: Retrieval method (RAG, Graph RAG, Advanced RAG)
- **Dependent Variables**: Retrieval quality metrics (RAGAS scores), retrieval speed
- **Control Variables**: Dataset size, embedding model, chunk size

### 1.3 Evaluation Framework

The evaluation framework is based on:
- **Quantitative Metrics**: RAGAS scores, retrieval times, precision/recall
- **Comparative Analysis**: Side-by-side comparison of methods
- **Statistical Analysis**: Mean, standard deviation, confidence intervals

## 2. Data Collection

### 2.1 Data Source

- **Primary Source**: ArXiv API
- **Categories**: cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computation and Language)
- **Selection Criteria**: Recent submissions, relevance to AI/ML domain
- **Sample Size**: 30-100 papers (configurable based on computational resources)

### 2.2 Data Preprocessing

1. **Extraction**: Title, abstract, authors, categories, publication date
2. **Cleaning**: Remove special characters, normalize text
3. **Chunking**: Split documents into fixed-size chunks with overlap
4. **Validation**: Check for completeness and quality

### 2.3 Ground Truth Creation

For evaluation queries:
- Manual annotation of expected answers
- Expert review for accuracy
- Multiple annotators for inter-annotator agreement (if available)

## 3. Knowledge Graph Construction

### 3.1 Entity Extraction Strategy

1. **Named Entity Recognition**: Extract authors, institutions
2. **Concept Extraction**: Identify AI/ML concepts using keyword matching
3. **Category Assignment**: Use ArXiv categories
4. **Keyword Extraction**: Significant terms from abstracts

### 3.2 Relationship Identification

1. **Explicit Relationships**: Author-paper, paper-category
2. **Implicit Relationships**: Similarity based on shared authors/concepts
3. **Weighting**: Assign weights based on relationship strength

### 3.3 Graph Validation

- Check for disconnected components
- Validate entity consistency
- Verify relationship types

## 4. Retrieval Methods

### 4.1 Standard RAG

**Implementation**:
- Embedding: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: ChromaDB with cosine similarity
- Retrieval: Top-k nearest neighbors

**Hypothesis**: Provides baseline performance for semantic search.

### 4.2 Graph RAG

**Implementation**:
- Initial retrieval: Standard vector search
- Graph expansion: Traverse graph from retrieved nodes
- Context aggregation: Combine initial and expanded results

**Hypothesis**: Graph expansion improves context richness for relationship queries.

### 4.3 Advanced RAG (Hybrid + Reranking)

**Implementation**:
- Hybrid search: Weighted combination of semantic (70%) and keyword (30%) search
- Reranking: Cross-encoder for relevance scoring
- Final selection: Top-k after reranking

**Hypothesis**: Hybrid search improves precision, reranking improves relevance.

## 5. Model Training

### 5.1 Fine-tuning Strategy

- **Base Model**: DialoGPT-small (computational efficiency)
- **Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Training Data**: Paper titles and abstracts formatted as sequences
- **Validation**: Hold-out validation set

### 5.2 Training Protocol

1. Data preparation: Format as question-answer pairs
2. Hyperparameter tuning: Learning rate, batch size, epochs
3. Training: Monitor loss and validation metrics
4. Evaluation: Test on held-out set

## 6. Evaluation Protocol

### 6.1 Metrics

**RAGAS Metrics**:
- Faithfulness: Measures grounding in context (0-1)
- Answer Relevancy: Measures answer quality (0-1)
- Context Precision: Precision of retrieved context (0-1)
- Context Recall: Recall of retrieved context (0-1)

**Performance Metrics**:
- Average retrieval time per query
- Throughput (queries per second)
- Memory usage

### 6.2 Experimental Procedure

1. **Setup**: Initialize all pipelines
2. **Baseline**: Run standard RAG on test queries
3. **Variants**: Run Graph RAG and Advanced RAG
4. **Collection**: Gather metrics for each method
5. **Analysis**: Statistical comparison

### 6.3 Statistical Analysis

- **Descriptive Statistics**: Mean, median, standard deviation
- **Comparative Analysis**: Pairwise comparisons
- **Significance Testing**: t-tests or non-parametric tests (if applicable)

## 7. Validation and Reliability

### 7.1 Internal Validity

- Control for confounding variables (same dataset, same queries)
- Multiple runs for reliability
- Consistent evaluation protocol

### 7.2 External Validity

- Test on different query types
- Vary dataset size
- Test with different paper categories

### 7.3 Reproducibility

- Document all hyperparameters
- Save random seeds
- Provide code and data access

## 8. Ethical Considerations

### 8.1 Data Usage

- Use publicly available ArXiv papers
- Respect copyright and citation requirements
- No personal data collection

### 8.2 Computational Resources

- Use free/open-source tools
- Optimize for efficiency
- Document resource requirements

## 9. Limitations

### 9.1 Dataset Limitations

- Limited to ArXiv (single source)
- English language only
- Recent papers (temporal bias)

### 9.2 Method Limitations

- Simple entity extraction (no advanced NER)
- Heuristic-based graph construction
- Small model for fine-tuning (computational constraints)

### 9.3 Evaluation Limitations

- Manual ground truth creation
- Limited query diversity
- Subjective quality assessment

## 10. Future Research Directions

1. **Enhanced Entity Extraction**: Use transformer-based NER models
2. **Citation Integration**: Incorporate citation networks
3. **Multi-modal RAG**: Include figures, tables, equations
4. **Larger Scale**: Test with thousands of papers
5. **User Studies**: Evaluate with real users
6. **Real-time Updates**: Incremental graph updates

## 11. Expected Outcomes

### 11.1 Quantitative Results

- RAGAS scores for each method
- Speed comparisons
- Statistical significance of differences

### 11.2 Qualitative Insights

- Use cases where Graph RAG excels
- Scenarios for hybrid search
- Trade-offs between methods

### 11.3 Contributions

- Comparative analysis of RAG methods
- Knowledge graph construction methodology
- Open-source implementation

---

This methodology ensures rigorous, reproducible research suitable for Master's level academic work.

