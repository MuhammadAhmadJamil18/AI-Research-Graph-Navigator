"""
Comprehensive Evaluation Script
Run this to generate evaluation results for the Master's project
"""
import os
import sys
from src.data_ingestion import ArXivIngester
from src.rag_pipeline import RAGPipeline
from src.graph_builder import KnowledgeGraphBuilder
from src.graph_rag import GraphRAGPipeline
from src.advanced_evaluation import AdvancedEvaluator
from src.experiment_tracker import ExperimentTracker

# Sample evaluation queries
EVALUATION_QUERIES = [
    "What are the main approaches to transformer architectures?",
    "How do large language models handle few-shot learning?",
    "What is the difference between supervised and unsupervised learning?",
    "How do neural networks learn from data?",
    "What are the applications of computer vision?",
    "Explain attention mechanisms in deep learning",
    "What is transfer learning and how is it used?",
    "How do generative adversarial networks work?",
    "What are the challenges in natural language processing?",
    "Explain reinforcement learning algorithms"
]


def main():
    """Run comprehensive evaluation"""
    
    print("=" * 60)
    print("AI Research Graph Navigator - Comprehensive Evaluation")
    print("=" * 60)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    exp_id = tracker.start_experiment(
        experiment_name="comprehensive_evaluation",
        description="Full system evaluation with RAG, Graph RAG, and advanced metrics",
        hyperparameters={
            "num_papers": 50,
            "top_k": 5,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )
    
    print(f"\nExperiment ID: {exp_id}")
    print("\nStep 1: Loading data...")
    
    # Load papers
    ingester = ArXivIngester()
    papers = ingester.load_papers()
    
    if not papers:
        print("No papers found. Please fetch papers first using the Streamlit app.")
        return
    
    print(f"Loaded {len(papers)} papers")
    tracker.log_metric("num_papers", len(papers))
    
    print("\nStep 2: Initializing RAG pipeline...")
    
    # Initialize RAG
    rag_pipeline = RAGPipeline()
    
    # Check if vector DB exists, if not, build it
    if rag_pipeline.collection.count() == 0:
        print("Building vector database...")
        all_chunks = []
        for paper in papers:
            chunks = ingester.chunk_paper(paper)
            all_chunks.extend(chunks)
        rag_pipeline.add_documents(all_chunks)
        print(f"Added {len(all_chunks)} chunks to vector database")
    else:
        print(f"Vector database already contains {rag_pipeline.collection.count()} chunks")
    
    tracker.log_metric("vector_db_size", rag_pipeline.collection.count())
    
    print("\nStep 3: Building knowledge graph...")
    
    # Build graph
    graph_builder = KnowledgeGraphBuilder()
    if not graph_builder.load_graph():
        print("Building knowledge graph...")
        graph_builder.build_graph(papers)
        graph_builder.save_graph()
    
    graph_stats = graph_builder.get_statistics()
    print(f"Graph: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    tracker.log_result("graph_statistics", graph_stats)
    
    print("\nStep 4: Initializing Graph RAG...")
    
    # Initialize Graph RAG
    graph_rag_pipeline = GraphRAGPipeline(graph_builder, rag_pipeline)
    
    print("\nStep 5: Running comprehensive evaluation...")
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(rag_pipeline, graph_rag_pipeline)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report(
        queries=EVALUATION_QUERIES,
        output_file=f"results/evaluation_report_{exp_id}.json"
    )
    
    # Log key metrics
    if "ragas" in report["evaluation_results"]:
        ragas = report["evaluation_results"]["ragas"]
        for metric, value in ragas.items():
            if isinstance(value, (int, float)):
                tracker.log_metric(f"ragas_{metric}", value)
    
    # Generate visualizations
    print("\nStep 6: Generating visualizations...")
    evaluator.visualize_results(report, output_dir=f"results/plots_{exp_id}")
    
    # Log results
    tracker.log_result("evaluation_report", report)
    
    # End experiment
    tracker.end_experiment(
        summary=f"Comprehensive evaluation completed. RAGAS average: {report['evaluation_results'].get('ragas', {}).get('average_score', 0):.3f}"
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - Evaluation report: results/evaluation_report_{exp_id}.json")
    print(f"  - Visualizations: results/plots_{exp_id}/")
    print(f"  - Experiment data: experiments/{exp_id}/")
    print("\nKey Metrics:")
    
    if "ragas" in report["evaluation_results"]:
        ragas = report["evaluation_results"]["ragas"]
        print(f"  Faithfulness: {ragas.get('faithfulness', 0):.3f}")
        print(f"  Answer Relevancy: {ragas.get('answer_relevancy', 0):.3f}")
        print(f"  Context Precision: {ragas.get('context_precision', 0):.3f}")
        print(f"  Context Recall: {ragas.get('context_recall', 0):.3f}")
        print(f"  Average Score: {ragas.get('average_score', 0):.3f}")


if __name__ == "__main__":
    main()

