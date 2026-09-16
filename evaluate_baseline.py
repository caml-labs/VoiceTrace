import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

from FlagEmbedding import FlagAutoModel
from embedder import ECAPASpeakerEmbedder, SpeakerEmbedder
from segmenter import WhisperXSegmenter

import datasets
import numpy as np


@dataclass
class EvalPair:
    source: str
    query_audio_path: str
    query_text: str
    doc_audio_path: str
    query_id: str
    positive_doc_id: str

EMBEDDING_MODELS = {
    "ecapa": (ECAPASpeakerEmbedder, "speechbrain/spkrec-ecapa-voxceleb"),
}


def resolve_audio_path(dataset_root: str, relative_or_abs: str) -> str:
    if os.path.isabs(relative_or_abs):
        return relative_or_abs
    return os.path.normpath(os.path.join(dataset_root, relative_or_abs))


def pick_eval_split(ds: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    if isinstance(ds, datasets.Dataset):
        return ds
    if "test" in ds:
        return ds["test"]
    first_key = next(iter(ds.keys()))
    return ds[first_key]


def load_eval_pairs(
    single_dataset_path: str,
    multi_dataset_path: str,
) -> List[EvalPair]:
    pairs: List[EvalPair] = []

    single_ds = datasets.load_dataset("parquet", data_files=f"{single_dataset_path}/data.parquet")
    for ind, item in enumerate(pick_eval_split(single_ds)):
        query_id = f"single:{ind}"
        doc_id = f"single:{ind}"
        pairs.append(
            EvalPair(
                source="single",
                query_audio_path=resolve_audio_path(single_dataset_path, item["query_audio_path"]),
                query_text=str(item["query_text"]).strip(),
                doc_audio_path=resolve_audio_path(single_dataset_path, item["document_audio_path"]),
                query_id=query_id,
                positive_doc_id=doc_id,
            )
        )

    multi_ds = datasets.load_dataset("parquet", data_files=f"{multi_dataset_path}/data.parquet")
    for ind, item in enumerate(pick_eval_split(multi_ds)):
        query_id = f"multi:{ind}"
        doc_id = f"multi:{ind}"
        pairs.append(
            EvalPair(
                source="multi",
                query_audio_path=resolve_audio_path(multi_dataset_path, item["query_audio_path"]),
                query_text=str(item["query_text"]).strip(),
                doc_audio_path=resolve_audio_path(multi_dataset_path, item["document_audio_path"]),
                query_id=query_id,
                positive_doc_id=doc_id,
            )
        )

    return pairs


def recall_at_k(rankings: Dict[str, List[str]], query_to_positive: Dict[str, str], k: int) -> float:
    hit = 0
    total = 0
    for qid, ranked_docs in rankings.items():
        pos = query_to_positive[qid]
        total += 1
        if pos in ranked_docs[:k]:
            hit += 1
    if total == 0:
        return 0.0
    return hit / total


def _dcg_at_k_binary(ranked_docs: List[str], positive_doc_id: str, k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_docs[:k], start=1):
        if did == positive_doc_id:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def _idcg_at_k_single_relevant(k: int) -> float:
    return 1.0 / math.log2(2)


def ndcg_at_k(rankings: Dict[str, List[str]], query_to_positive: Dict[str, str], k: int) -> float:
    if not rankings:
        return 0.0
    idcg = _idcg_at_k_single_relevant(k)
    if idcg <= 0:
        return 0.0
    total = 0.0
    for qid, ranked_docs in rankings.items():
        pos = query_to_positive[qid]
        total += _dcg_at_k_binary(ranked_docs, pos, k) / idcg
    return total / len(rankings)


def _safe_text_embedding(text_embedder: FlagAutoModel, text: str, fallback_dim: int) -> np.ndarray:
    cleaned = text.strip()
    if not cleaned:
        return np.zeros((fallback_dim,), dtype=np.float32)
    emb = text_embedder.encode([cleaned])[0]
    return np.asarray(emb, dtype=np.float32).reshape(-1)


def evaluate_retrieval(
    pairs: Sequence[EvalPair],
    segmenter: WhisperXSegmenter,
    speaker_embedder: SpeakerEmbedder,
    text_embedder: FlagAutoModel,
    rerank_top_k: int,
) -> Dict[str, Dict[str, float]]:
    doc_id_to_path = {}
    query_id_to_path = {}
    query_id_to_text = {}
    query_to_positive: Dict[str, str] = {}
    query_to_source: Dict[str, str] = {}
    for p in pairs:
        doc_id_to_path[p.positive_doc_id] = p.doc_audio_path
        query_id_to_path[p.query_id] = p.query_audio_path
        query_id_to_text[p.query_id] = p.query_text
        query_to_positive[p.query_id] = p.positive_doc_id
        query_to_source[p.query_id] = p.source

    doc_ids = sorted(doc_id_to_path.keys())
    query_ids = sorted(query_id_to_path.keys())
    text_emb_dim = int(np.asarray(text_embedder.encode(["text-dim-probe"])[0]).reshape(-1).shape[0])

    query_speaker_embs: Dict[str, np.ndarray] = {}
    query_text_embs: Dict[str, np.ndarray] = {}
    for qid in query_ids:
        q_path = query_id_to_path[qid]
        q_text = query_id_to_text[qid]
        query_speaker_embs[qid] = speaker_embedder.embed_file(q_path)
        query_text_embs[qid] = _safe_text_embedding(text_embedder, q_text, text_emb_dim)
        print(f"Query speaker embedding [{qid}]: {query_speaker_embs[qid][:5]}")
        print(f"Query text [{qid}]: {q_text[:120]}")

    doc_seg_speaker_embs: Dict[str, np.ndarray] = {}
    doc_text_embs: Dict[str, np.ndarray] = {}
    for did in doc_ids:
        d_path = doc_id_to_path[did]
        segments, seg_texts = segmenter.segment_audio_with_text(d_path)
        seg_emb_list: List[np.ndarray] = []
        for seg_wav in segments:
            seg_emb_list.append(speaker_embedder.embed_mono_numpy(seg_wav, sample_rate=16000))
        if seg_emb_list:
            print(f"Doc Segment embeddings [{did}]: {seg_emb_list[-1][:5]}")
            doc_seg_speaker_embs[did] = np.stack(seg_emb_list, axis=0)
        else:
            # Fallback when segmenter returns no segments: embed whole file as single segment
            emb = speaker_embedder.embed_file(d_path)
            doc_seg_speaker_embs[did] = emb.reshape(1, -1) if emb.ndim == 1 else emb
            print(f"Doc embedding (no segments, whole file) [{did}]: {doc_seg_speaker_embs[did][0][:5]}")
        doc_text = " ".join(x for x in seg_texts if x).strip()
        doc_text_embs[did] = _safe_text_embedding(text_embedder, doc_text, text_emb_dim)
        print(f"Doc text [{did}]: {doc_text[:120]}")

    def compute_subset_metrics(
        subset_query_ids: List[str],
        allowed_doc_source: str = None,  # "single", "multi", or None for mixed
    ) -> Dict[str, float]:
        subset_rankings = {}
        subset_pos = {}
        for qid in subset_query_ids:
            # Determine which document sources should be considered for this query
            if allowed_doc_source is None:
                # Use all docs
                candidate_dids = [did for did in doc_ids]
            else:
                candidate_dids = [
                    did for did in doc_ids
                    if did.startswith(f"{allowed_doc_source}:")
                ]
            q_spk_emb = query_speaker_embs[qid]
            q_txt_emb = query_text_embs[qid]
            stage1_scores = []
            for did in candidate_dids:
                seg_matrix = doc_seg_speaker_embs[did]
                scores = np.dot(seg_matrix, q_spk_emb)
                best_score = float(np.nanmax(scores))
                stage1_scores.append((did, best_score))
            stage1_scores.sort(key=lambda x: x[1], reverse=True)

            k = min(max(1, rerank_top_k), len(stage1_scores))
            topk = stage1_scores[:k]
            reranked = []
            for did, _ in topk:
                d_txt_emb = doc_text_embs[did]
                bge_score = float(np.dot(q_txt_emb, d_txt_emb))
                reranked.append((did, bge_score))
            reranked.sort(key=lambda x: x[1], reverse=True)

            reranked_doc_ids = [x[0] for x in reranked]
            remainder_doc_ids = [x[0] for x in stage1_scores[k:]]
            subset_rankings[qid] = reranked_doc_ids + remainder_doc_ids
            subset_pos[qid] = query_to_positive[qid]
            
        return {
            "R@1": recall_at_k(subset_rankings, subset_pos, 1),
            "R@3": recall_at_k(subset_rankings, subset_pos, 3),
            "R@5": recall_at_k(subset_rankings, subset_pos, 5),
            "R@10": recall_at_k(subset_rankings, subset_pos, 10),
            "NDCG@1": ndcg_at_k(subset_rankings, subset_pos, 1),
            "NDCG@3": ndcg_at_k(subset_rankings, subset_pos, 3),
            "NDCG@5": ndcg_at_k(subset_rankings, subset_pos, 5),
            "NDCG@10": ndcg_at_k(subset_rankings, subset_pos, 10),
            "num_queries": float(len(subset_query_ids)),
        }

    single_queries = [qid for qid in query_ids if query_to_source[qid] == "single"]
    multi_queries = [qid for qid in query_ids if query_to_source[qid] == "multi"]
    mixed_queries = list(query_ids)
    return {
        "single": compute_subset_metrics(single_queries, allowed_doc_source="single"),
        "multi": compute_subset_metrics(multi_queries, allowed_doc_source="multi"),
        "mixed": compute_subset_metrics(mixed_queries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speaker retrieval evaluation script.")
    parser.add_argument(
        "--single-dataset-path",
        type=str,
        required=True,
        help="Path to single-speaker dataset root.",
    )
    parser.add_argument(
        "--multi-dataset-path",
        type=str,
        required=True,
        help="Path to multi-speaker dataset root.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="eval_results.json",
        help="Where to write evaluation metrics json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for WhisperX model.",
    )
    parser.add_argument(
        "--whisperx-model",
        type=str,
        default="large-v2",
        help="WhisperX ASR model name.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        help="WhisperX compute_type, e.g. float16/int8.",
    )
    parser.add_argument(
        "--embedding-model-type",
        type=str,
        default="ecapa",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Speaker embedding model type. Only 'ecapa' (SpeechBrain ECAPA-TDNN) is supported. Default: ecapa.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help=(
            "Override the default model name for the chosen embedding model type. "
            "Default: ecapa → 'speechbrain/spkrec-ecapa-voxceleb'."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=25,
        help="Stage-1 top-k docs to rerank using BGE text embeddings.",
    )
    parser.add_argument(
        "--bge-model-name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="BGE model name for text embedding.",
    )
    parser.add_argument(
        "--bge-use-fp16",
        action="store_true",
        help="Enable fp16 for BGE model.",
    )
    parser.add_argument(
        "--bge-query-instruction",
        type=str,
        default="Represent this sentence for searching relevant passages:",
        help="Instruction prefix used by BGE for query embedding.",
    )
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    pairs = load_eval_pairs(
        single_dataset_path=args.single_dataset_path,
        multi_dataset_path=args.multi_dataset_path,
    )

    print("Evaluation mode: all")
    print(f"Loaded eval pairs: {len(pairs)}")

    segmenter = WhisperXSegmenter(
        device=args.device,
        model_name=args.whisperx_model,
        compute_type=args.compute_type,
    )

    # --- build speaker embedder ---
    embedder_cls, default_model_name = EMBEDDING_MODELS[args.embedding_model_type]
    model_name = args.embedding_model_name or default_model_name
    embedder = embedder_cls(model_name, device=args.device)
    print(f"Embedding model: {args.embedding_model_type} ({model_name})")
    print("Single-doc embedding mode: segmented")
    print(f"Rerank top-k: {args.rerank_top_k}")

    text_embedder = FlagAutoModel.from_finetuned(
        args.bge_model_name,
        query_instruction_for_retrieval=args.bge_query_instruction,
        use_fp16=args.bge_use_fp16,
    )
    print(f"BGE model: {args.bge_model_name}")

    metrics = evaluate_retrieval(
        pairs=pairs,
        segmenter=segmenter,
        speaker_embedder=embedder,
        text_embedder=text_embedder,
        rerank_top_k=args.rerank_top_k,
    )

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Evaluation done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {args.output_json}")
    total_elapsed = time.perf_counter() - total_start
    print(f"Total evaluation time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
