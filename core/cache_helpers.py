import os, re
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    from dotenv import load_dotenv as _ld, dotenv_values as _dv
    _ld(override=True); _ENV = _dv()
except ImportError:
    _ENV = {}
import streamlit as st

from pipeline_cache import PipelineCache

def _embedder_kwargs_from_cfg(emb_cfg: dict) -> dict:
    """
    Build kwargs cho get_embedder() từ emb_cfg dict.
    Mỗi provider chỉ nhận đúng kwargs của nó — không truyền thừa.
    """
    p  = emb_cfg["provider"]
    mn = emb_cfg["model_name"]
    ex: dict = {}
    if p == "openai":
        dims = emb_cfg.get("dimensions")
        if dims:
            ex["dimensions"] = dims
    elif p == "cohere":
        ex["input_type"] = emb_cfg.get("input_type", "search_document")
    elif p == "ollama":
        ex["base_url"] = emb_cfg.get("ollama_base_url", "http://localhost:11434")
    elif p == "huggingface":
        ex["device"] = emb_cfg.get("device", "cpu")
    return {"provider": p, "model_name": mn, **ex}


@st.cache_resource(show_spinner=False)
def _get_cached_embedder(provider: str, model_name: str, **kwargs):
    """
    Cache embedder theo provider+model — singleton per process.
    Tránh re-import torch / re-load tokenizer mỗi lần Load index.
    st.cache_resource tồn tại suốt vòng đời Streamlit server process.
    """
    from embedding.factory import get_embedder
    return get_embedder(provider=provider, model_name=model_name, **kwargs)


@st.cache_data(show_spinner=False, ttl=10)
def _cached_list_entries(_cache_dir: str = "processed_data", _version: int = 0) -> list:
    """Cache list_entries() theo version token. TTL=10s safety net."""
    from pipeline_cache import PipelineCache
    return PipelineCache(_cache_dir).list_entries()


@st.cache_data(show_spinner=False, ttl=10)
def _cached_list_pipelines(_cache_dir: str = "processed_data", _version: int = 0) -> list:
    """
    Cache list_complete_pipelines() — trả về các pipeline hoàn chỉnh.
    TTL=10s: safety net để không bao giờ hiển thị dữ liệu cũ quá 10 giây,
    ngay cả khi _version không được tăng đúng cách.
    """
    from pipeline_cache import PipelineCache
    return PipelineCache(_cache_dir).list_complete_pipelines()


def _invalidate_list_entries_cache() -> None:
    """Tăng version counter → _cached_list_entries sẽ re-scan disk ở lần gọi tiếp theo."""
    st.session_state["_cache_list_version"] = (
        st.session_state.get("_cache_list_version", 0) + 1
    )


def _resolve_autoload_pending() -> None:
    """
    Hoàn tất kết nối VDB cho autoload pending (gọi khi user ấn Run).
    Chỉ chạy 1 lần — sau đó xoá "_autoload_pending" khỏi session_state.
    """
    pending = st.session_state.pop("_autoload_pending", None)
    if pending is None:
        return
    if "vdb_result" in st.session_state:
        return  # đã connect rồi

    try:
        p               = pending["pipeline"]
        full_emb_cfg    = pending["full_emb_cfg"]
        provider        = pending["provider"]
        collection_name = pending["collection_name"]
        persist_dir     = pending["persist_dir"]
        n_vectors       = pending["n_vectors"]

        kwargs = _embedder_kwargs_from_cfg(full_emb_cfg)
        embedder = _get_cached_embedder(**kwargs)
        vector_store = _load_vector_store_readonly(provider, embedder, {
            "collection_name": collection_name,
            "persist_dir":     persist_dir,
        })
        st.session_state["vdb_result"] = {
            "vector_store":    vector_store,
            "sparse_index":    None,
            "provider":        provider,
            "collection_name": collection_name,
            "n_vectors":       n_vectors,
            "persist_dir":     persist_dir,
        }
    except Exception as e:
        st.error(f"❌ Không thể kết nối index: {e}")


def _get_chroma_client(persist_dir: str):
    """
    Trả về Chroma PersistentClient từ module-level lru_cache trong chroma_store.
    Dùng chung với ChromaStore._make_client → đảm bảo chỉ có MỘT instance
    per persist_dir trong toàn bộ process (tránh conflict settings).
    st.cache_resource không cần thiết vì lru_cache đã singleton ở process level.
    """
    from vector_db.chroma_store import _get_or_create_chroma_client
    return _get_or_create_chroma_client(persist_dir)

def _load_vector_store_readonly(provider: str, embedder, vdb_kwargs: dict):
    """
    Load một vector store đã tồn tại mà KHÔNG insert thêm document nào.
    Tránh crash Chroma.from_documents([]) khi chunks=[].
    """
    lc_embedder = embedder.embedder if hasattr(embedder, "embedder") else embedder

    if provider == "chroma":
        from langchain_chroma import Chroma
        import chromadb
        from chromadb.config import Settings
        persist_dir = vdb_kwargs.get("persist_dir") or vdb_kwargs.get("persist_directory", "./storage/chroma")
        collection_name = vdb_kwargs["collection_name"]
        # cache_resource giữ client sống suốt session, tránh GC destroy RustBindingsAPI
        chroma_client = _get_chroma_client(persist_dir)
        return Chroma(
            client             = chroma_client,
            collection_name    = collection_name,
            embedding_function = lc_embedder,
        )

    if provider == "faiss":
        from langchain_community.vectorstores import FAISS
        from pathlib import Path as _P
        persist_dir     = vdb_kwargs.get("persist_dir") or vdb_kwargs.get("persist_directory", "./storage/faiss")
        collection_name = vdb_kwargs["collection_name"]
        idx_dir  = _P(persist_dir)
        idx_file = idx_dir / "index.faiss"
        if idx_file.exists():
            return FAISS.load_local(
                folder_path                     = str(idx_dir),
                embeddings                      = lc_embedder,
                allow_dangerous_deserialization = True,
            )
        raise FileNotFoundError(
            f"FAISS index not found: {idx_file}\n"
            f"Expected: {idx_dir}/index.faiss và {idx_dir}/index.pkl"
        )

    if provider == "lancedb":
        import lancedb
        from langchain_community.vectorstores import LanceDB
        persist_dir = vdb_kwargs.get("persist_dir") or vdb_kwargs.get("persist_directory", "./storage/lancedb")
        db = lancedb.connect(persist_dir)
        return LanceDB(
            connection      = db,
            embedding       = lc_embedder,
            table_name      = vdb_kwargs["collection_name"],
        )

    if provider == "qdrant":
        from langchain_qdrant import Qdrant
        return Qdrant.from_existing_collection(
            embedding       = lc_embedder,
            collection_name = vdb_kwargs["collection_name"],
            url             = vdb_kwargs.get("url", "http://localhost:6333"),
            api_key         = vdb_kwargs.get("api_key"),
        )

    if provider == "weaviate":
        import weaviate
        from langchain_weaviate import WeaviateVectorStore
        client = weaviate.connect_to_local(
            host    = vdb_kwargs.get("url", "http://localhost:8080").replace("http://","").split(":")[0],
            port    = int(vdb_kwargs.get("url","http://localhost:8080").rsplit(":",1)[-1]) if ":" in vdb_kwargs.get("url","") else 8080,
            api_key = weaviate.auth.ApiKey(vdb_kwargs["api_key"]) if vdb_kwargs.get("api_key") else None,
        )
        return WeaviateVectorStore(client=client, index_name=vdb_kwargs["collection_name"], text_key="text", embedding=lc_embedder)

    if provider == "pgvector":
        from langchain_postgres import PGVector
        return PGVector(
            embeddings         = lc_embedder,
            collection_name    = vdb_kwargs["collection_name"],
            connection         = vdb_kwargs.get("connection_string", ""),
        )

    if provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name  = vdb_kwargs["collection_name"],
            embedding   = lc_embedder,
            api_key     = vdb_kwargs.get("api_key", ""),
        )

    raise ValueError(f"_load_vector_store_readonly: provider '{provider}' không được hỗ trợ.")


