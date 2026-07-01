"""HyDE (Hypothetical Document Embeddings) query expansion for financial QA."""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

_HYDE_PROMPT = PromptTemplate.from_template(
    "Generate a concise hypothetical answer (2-3 sentences) to the following "
    "financial question, as if you were an expert analyst reading a 10-K or 10-Q filing.\n\n"
    "Question: {question}\n\n"
    "Hypothetical Answer:"
)


def hyde_retrieve(
    question: str,
    vectorstore: Any,
    llm: BaseLanguageModel,
    k: int = 5,
) -> list[Document]:
    """Generate a hypothetical answer, embed it, then retrieve similar chunks.

    The hypothetical answer bridges the vocabulary gap between the query style
    and the dense financial prose found in SEC filings.
    """
    chain = _HYDE_PROMPT | llm | StrOutputParser()
    hypothetical_answer = chain.invoke({"question": question})
    return vectorstore.similarity_search(hypothetical_answer, k=k)


class HyDERetriever(BaseRetriever):
    """LangChain BaseRetriever wrapper around :func:`hyde_retrieve`."""

    vectorstore: Any
    llm: Any
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return hyde_retrieve(query, self.vectorstore, self.llm, self.k)
