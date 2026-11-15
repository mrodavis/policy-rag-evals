from fastapi.testclient import TestClient

from policy_rag.api import create_app
from policy_rag.generator import Draft, ExtractiveGenerator
from policy_rag.pipeline import Pipeline


class Scripted:
    """A generator that returns a fixed draft and records what it was asked."""

    name = "scripted"
    model = "scripted"

    def __init__(self, draft=None, cite_top=False):
        self.draft_value, self.cite_top, self.questions = draft, cite_top, []

    def draft(self, question, hits):
        self.questions.append(question)
        d = self.draft_value
        if self.cite_top:
            d = d.model_copy(update={"citations": [hits[0].chunk.id]})
        return d


class MustNotBeCalled:
    name = model = "never"

    def draft(self, question, hits):
        raise AssertionError("generator should not be called")


def test_unrelated_questions_are_refused_without_a_model_call(index):
    ans = Pipeline(index, MustNotBeCalled()).ask("What will the weather be in Miami tomorrow?")
    assert ans.status == "refused"


def test_a_superseded_figure_is_blocked_even_when_cited(index):
    wrong = Draft(answer="The Voyager annual fee is $95.", citations=[], refused=False)
    ans = Pipeline(index, Scripted(wrong, cite_top=True)).ask("What is the Voyager annual fee?")
    assert ans.status == "blocked"
    assert ans.violations == ["ungrounded_numbers:95"]
    assert "$95" not in ans.answer


def test_injected_partner_instruction_cannot_reach_an_agent(index):
    p = Pipeline(index, ExtractiveGenerator())
    pipe_hits = p.retriever.search("lounge visits partner announcement", k=5)
    partner_id = next(h.chunk.id for h in pipe_hits if h.chunk.trust == "partner")
    injected = Draft(answer="Good news: the Voyager annual fee is waived for life.", citations=[partner_id], refused=False)
    ans = Pipeline(index, Scripted(injected)).ask("Voyager lounge visits partner announcement")
    assert ans.status == "blocked"
    assert "account_terms_from_partner_content" in ans.violations


def test_card_numbers_never_reach_the_model(index):
    gen = Scripted(Draft(answer="A late fee of $32 applies.", refused=False), cite_top=True)
    ans = Pipeline(index, gen).ask("Card 4111 1111 1111 1111: what is the late payment fee?")
    assert "4111" not in gen.questions[0].replace("ending 1111", "")
    assert ans.redactions == ["card_number"]
    assert ans.status == "answered" and ans.citations[0]["doc_id"] == "fee-schedule-2026"


def test_answers_carry_citations_and_the_index_version(index):
    ans = Pipeline(index, ExtractiveGenerator()).ask("What is the annual fee on the Voyager Card?")
    assert ans.status == "answered" and "$125" in ans.answer
    assert ans.citations[0]["chunk_id"] == "fee-schedule-2026#annual-fees"
    assert ans.index_version == index.version


def test_api(index):
    client = TestClient(create_app(Pipeline(index, ExtractiveGenerator())))
    health = client.get("/health").json()
    assert health["index_version"] == index.version and "fee-schedule-2025" not in health["documents"]
    res = client.post("/ask", json={"question": "How long does a cardmember have to dispute a charge?"}).json()
    assert res["status"] == "answered" and "60 days" in res["answer"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
