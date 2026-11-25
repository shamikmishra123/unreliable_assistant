from assistant.execution import ReActRunner, State, TransactionPlanner, parse_intent
from assistant.contracts import CallContext, ToolError, ErrorCode
from assistant.harness import Tier
from assistant.triage import Route, classify
from assistant.verify import check_sentence
import uuid

URI = "rag://chunks/" + "c" * 64
RETRIEVED = {"c" * 64: "SKU-099 is Sterile saline 500 ml bag. Shelf life is 24 months from manufacture."}
ABS = "I can't find that in the approved knowledge vault, so I won't answer from memory."


def test_grounded_sentence_passes():
    assert check_sentence(f"Shelf life is 24 months from manufacture. [{URI}]", RETRIEVED, ABS).grounded


def test_altered_number_is_vetoed():
    verdict = check_sentence(f"Shelf life is 36 months from manufacture. [{URI}]", RETRIEVED, ABS)
    assert not verdict.grounded and "number" in verdict.reason


def test_unretrieved_citation_is_vetoed():
    other = "rag://chunks/" + "d" * 64
    assert not check_sentence(f"Shelf life is 24 months. [{other}]", RETRIEVED, ABS).grounded


def test_unsupported_claim_is_vetoed_and_abstention_passes():
    assert not check_sentence("The product is radioactive and glows at night.", RETRIEVED, ABS).grounded
    assert check_sentence(ABS, RETRIEVED, ABS).grounded


def test_triage_routes():
    assert classify("hi there").route == Route.DIRECT_ANSWER
    assert classify("What's the weather in Paris?").route == Route.REJECT_OOS
    assert classify("Release LOT-00044").route == Route.TRANSACTION_EXEC
    assert classify("Verify batch LOT-88392 safety protocol").route == Route.TRANSACTION_EXEC
    # A policy question that mentions "hold" must never become a hold request (FMEA #9)
    assert classify("What is the hold policy for LOT-00044?").route == Route.HYBRID_RAG
    assert classify("release the lot").route == Route.HYBRID_RAG  # no lot ID: no transaction
    assert classify("What is the shelf life of SKU-099?").route == Route.HYBRID_RAG


def test_intent_parsing():
    assert parse_intent("release LOT-00044").action == "release"
    assert parse_intent("quarantine LOT-00099 after a temperature alarm").action == "quarantine"
    i = parse_intent("set status to on hold for lot 44")
    assert (i.action, i.payload) == ("update_status", {"lot_id": "LOT-00044", "status": "ON_HOLD"})
    i = parse_intent("open deviation on LOT-00044: temperature excursion in cold room")
    assert i.action == "open_deviation" and i.payload["description"].startswith("temperature")
    assert parse_intent("release it").kind == "clarify"


class _AlwaysFails:
    def call(self, method, params, ctx, tier):
        raise ToolError(ErrorCode.SCHEMA_VALIDATION_FAILED, "bad")


def test_react_loop_is_bounded_and_halts_on_repeated_failure():
    ctx = lambda: CallContext(tenant_id="tenant_a", user_id="user_op1", session_id=uuid.uuid4(),  # noqa: E731
                              idempotency_key=uuid.uuid4(),
                              traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")
    runner = ReActRunner(_AlwaysFails(), ctx, Tier.TIER_2_COMPUTE, max_iterations=5)
    result = runner.run(TransactionPlanner("read LOT-00044"))
    assert result.state in (State.HALTED, State.HALTED_BREAKER)
    assert len(result.steps) <= 5
    assert runner.state == result.state
