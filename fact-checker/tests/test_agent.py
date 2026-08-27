from unittest.mock import MagicMock, patch

from rustic_ai.fact_checker.agent import (
    Claim,
    ClaimReview,
    ClaimsResponse,
    FactCheckerAgent,
    Publisher,
)


@patch("rustic_ai.fact_checker.agent.build")
@patch("rustic_ai.fact_checker.agent.spacy.load", side_effect=OSError)
def test_initializes_without_optional_spacy_model(_load, build):
    agent = FactCheckerAgent()

    build.assert_not_called()
    assert agent.service is None
    assert agent.nlp.lang == "en"
    assert "spacytextblob" in agent.nlp.pipe_names


@patch.dict("os.environ", {"GOOGLE_CLOUD_API_KEY": "test-key"})
@patch("rustic_ai.fact_checker.agent.build")
def test_uses_configured_google_api_key(build):
    agent = FactCheckerAgent.__new__(FactCheckerAgent)
    with patch("rustic_ai.fact_checker.agent.spacy.load"):
        FactCheckerAgent.__init__(agent)

    build.assert_called_once_with("factchecktools", "v1alpha1", developerKey="test-key")


def test_calculate_verdict_and_supporting_urls():
    agent = FactCheckerAgent.__new__(FactCheckerAgent)
    agent.nlp = MagicMock()
    response = ClaimsResponse(
        claims=[
            Claim(
                text="A claim",
                claimReview=[
                    ClaimReview(
                        publisher=Publisher(name="Reviewer"),
                        textualRating="False",
                        url="https://example.com/review",
                    )
                ],
            )
        ]
    )

    verdict, urls = agent.calculate_verdict_and_supporting_urls(response)

    assert verdict == "False"
    assert urls == ["https://example.com/review"]
