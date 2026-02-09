from scripts.pipeline.policy import policy_period


def test_policy_period_mapping():
    assert policy_period(2008) == "Pre-GreenDeal"
    assert policy_period(2014) == "GreenDeal-ECO1"
    assert policy_period(2017) == "ECO2"
    assert policy_period(2019) == "MEES"
    assert policy_period(2023) == "Post-Strategy"
    assert policy_period(1999) == "Other"
