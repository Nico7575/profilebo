from profilebo.rng import derive_seed


def test_derive_seed_is_deterministic_and_in_serpent_range():
    s = derive_seed(42, "bo_016", "transport")
    assert s == derive_seed(42, "bo_016", "transport")
    assert 1 <= s <= 2**31 - 1
    others = {derive_seed(42, "bo_016", "disperser"), derive_seed(42, "bo_017", "transport"),
              derive_seed(43, "bo_016", "transport")}
    assert s not in others and len(others) == 3
