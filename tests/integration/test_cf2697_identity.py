"""CF-26-97 P0: bit-identity fingerprints of the backtest engine at the base commit.

``BASE_DIGESTS`` holds one sha256 per matrix entry of ``tests/_cf2697_fingerprint.py``
(strategies x symbols x sizes x seeds, plus the OOS, ATR bracket + trailing, optimizer
in-sample and walk-forward cases, and a resting-LIMIT paper-engine scenario added in P0
revision r2; P0 r3 added rising-market and fresh-bar entries, a circuit-breaker entry, a
firing trailing stop and two tied optimizer grids: 49 entries).  The literals were recorded
from UNPATCHED production code, base commit
c1b45f5f6b0e07e74b6aa263ec4635d549969f36 (``packages`` and ``apps`` unchanged), with::

    PYTHONHASHSEED=0 .venv/bin/python reports/vp2-cf2697/p0-record.py \\
        --tree <checkout at c1b45f5> --harness tests/_cf2697_fingerprint.py \\
        > reports/vp2-cf2697/p0-digests.json

(byte-identical at PYTHONHASHSEED=0 and 1, on repeated runs, and with ``--no-git --repo
<repo>`` on a ``git archive c1b45f5`` export).  NEVER regenerate them from
patched code: a changed digest means the engine result changed.  This file and the harness
are append-only from P1 on (SY-97-22).  About 70 s; the 8000-bar entry takes about 15 s.
"""

# ruff: noqa: E501  (the full sha256 literals exceed the line length)

from __future__ import annotations

import pytest

from tests._cf2697_fingerprint import capture, matrix, run_case

BASE_DIGESTS: dict[str, str] = {
    "breakout-1s-2000-s7": "c118ace36d8172498a46fe9e1eb39426a928c4eb2053c3f37b95c78459f9e4ab",
    "breakout-1s-2000-up2000": "6139096b18010c4bdd4864e0b02a4b693ec9bdcd43ac133495c7cbfa39ac5119",
    "breakout-1s-300-s7": "661a0fae440b3dac23b08a2b3e3e89278aeae939e7a1b75d2f1cfd53406566cd",
    "breakout-2s-2000-s7": "7394fc23a289f796ac6c73c05ae835842fc67fd7eaf1ccc00118377216427a22",
    "breakout-2s-300-s7": "2f39241975cecf17f459957779f3a4183bb24596a2cc9436a1dab6194831938c",
    "dca_rsi-1s-2000-s7": "af53fe5bf31805003e03b4e00332c92e9923d89df7108e27d1d978ac978b8f23",
    "dca_rsi-1s-300-s7": "992f433ccf5ffe6861a31ebc1309227de7c14415eaede239162b5c261a1fc98a",
    "dca_rsi-2s-2000-s7": "c63966c1713634aa33e7173f150311402dc61bb648943b46b7dc278da9b439d5",
    "dca_rsi-2s-300-s7": "2118bb4a631902a1f670bc08d2fca64405a19ceaf946027527da7534cf1ae782",
    "grid-1s-2000-s7": "f828cfe42f89979b32f5dd9d78dc48ef2325a3fa0a4f3d4fc01eaf4315b27747",
    "grid-1s-300-s7": "738510e175ca1a6f44d301d50d1a7984bab39d5399531890698d438bfae8c0f9",
    "grid-2s-2000-s7": "7a421eb432b0aca80d03b71b6e3c9c37cce1e4d811e8009e694625ca38f5a5db",
    "grid-2s-300-s7": "bee82f3b92fc3f187fdcad3c719e4d7fad9dc2fab878047933cd065655e1ac9e",
    "ma_crossover-1s-2000-atr-trailing": "c56eb50fc9ce0478307b9fddadc15b41c4c450c9f19527a426349d20bb6b0f54",
    "ma_crossover-1s-2000-fixed-trailing": "6de818520567b7913e429fc77a9fc07b9f81c7d7c5dfa1a62ed20082b2053169",
    "ma_crossover-1s-2000-oos": "e748211b481e01ec88f504f9165c69e97ae3a41ab53553073eeaa3211bc7a079",
    "ma_crossover-1s-2000-s7": "73290a7b435b4859d3269f4f8abbb5645cd2341d48a55872d8ceba784a9ac233",
    "ma_crossover-1s-2000-up2000": "68a9216f52f7cd8177f88840ad8273d86396b8c2ae8689021064ceb5856554e2",
    "ma_crossover-1s-300-s7": "f5c69eb5915eb0e8978e947987127a9a18ab1b67213f2fbd5c95340e19764fd6",
    "ma_crossover-1s-8000-s7": "c640dae4a92d5c26788ab59a4bcec5254ae338da83fb8f941100ad9ffdf9da72",
    "ma_crossover-2s-2000-bars11": "2ccd1b250b8186266f4a2b6c400b9ab355d6d7677d23203a59ab332bf5afe5d7",
    "ma_crossover-2s-2000-breaker": "50c9499d1fcc23c6fa5ec82f361cdfd8e240b68e0d1c92052fc04612edd2c87f",
    "ma_crossover-2s-2000-s11": "5d152624a5e37b69fc52606ff519d3aa833d90d30d6c6bc12499bdc35c572858",
    "ma_crossover-2s-2000-s7": "5c83b33e810c74a1a7d132dbf26fd6907c07d15eb1ce767f1de18b8837a4f373",
    "ma_crossover-2s-2000-up2000": "9fe8fc0f22811d5250155ff171204aaafb19bd5818e8b854c68b084d39f930cc",
    "ma_crossover-2s-300-s7": "2fada0ad7186ade826fbc265b7c6f98226ec99caae077d53ce90e254d47a8479",
    "momentum-1s-2000-s7": "6deca7a80e0f2c22f76dcd5d98d08a3f9d8cb6c52aa6c0e79a98a37f2ce2b78f",
    "momentum-1s-300-s7": "a41b80ff620505b9def73d8c8cb0449f6e279ac36b068274017dcf0749f332ee",
    "momentum-2s-2000-s7": "37556d452952840850dd40ad69203e87bafb344c39c40e04685ab72783d421d9",
    "momentum-2s-300-s7": "59c03374bc7f92946653be1c93802b68e1772f62614454bf0439bbfbc950435d",
    "optimizer-in-sample": "f934ba65e8cabc7305c1e6e7777156858f2f797fb4ae9d4af02a335ecaf7a182",
    "optimizer-tied-in-sample": "285137a6cb42fea4a480105b16c5fe6e4f6cf28c3f1e61d7f1d254f930366e48",
    "optimizer-tied-walk-forward": "ae74efa166c8405b351e8d10bdd03213ef8567b7f612431669ce1e1d2a7863b4",
    "optimizer-walk-forward": "f5b6f2d8bb90138a45b49111b61f9b84cdbffe0861940680352af9acd265ef76",
    "paper-resting-limit-orders": "41bf920f69772410d1d51a7c9d57c5890f72ae98daa4dcc0d55f94fea4e4af0b",
    "rsi-1s-2000-s7": "23b7fc315ca57cdd9bdb0094f90602067b9227e518ac4ddc5ed4197ef0091c3a",
    "rsi-1s-300-s7": "70a12ded9cfd96039a5bc6e3ea25017b5ddd5a3416c7514b97ec14f04fa5fb68",
    "rsi-2s-2000-s11": "d5a474c67c5159e557dbcd93f393593bdaae97e986a7f74cc4d96565442930d7",
    "rsi-2s-2000-s7": "e7ba7ebb35a31b5b9b4882b7d111d74ec146f1e1bb5358a82c626f80598fc7f2",
    "rsi-2s-300-s7": "58f25f37028756d8c811fb1212955b0c1b1e33630e764b68e4d1e8e910d87e39",
    "sl_tp-1s-2000-s7": "5a67d0378fe98e475dce937e2be56c94481bd24a7b46d9e69b765e651d4a3fa5",
    "sl_tp-1s-2000-up2000": "62f11e76364f356a5b44992d80354cfb72e19e1ddacaaae811dbd65df288856b",
    "sl_tp-1s-300-s7": "5b75ad6ab1f734b1a6e70d4113f33700aaa98ff81e4f15f92e3d9139ed52230a",
    "sl_tp-2s-2000-s7": "c09cd0fe48d00d72b4907b7c90d04205090f6fd7230f9c49f16c99cfc7dacd96",
    "sl_tp-2s-300-s7": "6e4e8e4f78eebf82bc84a8d8760e7c8372236a4b6a7fef9f6a1b03843cbdaee2",
    "smoke-1s-2000-s7": "b2a91e6ef5375e3272b0cc82d97596ffd9d805199560f40274787acb7f152f66",
    "smoke-1s-300-s7": "58ff24beeb1333712369762e556157ec76c277607e72e040d7c3c11bbebec8c7",
    "smoke-2s-2000-s7": "6bb34b6cf0f1c0781918b435123d3edcd08cb6a7e37af9f28f315298168477e1",
    "smoke-2s-300-s7": "c998536ea3ab1dce16291f61a946b2a852f93861d46b1310e1e7c80a5424fa09",
}


def test_literals_cover_exactly_the_matrix() -> None:
    assert sorted(BASE_DIGESTS) == sorted(matrix())


def test_matrix_has_the_required_shapes() -> None:
    cases = list(matrix().values())
    assert any(c["nsym"] == 2 for c in cases)
    assert any(c.get("trailing") and c["bracket"]["bracket_mode"] == "atr" for c in cases)
    assert any("oos" in c for c in cases)
    assert any("limit" in c for c in cases)
    assert any("risk" in c for c in cases)
    assert {c["opt"] for c in cases if "opt" in c} == {False, True}


@pytest.mark.parametrize("name", sorted(BASE_DIGESTS))
async def test_digest_equals_base(name: str) -> None:
    with capture() as cap:
        assert await run_case(name, cap) == BASE_DIGESTS[name]
