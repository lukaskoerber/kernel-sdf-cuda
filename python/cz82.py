"""The cz82 characteristics list, copied verbatim from
reference/ck_pca_cuda_v_aug25.py (lines 657-739).

Pinned here so the C++/CUDA pipeline and validation share one source of truth.
"""

CZ82 = [
    "Accruals", "AssetGrowth", "Beta", "BetaFP", "BetaTailRisk", "BidAskSpread",
    "BMdec", "BookLeverage", "Cash", "CashProd", "CBOperProf", "CF", "cfp",
    "ChEQ", "ChInv", "ChInvIA", "CompEquIss", "CompositeDebtIssuance",
    "Coskewness", "DelCOA", "DelCOL", "DelFINL", "DelLTI", "DelNetFin",
    "EarningsSurprise", "EBM", "EntMult", "EP", "EquityDuration", "GP",
    "grcapx", "GrLTNOA", "GrSaleToGrInv", "Herf", "High52", "hire", "IdioVol3F",
    "Illiquidity", "IndMom", "IntMom", "Investment", "InvestPPEInv", "Leverage",
    "LRreversal", "MaxRet", "MeanRankRevGrowth", "Mom12m", "Mom12mOffSeason",
    "Mom6m", "MomOffSeason", "MomOffSeason06YrPlus", "MomSeason",
    "MomSeason06YrPlus", "MomSeasonShort", "MRreversal", "NetDebtFinance",
    "NetEquityFinance", "NOA", "OPLeverage", "PriceDelayRsq", "PriceDelaySlope",
    "PriceDelayTstat", "RDS", "ResidualMomentum", "ReturnSkew", "roaq", "RoE",
    "ShareIss1Y", "Size", "SP", "STreversal", "Tax", "TotalAccruals",
    "TrendFactor", "VarCF", "VolMkt", "VolSD", "VolumeTrend", "XFIN",
    "zerotrade6M", "InvGrowth", "OperProf",
]

assert len(CZ82) == 82, f"expected 82 chars, got {len(CZ82)}"
