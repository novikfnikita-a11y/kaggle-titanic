from __future__ import annotations

import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}

COMMON_TITLES = ("Mr", "Mrs", "Miss", "Master")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Age_missing"] = df.Age.isna().astype(int)
    df["HasCabin"] = df.Cabin.notna().astype(int)

    df["Fare_log"] = np.log1p(df.Fare)

    df["FamilySize"] = df.SibSp + df.Parch + 1
    df["IsAlone"] = (df.FamilySize == 1).astype(int)
    df["Surname"] = df.Name.str.extract(r"^([^,]+),", expand=False).str.strip()

    deck = df.Cabin.str.strip().str[:1]
    df["Deck"] = deck.mask(deck.eq("") | deck.isna(), "U")

    title = df.Name.str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    canonical = title.map(TITLE_MAP).fillna(
        title.where(title.isin(COMMON_TITLES), "Rare")
    )
    df["Title"] = canonical.mask(title.isna())

    ticket = df.Ticket.astype("string").str.upper().str.strip()
    prefix = (
        ticket.str.replace(r"\d+", "", regex=True)
        .str.replace(r"[^A-Z]+", "", regex=True)
        .str.strip()
    )
    df["TicketPrefix"] = prefix.mask(prefix.eq(""), "NUMERIC").mask(
        ticket.isna() | ticket.eq("")
    )
    df["FarePerPerson"] = np.log1p(df.Fare / df.FamilySize)

    df["AgeBin"] = pd.cut(
        df.Age,
        bins=[-1, 12, 17, 29, 44, 59, 200],
        labels=["child", "teen", "young", "adult", "middle", "senior"],
    ).astype("object")
    df["FareBin"] = pd.cut(
        df.Fare,
        bins=[-1, 8, 15, 31, 600],
        labels=["low", "mid", "high", "top"],
    ).astype("object")

    df["SexPclass"] = df.Sex + "_" + df.Pclass.astype(str)

    return df
