import polars as pl


def load_budget() -> pl.DataFrame:
    return (
        pl.concat(
            [
                pl.read_json("budget2024.json", infer_schema_length=None),
                pl.read_json("budget2025.json", infer_schema_length=None),
            ]
        )
        .unnest("data")
        .unnest("budgetVsActualV2")
    )


def extract_cashflow(cashflow: pl.Series):
    return (
        cashflow.explode()
        .struct.unnest()
        .select("subAccounts", headline_category="category")
        .explode("subAccounts")
        .unnest("subAccounts")
        .select(
            "headline_category",
            "category",
            pl.col("annual").struct.field("transactions"),
        )
        .explode("transactions")
        .unnest("transactions")
        .with_columns(pl.col("date").str.to_date())
        .drop(["transactionType", "__typename"])
        .drop_nulls()
    )

def aggregate_cashflow(cashflow: pl.DataFrame) -> pl.DataFrame:
    return cashflow.group_by(
        year=pl.col("date").dt.year(), month=pl.col("date").dt.month()
    ).agg(pl.col("amount").sum())